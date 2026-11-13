"""run_ptb.py: evaluation driver for the PTB Diagnostic ECG Database.

Runs both QTV paths over one record per patient and appends one row per
(record, segment length, method, filter) to a CSV, covering the group
comparison MI vs. healthy on the full recording and the segment length
comparison (10 / 15 / 30 / 60 s) on the same patients. Computation only,
analysis happens separately on the CSV. The run is resumable: combinations
already present in the CSV are skipped.

Usage:
    python -m src.runners.run_ptb --limit 5            # smoke test, 5 patients per group
    python -m src.runners.run_ptb                      # full run
    python -m src.runners.run_ptb --segments full 10   # only selected segment lengths
"""

import argparse
import csv
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import numpy as np
import wfdb

from src.pipeline.loaders import Record, load_ptb_record, _group_from_header
from src.pipeline.metrics import compute_metrics
from src.pipeline.preprocessing import FILTER_CONFIGS
from src.pipeline.qtv_neurokit import compute_qt_intervals as nk_intervals
from src.pipeline.qtv_i2dsw import I2DSW_AVAILABLE, compute_qt_intervals as i2dsw_intervals

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = ROOT / "data" / "ptbdb"
OUT_PATH = ROOT / "results" / "ptb_full_v2.csv"
REVIEW_CSV = ROOT / "results" / "manual_review.csv"

LEAD = "ii"                      # lead II recommended for QTV, or the lead with the highest T amplitude (Baumert 2016)
GROUPS = ("MI", "healthy")
METHODS = {"neurokit": nk_intervals}
if I2DSW_AVAILABLE:  # i2DSW core not included in this repo, see src/kammler/i2DSW.py
    METHODS["i2dsw"] = i2dsw_intervals

# records shorter than this are dropped, so the segment length comparison runs
# on the same patients at every length
MIN_DURATION_S = 110.0

SEGMENT_LENGTHS = ("full", 60, 30, 15, 10)
SEGMENT_OFFSET_S = 0.0           # segments are cut from the start of the recording

APPLY_FILTER = True              # real ECG: high-pass per Schmidt 2018 Sec. 4.1

# both filter configurations are run on every cell, see preprocessing.FILTER_CONFIGS
FILTERS = tuple(FILTER_CONFIGS)

# records whose accepted beat count falls below this are flagged, not dropped
MIN_BEATS = 8

FIELDS = [
    # identity / configuration
    "record_id", "subject_id", "group", "age", "sex", "method", "filter_cfg",
    "segment_length_s", "record_duration_s",
    # beat funnel, in the order beats are lost
    "n_r_peaks", "n_beats_raw", "threshold_uv", "n_kept", "n_total",
    "n_failed", "n_adapt_rejected", "n_implausible", "n_t_amp_nan",
    "n_iso_fallback", "n_rejected", "n_beats",
    "below_min_beats",
    # results
    "sdqt", "sdrt", "qtvi",
    # potential confounders, needed to interpret a group difference in sdqt
    "qt_mean", "rr_mean", "sdnn", "t_amp_median",
    # quality control
    "adapt_distance_median", "adapt_distance_max",
    "align_shift_mean_ms", "align_shift_max_ms", "align_corr_mean",
    "n_align_unusable",
    # bookkeeping
    "elapsed_s", "status", "error",
]


def _age_sex(comments: list[str]) -> tuple[str, str]:
    """Age and sex from PTB header comments, as strings ("n/a" when absent)."""
    age = sex = "n/a"
    for line in comments:
        low = line.lower()
        if low.startswith("age:"):
            age = line.split(":", 1)[1].strip()
        elif low.startswith("sex:"):
            sex = line.split(":", 1)[1].strip()
    return age, sex


def _usable_records(review_csv: Path) -> set[str]:
    """Record ids the manual review marked usable with upright QRS and T wave."""
    with review_csv.open(newline="") as f:
        return {r["record_id"] for r in csv.DictReader(f)
                if r["decision"] == "usable"
                and r["qrs_polarity"] == "1" and r["t_polarity"] == "1"}


def select_records(root: Path = DATA_ROOT, review_csv: Path = REVIEW_CSV,
                   lead: str = LEAD, groups=GROUPS,
                   min_duration_s: float = MIN_DURATION_S,
                   limit_per_group: int | None = None) -> list[tuple[Path, str, float]]:
    """One usable record per patient.

    Usability is applied before the per-patient choice. Among several usable
    recordings the first in sorted order wins.
    """
    usable = _usable_records(review_csv)
    seen: set[str] = set()
    selected: list[tuple[Path, str, float]] = []
    counts = {g: dict(records=0, no_lead=0, too_short=0, not_usable=0, kept=0) for g in groups}

    for hea in sorted(root.rglob("*.hea")):
        base = hea.with_suffix("")
        header = wfdb.rdheader(str(base))
        group = _group_from_header(header.comments)
        if group not in groups:
            continue
        counts[group]["records"] += 1

        if lead.lower() not in [n.lower() for n in header.sig_name]:
            counts[group]["no_lead"] += 1
            continue
        duration_s = header.sig_len / header.fs
        if duration_s < min_duration_s:
            counts[group]["too_short"] += 1
            continue
        if base.name not in usable:
            counts[group]["not_usable"] += 1
            continue

        subject = base.parent.name
        if subject in seen:
            continue          # one usable record per patient
        seen.add(subject)
        counts[group]["kept"] += 1
        age, sex = _age_sex(header.comments)
        selected.append((base, group, duration_s, age, sex))

    print(f"record selection (lead {lead.upper()}, >= {min_duration_s:.0f} s, "
          f"manually reviewed, one per patient):")
    for g in groups:
        c = counts[g]
        print(f"  {g:<8} {c['kept']:>3} patients kept of {c['records']:>3} records "
              f"(dropped: {c['no_lead']} without lead, {c['too_short']} too short, "
              f"{c['not_usable']} not usable)")

    if limit_per_group is not None:
        capped, per_group = [], {g: 0 for g in groups}
        for entry in selected:
            group = entry[1]
            if per_group[group] < limit_per_group:
                capped.append(entry)
                per_group[group] += 1
        selected = capped
        print(f"  limited to {limit_per_group} per group -> {len(selected)} records")

    return selected


def cut_segment(record: Record, seconds, offset_s: float = SEGMENT_OFFSET_S) -> Record:
    """Return a Record shortened to the first ``seconds`` after ``offset_s``.

    ``seconds="full"`` returns the record unchanged. Segments are cut from a
    fixed offset and are therefore nested: the 10 s segment lies inside the
    15 s one, and so on.
    """
    if seconds == "full":
        return record
    start = int(round(offset_s * record.fs))
    stop = start + int(round(float(seconds) * record.fs))
    if stop > record.signal.size:
        raise ValueError(f"record too short for a {seconds} s segment")
    return replace(record, signal=record.signal[start:stop])


def _run_one(record: Record, duration_s: float, seconds, method_name: str, fn,
             filter_cfg: str = "own",
             age: str = "n/a", sex: str = "n/a") -> dict:
    """One (record, segment length, method, filter) cell. Never raises, failures become rows."""
    row = {f: "" for f in FIELDS}
    row.update(
        record_id=record.record_id, subject_id=record.subject_id, group=record.group,
        age=age, sex=sex,
        method=method_name, filter_cfg=filter_cfg,
        segment_length_s=seconds, record_duration_s=round(duration_s, 1),
    )

    t0 = time.time()
    try:
        segment = cut_segment(record, seconds)
        intervals, t_amps, rr, info = fn(segment, apply_filter=APPLY_FILTER,
                                         **FILTER_CONFIGS[filter_cfg])
        res = compute_metrics(intervals, t_amps, rr)

        row.update(
            sdqt=res["sdqt"], qtvi=res["qtvi"],
            qt_mean=res["qt_mean"], rr_mean=res["rr_mean"],
            sdnn=res["sdnn"], t_amp_median=res["t_amp_median"],
            n_beats=res["n_beats"], n_rejected=res["n_rejected"],
            # flagged, not dropped
            below_min_beats=int(res["n_beats"] < MIN_BEATS),
            # failure counter is named differently per path
            n_failed=info.get("n_warp_failed", info.get("n_delineate_failed", "")),
            n_adapt_rejected=info.get("n_adapt_rejected", ""),
            n_implausible=res.get("n_implausible", ""),
            n_t_amp_nan=res.get("n_t_amp_nan", ""),
            n_iso_fallback=info.get("n_iso_fallback", ""),
            # template stats exist for i2DSW only
            threshold_uv=info.get("threshold_uv", ""),
            n_kept=info.get("n_kept", ""),
            n_total=info.get("n_total", ""),
            n_beats_raw=info.get("n_beats_raw", ""),
            adapt_distance_median=info.get("adapt_distance_median", ""),
            adapt_distance_max=info.get("adapt_distance_max", ""),
            # preprocessing diagnostics, both paths
            n_r_peaks=info.get("n_r_peaks", ""),
            align_shift_mean_ms=info.get("align_shift_mean_ms", ""),
            align_shift_max_ms=info.get("align_shift_max_ms", ""),
            align_corr_mean=info.get("align_corr_mean", ""),
            n_align_unusable=info.get("n_align_unusable", ""),
            status="ok",
        )
        # RT is secondary and shorter than QT, so its rejection must not
        # discard a valid QT measurement
        if "rt_ms" in info:
            try:
                row["sdrt"] = compute_metrics(info["rt_ms"], t_amps, rr)["sdqt"]
            except Exception:
                row["sdrt"] = ""
    except Exception as e:
        row.update(status="failed", error=f"{type(e).__name__}: {e}")

    row["elapsed_s"] = round(time.time() - t0, 1)
    return row


def _run_record(base: Path, group: str, duration_s: float, age: str, sex: str,
                todo: list[tuple]) -> list[dict]:
    """All pending cells of one patient, as a list of rows; runs in a worker process."""
    try:
        record = load_ptb_record(base, lead=LEAD)
    except Exception as e:
        return [{**{f: "" for f in FIELDS}, "record_id": base.name, "group": group,
                 "age": age, "sex": sex,
                 "status": "failed", "error": f"load failed: {type(e).__name__}: {e}"}]

    return [_run_one(record, duration_s, sec, name, METHODS[name], flt, age, sex)
            for sec, name, flt in todo]


def _done_keys(out_path: Path) -> set:
    """Keys already present in the CSV, so a restart can skip them."""
    if not out_path.exists():
        return set()
    with out_path.open(newline="") as f:
        return {
            (r["record_id"], r["segment_length_s"], r["method"], r["filter_cfg"])
            for r in csv.DictReader(f)
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None,
                        help="records per group (smoke test); default: all")
    parser.add_argument("--segments", nargs="+", default=list(SEGMENT_LENGTHS),
                        help="segment lengths in s, or 'full'")
    parser.add_argument("--filters", nargs="+", default=list(FILTERS),
                        choices=list(FILTER_CONFIGS),
                        help="filter configurations to run; default: all")
    parser.add_argument("--workers", type=int, default=max(os.cpu_count() - 1, 1),
                        help="parallel worker processes; 1 runs serially, which "
                             "keeps tracebacks readable when debugging")
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    parser.add_argument("--data", type=Path, default=DATA_ROOT)
    args = parser.parse_args()
    _t0 = time.time()

    segments = [s if s == "full" else int(s) for s in args.segments]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = _done_keys(args.out)
    is_new = not args.out.exists()

    selected = select_records(args.data, limit_per_group=args.limit)

    # pending cells per patient, so finished work is skipped before starting workers
    jobs = []
    for base, group, duration_s, age, sex in selected:
        todo = [(sec, name, flt) for sec in segments for name in METHODS for flt in args.filters
                if (base.name, str(sec), name, flt) not in done]
        if todo:
            jobs.append((base, group, duration_s, age, sex, todo))

    n_cells = sum(len(j[5]) for j in jobs)
    print(f"{n_cells} cells to compute across {len(jobs)} patients, "
          f"{len(done)} already done, {args.workers} worker(s)\n")
    if not jobs:
        return

    with args.out.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if is_new:
            writer.writeheader()

        def _emit(rows: list[dict], done_count: int) -> int:
            for row in rows:
                writer.writerow(row)
                done_count += 1
                label = (f"{row['record_id']} {row['group']:<7} "
                         f"{str(row['segment_length_s']):>4}s {row['method']:<8} "
                         f"{row['filter_cfg']}")
                sdqt = row["sdqt"]
                sdqt_str = f"{sdqt:>7.2f}" if isinstance(sdqt, float) else f"{'':>7}"
                print(f"[{done_count}/{n_cells}] {label:<34} "
                      f"{row['elapsed_s']:>6}s  sdqt={sdqt_str}  {row['status']}"
                      + (f"  {row['error']}" if row["status"] == "failed" else ""))
            f.flush()   # survive a crash
            return done_count

        i = 0
        if args.workers == 1:
            for job in jobs:
                i = _emit(_run_record(*job), i)
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(_run_record, *job): job for job in jobs}
                # completion order, not submission order
                for future in as_completed(futures):
                    base = futures[future][0]
                    try:
                        i = _emit(future.result(), i)
                    except Exception as e:
                        print(f"  !! {base.name}: worker died — {type(e).__name__}: {e}")

    print(f"\ndone in {time.time() - _t0:.1f}s")


if __name__ == "__main__":
    main()
