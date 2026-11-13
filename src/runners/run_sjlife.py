"""run_sjlife.py: evaluation driver for the SJLIFE paired ECG repository.

Runs both QTV paths over one segment per (subject, source) and appends one row
per (record, method, filter) to a CSV. Computation only, analysis happens
separately on the CSV. The run is resumable: combinations already present in
the CSV are skipped.

Usage:
    python -m src.runners.run_sjlife --limit 5      # smoke test
    python -m src.runners.run_sjlife                # full run
    python -m src.runners.run_sjlife --sources apple
"""



import argparse
import csv
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

from src.pipeline.loaders import Record, SJLifePair, load_sjlife_record, sjlife_pairs
from src.pipeline.metrics import compute_metrics
from src.pipeline.preprocessing import FILTER_CONFIGS
from src.pipeline.qtv_neurokit import compute_qt_intervals as nk_intervals
from src.pipeline.qtv_i2dsw import I2DSW_AVAILABLE, compute_qt_intervals as i2dsw_intervals

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = ROOT / "data" / "SJLIFE"
OUT_PATH = ROOT / "results" / "sjlife_full.csv"

SOURCES = ("apple", "clinical")
METHODS = {"neurokit": nk_intervals}
if I2DSW_AVAILABLE:  # i2DSW core not included in this repo, see src/kammler/i2DSW.py
    METHODS["i2dsw"] = i2dsw_intervals

CLINICAL_LEAD = "i"
SEGMENT_LENGTH_S = 10.0
APPLY_FILTER = True              # real ECG: high-pass per Schmidt 2018 Sec. 4.1
FILTERS = ("own",)
MIN_BEATS = 8

FIELDS = [
    # identity / configuration
    "record_id", "subject_id", "group", "source", "lead", "method", "filter_cfg",
    "segment_length_s", "record_duration_s",
    # per-subject metadata from the pairing CSV
    "age", "sex", "race", "hr_reported", "minutes_between",
    # beat funnel, in the order beats are lost
    "n_r_peaks", "n_beats_raw", "threshold_uv", "n_kept", "n_total",
    "n_failed", "n_adapt_rejected", "n_implausible", "n_t_amp_nan",
    "n_iso_fallback", "n_rejected", "n_beats",
    "below_min_beats",
    # results
    "sdqt", "sdrt", "qtvi",
    # potential confounders, needed to interpret a source difference in sdqt
    "qt_mean", "rr_mean", "sdnn", "t_amp_median",
    # quality control
    "adapt_distance_median", "adapt_distance_max",
    "align_shift_mean_ms", "align_shift_max_ms", "align_corr_mean",
    "n_align_unusable",
    # bookkeeping
    "elapsed_s", "status", "error",
]


def select_records(root: Path = DATA_ROOT, sources=SOURCES,
                   limit: int | None = None) -> list[tuple[SJLifePair, str]]:
    """One (subject, source) entry per recording to evaluate.

    Only subjects whose files exist for all requested sources are kept.
    """
    pairs = sjlife_pairs(root, tuple(sources))
    print(f"record selection (from {root.name}):")
    print(f"  {len(pairs):>3} subjects with all of {', '.join(sources)} on disk")

    if limit is not None:
        pairs = pairs[:limit]
        print(f"  limited to {limit} subjects -> {len(pairs)} subjects")

    return [(pair, source) for pair in pairs for source in sources]


def cut_segment(record: Record, seconds: float = SEGMENT_LENGTH_S) -> Record:
    """Return a Record shortened to a centred ``seconds`` window.

    Centring only applies to the Apple side, whose 30 s trace is trimmed to the
    clinical length. The 10 s clinical recording is used whole.
    """
    duration_s = record.signal.size / record.fs
    if seconds > duration_s + 1e-9:
        raise ValueError(f"record too short for a {seconds} s segment ({duration_s:.1f} s)")
    start = int(round((duration_s - seconds) / 2 * record.fs))
    stop = start + int(round(seconds * record.fs))
    return replace(record, signal=record.signal[start:stop])


def _run_one(record: Record, pair: SJLifePair, duration_s: float, source: str,
             method_name: str, fn, filter_cfg: str) -> dict:
    """One (record, method, filter) cell. Never raises, failures become rows."""
    row = {f: "" for f in FIELDS}
    row.update(
        record_id=record.record_id, subject_id=record.subject_id, group=record.group,
        source=source, lead=record.lead, method=method_name, filter_cfg=filter_cfg,
        segment_length_s=SEGMENT_LENGTH_S, record_duration_s=round(duration_s, 1),
        age=pair.age, sex=pair.sex, race=pair.race,
        # the rate each system reported for this recording
        hr_reported=pair.apple_hr if source == "apple" else pair.clinical_hr,
        minutes_between=pair.minutes_between,
    )

    t0 = time.time()
    try:
        segment = cut_segment(record)
        intervals, t_amps, rr, info = fn(segment, apply_filter=APPLY_FILTER,
                                         **FILTER_CONFIGS[filter_cfg])
        res = compute_metrics(intervals, t_amps, rr)

        row.update(
            sdqt=res["sdqt"], qtvi=res["qtvi"],
            qt_mean=res["qt_mean"], rr_mean=res["rr_mean"],
            sdnn=res["sdnn"], t_amp_median=res["t_amp_median"],
            n_beats=res["n_beats"], n_rejected=res["n_rejected"],
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


def _run_record(root: Path, pair: SJLifePair, source: str, clinical_lead: str,
                todo: list[tuple[str, str]]) -> list[dict]:
    """All pending cells of one (subject, source), as a list of rows; runs in a worker process."""
    try:
        record = load_sjlife_record(root, pair.subject_id, source, lead=clinical_lead)
        duration_s = record.signal.size / record.fs
    except Exception as e:
        return [{**{f: "" for f in FIELDS},
                 "record_id": f"sjlife_{source}_{pair.subject_id}",
                 "subject_id": pair.subject_id, "source": source, "status": "failed",
                 "error": f"load failed: {type(e).__name__}: {e}"}]

    return [_run_one(record, pair, duration_s, source, name, METHODS[name], flt)
            for name, flt in todo]


def _done_keys(out_path: Path) -> set:
    """Keys already present in the CSV, so a restart can skip them."""
    if not out_path.exists():
        return set()
    with out_path.open(newline="") as f:
        return {(r["record_id"], r["method"], r["filter_cfg"]) for r in csv.DictReader(f)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None,
                        help="subjects to run (smoke test); default: all")
    parser.add_argument("--sources", nargs="+", default=list(SOURCES),
                        choices=list(SOURCES),
                        help="recording sources to run; default: both")
    parser.add_argument("--clinical-lead", default=CLINICAL_LEAD,
                        help="lead taken from the clinical 12-lead recording; "
                             "the apple recording is single-lead I regardless")
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

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = _done_keys(args.out)
    is_new = not args.out.exists()

    selected = select_records(args.data, tuple(args.sources), args.limit)

    # pending cells per recording, so finished work is skipped before starting workers
    jobs = []
    for pair, source in selected:
        record_id = f"sjlife_{source}_{pair.subject_id}"
        todo = [(name, flt) for name in METHODS for flt in args.filters
                if (record_id, name, flt) not in done]
        if todo:
            jobs.append((args.data, pair, source, args.clinical_lead, todo))

    n_cells = sum(len(j[4]) for j in jobs)
    print(f"{n_cells} cells to compute across {len(jobs)} records, "
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
                label = (f"{row['record_id']:<24} {row['method']:<8} "
                         f"{row['filter_cfg']}")
                sdqt = row["sdqt"]
                sdqt_str = f"{sdqt:>7.2f}" if isinstance(sdqt, float) else f"{'':>7}"
                print(f"[{done_count}/{n_cells}] {label:<40} "
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
                    pair, source = futures[future][1], futures[future][2]
                    try:
                        i = _emit(future.result(), i)
                    except Exception as e:
                        print(f"  !! {pair.subject_id}/{source}: worker died — "
                              f"{type(e).__name__}: {e}")

    print(f"\ndone in {time.time() - _t0:.1f}s")


if __name__ == "__main__":
    main()
