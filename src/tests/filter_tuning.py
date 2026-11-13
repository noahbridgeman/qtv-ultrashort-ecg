"""filter_tuning.py: 2-D sweep of the highpass filter (cutoff x order), per segment length.

The optimization target is max(clean, baseline) SDQT, with gaussian and
amplitude_mod logged separately. The winning cutoff/order is chosen from the
10 s and 15 s cells only (best_setting_ultra_short), longer segments are swept
and reported as context (best_settings). At hr=60, n_beats in seconds equals
the segment length in seconds.

Usage:
    python -m src.tests.filter_tuning --workers 4
"""

import argparse
import csv
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from src.pipeline.synthetic import load_synthetic
from src.pipeline.qtv_neurokit import compute_qt_intervals
from src.pipeline.metrics import compute_metrics

ROOT = Path(__file__).resolve().parent.parent.parent
OUT_PATH = ROOT / "results" / "filter_tuning.csv"

CUTOFFS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
ORDERS = [1, 2, 4, 6, 8, 16, 32]
SEEDS = [13, 43, 82]
SEGMENT_LENGTHS_S = [10, 15, 30, 60, 120]
TARGET_SEGMENT_LENGTHS_S = (10, 15)   # these decide the winner
OPT_CONDITIONS = ("clean", "baseline")   # drive the cutoff/order choice
WATCH_CONDITIONS = ("gaussian", "amplitude_mod")
ALL_CONDITIONS = OPT_CONDITIONS + WATCH_CONDITIONS

FIELDS = [
    "segment_length_s", "cutoff", "order", "condition",
    "sdqt_mean", "sdqt_std", "qt_mean", "n_values",
    "elapsed_s", "status", "error",
]


def _run_cell(seg_len_s: int, cutoff: float, order: int) -> list[dict]:
    """One (segment_length_s, cutoff, order) cell: one row per condition, pooled over SEEDS x 10 k-values."""
    t0 = time.time()
    per_cond = {c: [] for c in ALL_CONDITIONS}
    per_cond_qt = {c: [] for c in ALL_CONDITIONS}
    error = ""

    for seed in SEEDS:
        for rec in load_synthetic(n_beats=seg_len_s, random_state=seed):
            cond = rec.record_id.removeprefix("synth_").rsplit("_k", 1)[0]
            if cond not in ALL_CONDITIONS:
                continue
            try:
                qt_ms, t_amps, rr_ms, _info = compute_qt_intervals(
                    rec, apply_filter=True, filter_order=order, filter_cutoff=cutoff
                )
                res = compute_metrics(qt_ms, t_amps, rr_ms)
                per_cond[cond].append(res["sdqt"])
                per_cond_qt[cond].append(res["qt_mean"])
            except Exception as e:
                error = f"{type(e).__name__}: {e}"

    elapsed = round(time.time() - t0, 1)
    rows = []
    for cond in ALL_CONDITIONS:
        vals = per_cond[cond]
        qt_vals = [q for q in per_cond_qt[cond] if np.isfinite(q)]
        rows.append({
            "segment_length_s": seg_len_s, "cutoff": cutoff, "order": order,
            "condition": cond,
            "sdqt_mean": float(np.mean(vals)) if vals else "",
            "sdqt_std": float(np.std(vals)) if vals else "",
            # distortion check: a cutoff that flattens the T wave lowers SDQT
            # but shifts qt_mean away from a physiological value
            "qt_mean": float(np.mean(qt_vals)) if qt_vals else "",
            "n_values": len(vals),
            "elapsed_s": elapsed,
            "status": "ok" if vals else "failed",
            "error": error if not vals else "",
        })
    return rows


def _done_cells(out_path: Path) -> set:
    """(segment_length_s, cutoff, order) cells already fully written."""
    if not out_path.exists():
        return set()
    with out_path.open(newline="") as f:
        return {
            (int(r["segment_length_s"]), float(r["cutoff"]), int(r["order"]))
            for r in csv.DictReader(f)
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=max(os.cpu_count() - 1, 1))
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args()
    _t0 = time.time()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = _done_cells(args.out)
    is_new = not args.out.exists()

    jobs = [
        (seg, cf, o)
        for seg in SEGMENT_LENGTHS_S for cf in CUTOFFS for o in ORDERS
        if (seg, cf, o) not in done
    ]
    print(f"{len(jobs)} cells to compute ({len(SEGMENT_LENGTHS_S)} segment lengths x "
          f"{len(CUTOFFS)} cutoffs x {len(ORDERS)} orders), {len(done)} already done, "
          f"{args.workers} worker(s)\n")
    if not jobs:
        return

    with args.out.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if is_new:
            writer.writeheader()

        i = 0
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_run_cell, *job): job for job in jobs}
            for future in as_completed(futures):
                seg, cf, o = futures[future]
                i += 1
                try:
                    rows = future.result()
                    for row in rows:
                        writer.writerow(row)
                    f.flush()
                    combined = max(
                        r["sdqt_mean"] for r in rows if r["condition"] in OPT_CONDITIONS
                    )
                    print(f"[{i}/{len(jobs)}] seg={seg:>2}s cutoff={cf:<4} order={o:<2} "
                          f"combined={combined:.3f}  {rows[0]['elapsed_s']}s")
                except Exception as e:
                    print(f"[{i}/{len(jobs)}] seg={seg:>2}s cutoff={cf:<4} order={o:<2} "
                          f"!! worker failed: {type(e).__name__}: {e}")

    print(f"\ndone in {time.time() - _t0:.1f}s")


def _combined_by_cell(out_path: Path = OUT_PATH) -> dict:
    """combined[(segment_length_s, cutoff, order)] = max(clean, baseline) sdqt_mean, complete cells only."""
    by_cell: dict = {}
    with out_path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["status"] != "ok" or row["condition"] not in OPT_CONDITIONS:
                continue
            key = (int(row["segment_length_s"]), float(row["cutoff"]), int(row["order"]))
            by_cell.setdefault(key, {})[row["condition"]] = float(row["sdqt_mean"])

    return {
        key: max(vals.values())
        for key, vals in by_cell.items()
        if set(vals) == set(OPT_CONDITIONS)
    }


def _qt_mean_by_cell(out_path: Path = OUT_PATH) -> dict:
    """qt_mean[(segment_length_s, cutoff, order)] = mean QT over OPT_CONDITIONS, in ms."""
    by_cell: dict = {}
    with out_path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["status"] != "ok" or row["condition"] not in OPT_CONDITIONS:
                continue
            if not row.get("qt_mean"):
                continue
            key = (int(row["segment_length_s"]), float(row["cutoff"]), int(row["order"]))
            by_cell.setdefault(key, []).append(float(row["qt_mean"]))

    return {key: float(np.mean(vals)) for key, vals in by_cell.items() if vals}


def best_settings(out_path: Path = OUT_PATH) -> list[tuple]:
    """Best (cutoff, order) per segment length, over the accumulated CSV.

    Context table, not the selection criterion; see best_setting_ultra_short().
    """
    combined = _combined_by_cell(out_path)
    qt_means = _qt_mean_by_cell(out_path)

    results = []
    for seg in sorted({k[0] for k in combined}):
        cell = min((k for k in combined if k[0] == seg), key=lambda k: combined[k])
        results.append((seg, cell[1], cell[2], combined[cell]))

    print("\n=== best cutoff/order per segment length (context only, not the selection) ===")
    print(f"{'seg (s)':>8}{'cutoff':>10}{'order':>8}{'combined':>12}{'qt_mean':>10}")
    for seg, cf, o, comb in results:
        qt = qt_means.get((seg, cf, o))
        qt_str = f"{qt:>10.1f}" if qt is not None else f"{'n/a':>10}"
        print(f"{seg:>8}{cf:>10}{o:>8}{comb:>12.3f}{qt_str}")
    return results


def best_setting_ultra_short(out_path: Path = OUT_PATH,
                             target_lengths=TARGET_SEGMENT_LENGTHS_S) -> tuple:
    """The selection criterion: winner chosen from the target lengths only.

    The target lengths are pooled via max(), the remaining segment lengths are
    printed afterwards as context. Returns (cutoff, order, combined_us).
    """
    combined = _combined_by_cell(out_path)

    candidates = {}
    for cf in CUTOFFS:
        for o in ORDERS:
            vals = [combined.get((seg, cf, o)) for seg in target_lengths]
            if any(v is None for v in vals):
                continue
            candidates[(cf, o)] = max(vals)

    if not candidates:
        print("no cutoff/order has complete results for all target segment lengths yet")
        return None

    (cf, o), combined_us = min(candidates.items(), key=lambda kv: kv[1])
    qt_means = _qt_mean_by_cell(out_path)

    print(f"\n=== ultra-short selection ({'/'.join(f'{s}s' for s in target_lengths)}) ===")
    print(f"winner: cutoff={cf} Hz, order={o}  (combined={combined_us:.3f} ms)")

    print("\nqt_mean at this setting (physiological plausibility check):")
    for seg in SEGMENT_LENGTHS_S:
        qt = qt_means.get((seg, cf, o))
        marker = " <- target" if seg in target_lengths else ""
        print(f"  {seg:>4}s: {qt:.1f} ms{marker}" if qt is not None
              else f"  {seg:>4}s: n/a{marker}")

    print("\ncontext at this setting, other segment lengths:")
    for seg in SEGMENT_LENGTHS_S:
        if seg in target_lengths:
            continue
        val = combined.get((seg, cf, o))
        print(f"  {seg:>4}s: {val:.3f} ms" if val is not None else f"  {seg:>4}s: n/a")

    return cf, o, combined_us


def diagnose_dip(seg_len_s: int = 30, cutoffs=(0.5, 0.6, 0.7), orders=(1, 2, 4)) -> None:
    """Per-seed SDQT, beat and rejection counts on clean signals, for the given cutoffs and orders.

    Standalone diagnostic, writes nothing to the CSV.
    """
    for cutoff in cutoffs:
        for order in orders:
            per_seed = []
            for seed in SEEDS:
                for rec in load_synthetic(n_beats=seg_len_s, random_state=seed):
                    cond = rec.record_id.removeprefix("synth_").rsplit("_k", 1)[0]
                    if cond != "clean":  # clean signals only
                        continue
                    qt_ms, t_amps, rr_ms, _info = compute_qt_intervals(
                        rec, apply_filter=True, filter_order=order, filter_cutoff=cutoff
                    )
                    res = compute_metrics(qt_ms, t_amps, rr_ms)
                    per_seed.append((res["sdqt"], res["n_beats"], res["n_rejected"]))
            sdqt_vals = [p[0] for p in per_seed]
            n_beats_vals = [p[1] for p in per_seed]
            n_rej_vals = [p[2] for p in per_seed]
            print(f"cutoff={cutoff} order={order}  "
                  f"sdqt_mean={np.mean(sdqt_vals):.2f}  sdqt_std={np.std(sdqt_vals):.2f}  "
                  f"n_beats_mean={np.mean(n_beats_vals):.1f}  n_rejected_mean={np.mean(n_rej_vals):.1f}")


if __name__ == "__main__":
    main()
    best_settings()
    best_setting_ultra_short()
    # diagnose_dip()
