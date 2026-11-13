"""run_synthetic.py: evaluation driver for the synthetic robustness set.

Runs both QTV paths over the full parameter matrix and appends one row per
(seed, k, condition, method, filter) to a CSV. Computation only, analysis
happens separately on the CSV. The run is resumable: combinations already
present in the CSV are skipped.

Usage:
    python -m src.runners.run_synthetic

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 python -m src.runners.run_synthetic --workers 6
"""

import argparse
import csv
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from src.pipeline.synthetic import make_synthetic_record
from src.pipeline.metrics import compute_metrics
from src.pipeline.preprocessing import FILTER_CONFIGS
from src.pipeline.qtv_neurokit import compute_qt_intervals as nk_intervals
from src.pipeline.qtv_i2dsw import I2DSW_AVAILABLE, compute_qt_intervals as i2dsw_intervals

OUT_PATH = Path(__file__).resolve().parent.parent.parent / "results" / "synthetic_v2.csv"

SEEDS = (67, 68, 69, 70, 71, 72, 73, 74, 75, 76,)
K_VALUES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
CONDITIONS = (None, "gaussian", "baseline", "amplitude_mod")
METHODS = {"neurokit": nk_intervals}
if I2DSW_AVAILABLE:  # i2DSW core not included in this repo, see src/kammler/i2DSW.py
    METHODS["i2dsw"] = i2dsw_intervals

N_BEATS = 15          # overridable via --n-beats
FS = 1000
HR = 60
APPLY_FILTER = True

# both filter configurations are run on every cell, see preprocessing.FILTER_CONFIGS
FILTERS = tuple(FILTER_CONFIGS)

FIELDS = [
    "seed", "k", "condition", "method", "filter_cfg", "n_beats_cfg",
    "sdqt", "sdrt", "qtvi", "n_beats", "n_rejected", "n_t_amp_nan", "n_iso_fallback",
    "n_failed", "threshold_uv", "n_kept", "n_total",
    "elapsed_s", "status", "error",
]


def _done_keys() -> set:
    """Keys already present in the CSV, so a restart can skip them."""
    if not OUT_PATH.exists():
        return set()
    with OUT_PATH.open(newline="") as f:
        return {
            (r["seed"], r["k"], r["condition"], r["method"], r["filter_cfg"],
             r.get("n_beats_cfg", ""))
            for r in csv.DictReader(f)
        }


def _run_one(seed, k, condition, method_name, fn, n_beats_cfg, filter_cfg="own") -> dict:
    """One (seed, k, condition, method, filter, n_beats_cfg) cell. Never raises."""
    row = {f: "" for f in FIELDS}
    row.update(seed=seed, k=k, condition=condition or "clean", method=method_name,
                filter_cfg=filter_cfg, n_beats_cfg=n_beats_cfg)

    t0 = time.time()
    try:
        rec = make_synthetic_record(
            n_beats=n_beats_cfg, fs=FS, hr=HR,
            distortion=condition, k=k, random_state=seed,
        )
        intervals, t_amps, rr, info = fn(rec, apply_filter=APPLY_FILTER,
                                         **FILTER_CONFIGS[filter_cfg])
        res = compute_metrics(intervals, t_amps, rr)

        row.update(
            sdqt=res["sdqt"], qtvi=res["qtvi"],
            n_beats=res["n_beats"], n_rejected=res["n_rejected"],
            n_t_amp_nan=res.get("n_t_amp_nan", ""),
            n_iso_fallback=info.get("n_iso_fallback", ""),
            # failure counter is named differently per path
            n_failed=info.get("n_warp_failed", info.get("n_delineate_failed", "")),
            # template stats exist for i2DSW only
            threshold_uv=info.get("threshold_uv", ""),
            n_kept=info.get("n_kept", ""),
            n_total=info.get("n_total", ""),
            status="ok",
        )
        if "rt_ms" in info:
            row["sdrt"] = compute_metrics(info["rt_ms"], t_amps, rr)["sdqt"]
    except Exception as e:
        row.update(status="failed", error=f"{type(e).__name__}: {e}")

    row["elapsed_s"] = round(time.time() - t0, 1)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-beats", type=int, default=N_BEATS,
                        help="beats per record (Schmidt uses 500; PTB comparison suggests 120)")
    parser.add_argument("--workers", type=int, default=max(os.cpu_count() - 1, 1),
                        help="parallel worker processes; 1 runs serially")
    parser.add_argument("--filters", nargs="+", default=list(FILTERS),
                        choices=list(FILTER_CONFIGS),
                        help="filter configurations to run; default: all")
    args = parser.parse_args()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    done = _done_keys()
    is_new = not OUT_PATH.exists()

    cells = [
        (seed, k, cond, name, fn, args.n_beats, flt)
        for seed in SEEDS
        for k in K_VALUES
        for cond in CONDITIONS
        for name, fn in METHODS.items()
        for flt in args.filters
    ]
    todo = [c for c in cells
            if (str(c[0]), str(c[1]), c[2] or "clean", c[3], c[6], str(c[5])) not in done]
    print(f"{len(cells)} cells total, {len(done)} done, {len(todo)} to run, "
          f"n_beats={args.n_beats}, {args.workers} worker(s)")

    with OUT_PATH.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if is_new:
            writer.writeheader()

        def _emit(row: dict, i: int) -> None:
            writer.writerow(row)
            f.flush()  # survive a crash
            label = (f"{row['method']} {row['filter_cfg']} k={row['k']} "
                     f"{row['condition']} seed={row['seed']}")
            print(f"[{i}/{len(todo)}] {label:<42} {row['elapsed_s']:>7}s  {row['status']}")

        if args.workers == 1:
            for i, (seed, k, cond, name, fn, nb, flt) in enumerate(todo, 1):
                _emit(_run_one(seed, k, cond, name, fn, nb, flt), i)
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(_run_one, seed, k, cond, name, fn, nb, flt):
                               (seed, k, cond, name, flt)
                           for seed, k, cond, name, fn, nb, flt in todo}
                # completion order, not submission order
                for i, future in enumerate(as_completed(futures), 1):
                    try:
                        _emit(future.result(), i)
                    except Exception as e:
                        seed, k, cond, name, flt = futures[future]
                        print(f"  !! {name} {flt} k={k} {cond} seed={seed}: worker died — {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()