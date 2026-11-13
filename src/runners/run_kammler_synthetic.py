"""run_kammler_synthetic.py: evaluation driver for Kammler's original Baumert 2012 data.

Runs both QTV paths over all 30 waveform files and appends one row per
(k, condition, method, filter) to a CSV. Quanta are converted to mV by a single
approximated constant (loaders._KAMMLER_QUANTA_TO_MV).

Usage:
    python -m src.runners.run_kammler_synthetic

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 python -m src.runners.run_kammler_synthetic --workers 6
"""

import argparse
import csv
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from src.pipeline.loaders import load_kammler_record
from src.pipeline.metrics import compute_metrics
from src.pipeline.preprocessing import FILTER_CONFIGS
from src.pipeline.qtv_neurokit import compute_qt_intervals as nk_intervals
from src.pipeline.qtv_i2dsw import I2DSW_AVAILABLE, compute_qt_intervals as i2dsw_intervals

OUT_PATH = Path(__file__).resolve().parent.parent.parent / "results" / "kammler_synthetic_15s_v2.csv"
DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "kammler_synth" / "waveform_src"

K_INDICES = range(1, 11)
DISTORTIONS = {"N": "gaussian", "B": "baseline", "M": "amplitude_mod"}
METHODS = {"neurokit": nk_intervals}
if I2DSW_AVAILABLE:  # i2DSW core not included in this repo, see src/kammler/i2DSW.py
    METHODS["i2dsw"] = i2dsw_intervals

DURATION_S = 360.0     # overridable via --duration
APPLY_FILTER = True

# both filter configurations are run on every cell, see preprocessing.FILTER_CONFIGS
FILTERS = tuple(FILTER_CONFIGS)

FIELDS = [
    "k", "condition", "method", "filter_cfg", "duration_cfg",
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
            (r["k"], r["condition"], r["method"], r["filter_cfg"], r.get("duration_cfg", ""))
            for r in csv.DictReader(f)
        }


def _run_one(k_index, letter, method_name, fn, duration_cfg, filter_cfg="own") -> dict:
    """One (k, condition, method, filter, duration_cfg) cell. Never raises."""
    condition = DISTORTIONS[letter]
    k = k_index / 10.0
    row = {f: "" for f in FIELDS}
    row.update(k=k, condition=condition, method=method_name, filter_cfg=filter_cfg,
               duration_cfg=duration_cfg)

    t0 = time.time()
    try:
        rec = load_kammler_record(DATA_ROOT, k_index, letter, duration_s=duration_cfg)
        intervals, t_amps, rr, info = fn(rec, apply_filter=APPLY_FILTER,
                                         **FILTER_CONFIGS[filter_cfg])
        res = compute_metrics(intervals, t_amps, rr)

        row.update(
            sdqt=res["sdqt"], qtvi=res["qtvi"],
            n_beats=res["n_beats"], n_rejected=res["n_rejected"],
            n_t_amp_nan=res.get("n_t_amp_nan", ""),
            n_iso_fallback=info.get("n_iso_fallback", ""),
            n_failed=info.get("n_warp_failed", info.get("n_delineate_failed", "")),
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
    parser.add_argument("--duration", type=float, default=DURATION_S,
                        help="seconds per record, sliced from the 360 s original")
    parser.add_argument("--workers", type=int, default=max(os.cpu_count() - 1, 1),
                        help="parallel worker processes; 1 runs serially")
    parser.add_argument("--filters", nargs="+", default=list(FILTERS),
                        choices=list(FILTER_CONFIGS),
                        help="filter configurations to run; default: all")
    args = parser.parse_args()

    if not DATA_ROOT.is_dir():
        raise FileNotFoundError(
            f"expected data under {DATA_ROOT} -- copy waveform_src/ there first "
            f"(see check_kammler_data.py)."
        )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    done = _done_keys()
    is_new = not OUT_PATH.exists()

    cells = [
        (k_index, letter, name, fn, args.duration, flt)
        for k_index in K_INDICES
        for letter in DISTORTIONS
        for name, fn in METHODS.items()
        for flt in args.filters
    ]
    todo = [c for c in cells
            if (str(c[0] / 10.0), DISTORTIONS[c[1]], c[2], c[5], str(c[4])) not in done]
    print(f"{len(cells)} cells total, {len(done)} done, {len(todo)} to run, "
          f"duration={args.duration}s, {args.workers} worker(s)")

    with OUT_PATH.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if is_new:
            writer.writeheader()

        def _emit(row: dict, i: int) -> None:
            writer.writerow(row)
            f.flush()  # survive a crash
            label = f"{row['method']} {row['filter_cfg']} k={row['k']} {row['condition']}"
            print(f"[{i}/{len(todo)}] {label:<32} {row['elapsed_s']:>7}s  {row['status']}")

        if args.workers == 1:
            for i, (k_index, letter, name, fn, dur, flt) in enumerate(todo, 1):
                _emit(_run_one(k_index, letter, name, fn, dur, flt), i)
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures = {
                    pool.submit(_run_one, k_index, letter, name, fn, dur, flt):
                        (k_index, letter, name, flt)
                    for k_index, letter, name, fn, dur, flt in todo
                }
                # completion order, not submission order
                for i, future in enumerate(as_completed(futures), 1):
                    try:
                        _emit(future.result(), i)
                    except Exception as e:
                        k_index, letter, name, flt = futures[future]
                        print(f"  !! {name} {flt} k={k_index/10.0} {DISTORTIONS[letter]}: "
                              f"worker died — {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
