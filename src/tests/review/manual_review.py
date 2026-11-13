"""manual_review.py: annotate QRS polarity, T polarity and usability by hand.

The signal is shown high-pass filtered and otherwise untouched, with no
polarity correction applied. QRS and T polarity are recorded independently per
record:

    u   QRS upright,  T upright     pipeline: no inversion
    i   QRS upright,  T inverted    pipeline: flip for T delineation only
    j   QRS inverted, T inverted    pipeline: flip whole signal
    k   QRS inverted, T upright     pipeline: flip whole signal, flip back for T
    x   exclude (unreliable R peaks, no usable T wave, unreadable morphology)
    d   toggle delineation overlay
    b   back one record
    q   save and quit

The delineation overlay is off by default and the review is blinded and
shuffled, restricted to MI/healthy records of at least 110 s. The excerpt is
taken from the middle of the recording, and decisions are written after every
keystroke, so the review can be interrupted and resumed at any point.

Usage:
    python -m src.tests.review.manual_review
"""

import csv
import random
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
import wfdb

from src.pipeline.loaders import load_ptb_record, _group_from_header
from src.pipeline.preprocessing import highpass

# interactive backend, so plt.show() opens a window
try:
    matplotlib.use("MacOSX")
except Exception:
    try:
        matplotlib.use("QtAgg")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = ROOT / "data" / "ptbdb"
OUT_CSV = ROOT / "results" / "manual_review.csv"

EXCERPT_S = 6.0                 # seconds of signal shown per record
MIN_DURATION_S = 110.0          # same filter as the evaluation driver
GROUPS = ("MI", "healthy")
FILTER_CUTOFF, FILTER_ORDER = 0.6, 2
SHUFFLE_SEED = 42

FIELDS = ["record_id", "group", "decision", "qrs_polarity", "t_polarity", "n_r_peaks"]

# key -> (decision, qrs_polarity, t_polarity)
_KEYS = {
    "u": ("usable", 1, 1),
    "i": ("usable", 1, -1),
    "j": ("usable", -1, -1),
    "k": ("usable", -1, 1),
    "x": ("excluded", "", ""),
}


def select_records() -> list[Path]:
    """One record per patient, MI/healthy only, at least MIN_DURATION_S long."""
    bases, seen = [], set()
    for hea in sorted(DATA_ROOT.rglob("*.hea")):
        patient = hea.parent.name
        if patient in seen:
            continue
        header = wfdb.rdheader(str(hea.with_suffix("")))
        if _group_from_header(header.comments) not in GROUPS:
            continue
        if header.sig_len / header.fs < MIN_DURATION_S:
            continue
        seen.add(patient)
        bases.append(hea.with_suffix(""))
    return bases


def _excerpt(base: Path):
    """A few seconds from the middle of the filtered signal, plus the R peaks as detected."""
    record = load_ptb_record(base)
    fs = int(record.fs)
    sig = highpass(record.signal, fs, cutoff=FILTER_CUTOFF, order=FILTER_ORDER)

    n = int(round(EXCERPT_S * fs))
    start = max((sig.size - n) // 2, 0)          # middle, away from settling artefacts
    excerpt = sig[start:start + n]

    _, info = nk.ecg_peaks(sig, sampling_rate=fs)
    r_peaks = np.asarray(info["ECG_R_Peaks"], dtype=int)
    r_local = r_peaks[(r_peaks >= start) & (r_peaks < start + excerpt.size)] - start

    return excerpt * 1000.0, np.arange(excerpt.size) / fs, r_local, int(r_peaks.size), record.group


def _delineation(base: Path):
    """Q onset and T offset indices within the excerpt, or empty arrays on failure."""
    try:
        record = load_ptb_record(base)
        fs = int(record.fs)
        sig = highpass(record.signal, fs, cutoff=FILTER_CUTOFF, order=FILTER_ORDER)
        n = int(round(EXCERPT_S * fs))
        start = max((sig.size - n) // 2, 0)
        segment = sig[start:start + n]

        _, info = nk.ecg_peaks(segment, sampling_rate=fs)
        r_peaks = np.asarray(info["ECG_R_Peaks"], dtype=int)
        _, waves = nk.ecg_delineate(segment, rpeaks=r_peaks, sampling_rate=fs, method="dwt")

        def _clean(key):
            idx = np.asarray(waves[key], dtype=float)
            idx = idx[np.isfinite(idx)].astype(int)
            return idx[(idx >= 0) & (idx < segment.size)]

        return _clean("ECG_R_Onsets"), _clean("ECG_T_Offsets")
    except Exception:
        return np.array([], dtype=int), np.array([], dtype=int)


def _load_done() -> dict:
    if not OUT_CSV.exists():
        return {}
    with OUT_CSV.open(newline="") as f:
        return {r["record_id"]: r for r in csv.DictReader(f)}


def _save(decisions: dict):
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for row in decisions.values():
            w.writerow({k: row.get(k, "") for k in FIELDS})


def main():
    bases = select_records()
    random.Random(SHUFFLE_SEED).shuffle(bases)

    decisions = _load_done()
    todo = [b for b in bases if b.name not in decisions]
    print(f"{len(bases)} eligible patients, {len(decisions)} reviewed, {len(todo)} to go")
    print("calibrate on a stretch in the middle before deciding anything you keep")
    if not todo:
        print("nothing left to review")
        return

    state = {"i": 0, "overlay": False}
    fig, ax = plt.subplots(figsize=(16, 5))

    def draw():
        ax.clear()
        base = todo[state["i"]]
        try:
            beat, time_s, r_local, n_peaks, group = _excerpt(base)
        except Exception as e:
            ax.text(0.5, 0.5, f"{base.parent.name}/{base.name}\n{type(e).__name__}: {e}\n\n"
                              f"[x] exclude   [b] back   [q] quit",
                    ha="center", va="center", transform=ax.transAxes)
            state["current"] = dict(record_id=base.name, group="?", n_r_peaks="")
            fig.canvas.draw_idle()
            return

        ax.plot(time_s, beat, color="#333333", lw=0.9)
        ax.axhline(0.0, color="#999999", lw=0.6, ls=":")
        if r_local.size:
            ax.plot(time_s[r_local], beat[r_local], "v", color="#0173B2", ms=7,
                    ls="none", label="R peak")

        if state["overlay"]:
            q_on, t_off = _delineation(base)
            for idx, colour, label in ((q_on, "#029E73", "Q onset"),
                                       (t_off, "#DE8F05", "T offset")):
                idx = idx[idx < beat.size]
                if idx.size:
                    ax.plot(time_s[idx], beat[idx], "|", color=colour, ms=16,
                            ls="none", label=label)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="upper right", ncol=3, fontsize=8)

        ax.set_xlabel("Time [s]")
        ax.set_ylabel(r"Amplitude [$\mu$V]")
        # record and patient id for later lookup, the group is withheld
        ax.set_title(
            f"[{state['i'] + 1}/{len(todo)}]   {base.parent.name}/{base.name}   "
            f"{n_peaks} beats detected"
            f"{'   | delineation shown' if state['overlay'] else ''}\n"
            f"[u] QRS+ T+    [i] QRS+ T-    [j] QRS- T-    [k] QRS- T+    "
            f"[x] exclude    [d] delineation    [b] back    [q] quit",
            fontsize=10)
        state["current"] = dict(record_id=base.name, group=group, n_r_peaks=n_peaks)
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == "q":
            plt.close(fig)
            return
        if event.key == "d":
            state["overlay"] = not state["overlay"]
            draw()
            return
        if event.key == "b":
            state["i"] = max(0, state["i"] - 1)
            state["overlay"] = False
            draw()
            return
        if event.key not in _KEYS:
            return

        decision, qrs_pol, t_pol = _KEYS[event.key]
        row = dict(state["current"], decision=decision,
                   qrs_polarity=qrs_pol, t_polarity=t_pol)
        decisions[row["record_id"]] = row
        _save(decisions)          # written immediately

        if state["i"] + 1 >= len(todo):
            plt.close(fig)
            return
        state["i"] += 1
        state["overlay"] = False   # overlay off on each new record
        draw()

    fig.canvas.mpl_connect("key_press_event", on_key)
    draw()
    plt.show()

    counts = {}
    for row in decisions.values():
        key = (row["decision"] if row["decision"] == "excluded"
               else f"QRS{row['qrs_polarity']:>2} T{row['t_polarity']:>2}")
        counts.setdefault(row["group"], {}).setdefault(key, 0)
        counts[row["group"]][key] += 1
    print(f"\n{len(decisions)} decisions -> {OUT_CSV}")
    for group, by_key in sorted(counts.items()):
        print(f"  {group:<10} " + ", ".join(f"{k}: {v}" for k, v in sorted(by_key.items())))


if __name__ == "__main__":
    main()