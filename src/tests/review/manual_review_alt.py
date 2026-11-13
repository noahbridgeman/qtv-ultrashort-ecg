"""manual_review_alt.py: second pass over alternative records of excluded patients.

Shows the remaining recordings of patients whose first reviewed record was
excluded, using the same criteria as manual_review.py (MI/healthy, lead II, at
least 110 s). Decisions from the first pass stay valid and are not revisited,
and this pass is not blinded. Keys are the same as the first pass:

    u   QRS upright,  T upright
    i   QRS upright,  T inverted
    j   QRS inverted, T inverted
    k   QRS inverted, T upright
    x   exclude
    d   toggle delineation overlay
    b   back one record
    q   save and quit

Results are appended to the same CSV. Where a patient ends up with several
usable records, the evaluation driver takes the first in sorted order.

Usage:
    python -m src.tests.review.manual_review_alt
"""

import csv
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
import wfdb

from src.pipeline.loaders import load_ptb_record, _group_from_header
from src.pipeline.preprocessing import highpass

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

EXCERPT_S = 6.0
MIN_DURATION_S = 110.0
GROUPS = ("MI", "healthy")
LEAD = "ii"
FILTER_CUTOFF, FILTER_ORDER = 0.6, 2

FIELDS = ["record_id", "group", "decision", "qrs_polarity", "t_polarity", "n_r_peaks"]
_KEYS = {
    "u": ("usable", 1, 1),
    "i": ("usable", 1, -1),
    "j": ("usable", -1, -1),
    "k": ("usable", -1, 1),
    "x": ("excluded", "", ""),
}


def _eligible(hea: Path) -> bool:
    """MI/healthy, lead II present and long enough, read from the header only."""
    try:
        header = wfdb.rdheader(str(hea.with_suffix("")))
    except Exception:
        return False
    return (_group_from_header(header.comments) in GROUPS
            and LEAD in [n.lower() for n in header.sig_name]
            and header.sig_len / header.fs >= MIN_DURATION_S)


def select_alternatives(decisions: dict) -> list[Path]:
    """Unreviewed recordings of patients whose reviewed record was excluded.

    Patients who already have a usable record are skipped.
    """
    by_patient: dict[str, list[Path]] = {}
    for hea in sorted(DATA_ROOT.rglob("*.hea")):
        if _eligible(hea):
            by_patient.setdefault(hea.parent.name, []).append(hea.with_suffix(""))

    alternatives = []
    for patient, bases in by_patient.items():
        reviewed = [b for b in bases if b.name in decisions]
        if not reviewed:
            continue                                  # never reached the first pass
        if any(decisions[b.name]["decision"] == "usable"
                and decisions[b.name]["qrs_polarity"] == "1"
                and decisions[b.name]["t_polarity"] == "1" for b in reviewed):
            continue
        alternatives.extend(b for b in bases if b.name not in decisions)
    return alternatives


def _excerpt(base: Path):
    """Filtered signal from the middle of the recording, plus detected R peaks."""
    record = load_ptb_record(base)
    fs = int(record.fs)
    sig = highpass(record.signal, fs, cutoff=FILTER_CUTOFF, order=FILTER_ORDER)

    n = int(round(EXCERPT_S * fs))
    start = max((sig.size - n) // 2, 0)
    excerpt = sig[start:start + n]

    _, info = nk.ecg_peaks(sig, sampling_rate=fs)
    r_peaks = np.asarray(info["ECG_R_Peaks"], dtype=int)
    r_local = r_peaks[(r_peaks >= start) & (r_peaks < start + excerpt.size)] - start
    return excerpt * 1000.0, np.arange(excerpt.size) / fs, r_local, int(r_peaks.size), record.group


def _delineation(base: Path):
    """Q onsets and T offsets within the excerpt; empty arrays if delineation fails."""
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


def _load(path: Path = OUT_CSV) -> dict:
    if not path.exists():
        return {}
    with path.open(newline="") as f:
        return {r["record_id"]: r for r in csv.DictReader(f)}


def _save(decisions: dict, path: Path = OUT_CSV):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for row in decisions.values():
            w.writerow({k: row.get(k, "") for k in FIELDS})


def main():
    decisions = _load()
    if not decisions:
        print(f"no first-pass decisions found at {OUT_CSV}")
        return

    todo = select_alternatives(decisions)
    n_patients = len({b.parent.name for b in todo})
    print(f"{len(todo)} alternative recordings from {n_patients} patients "
          f"whose reviewed record was excluded")
    if not todo:
        print("nothing to review")
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
        decisions[state["current"]["record_id"]] = dict(
            state["current"], decision=decision,
            qrs_polarity=qrs_pol, t_polarity=t_pol)
        _save(decisions)

        if state["i"] + 1 >= len(todo):
            plt.close(fig)
            return
        state["i"] += 1
        state["overlay"] = False
        draw()

    fig.canvas.mpl_connect("key_press_event", on_key)
    draw()
    plt.show()

    # patients recovered by the second pass
    usable_patients = {}
    for row in decisions.values():
        if row["decision"] == "usable" and row["qrs_polarity"] == "1" and row["t_polarity"] == "1":
            usable_patients.setdefault(row["group"], set()).add(row["record_id"])
    print(f"\n{len(decisions)} decisions total -> {OUT_CSV}")
    for group, records in sorted(usable_patients.items()):
        print(f"  {group:<10} {len(records)} records with upright QRS and T")


if __name__ == "__main__":
    main()
