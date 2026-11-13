"""metrics.py: QTV metric computation, shared by both QTV paths.

Takes the per-beat intervals (QT, or RT as a secondary series) together with the
RR intervals and computes SDQT and QTVi. Beat rejection on the T amplitude
(Schmidt 2018, Sec. 4.5) and on physiological plausibility bounds happens here;
warping-specific rejection (Schmidt 2014) happens in the i2DSW path, which marks
the affected beats NaN beforehand.
"""

import numpy as np

_T_AMP_REJECT_FRACTION = 0.10  # Schmidt 2018 Sec. 4.5: reject beats below 10% of median |T amp|

# plausibility window for a single raw, uncorrected QT interval, in ms
_QT_MIN_MS, _QT_MAX_MS = 200.0, 600.0

def reject_beats(intervals_ms: np.ndarray, t_amps_uv: np.ndarray) -> np.ndarray:
    """Keep beats with a plausible interval and a T wave large enough to delineate.

    Rejects beats whose |T amplitude| is below 10% of the median (Schmidt 2018,
    Sec. 4.5), whose interval lies outside the plausibility bounds, or whose
    interval is NaN.
    """
    intervals_ms = np.asarray(intervals_ms, dtype=float)
    t_amps_uv = np.asarray(t_amps_uv, dtype=float)


    median_abs = np.nanmedian(np.abs(t_amps_uv))
    if not np.isfinite(median_abs):
        raise ValueError("no beat has a finite T amplitude (isoelectric level failed)")
    keep = np.abs(t_amps_uv) >= _T_AMP_REJECT_FRACTION * median_abs
    keep &= (intervals_ms >= _QT_MIN_MS) & (intervals_ms <= _QT_MAX_MS)
    keep &= ~np.isnan(intervals_ms)
    return keep


def sdqt(intervals_ms: np.ndarray) -> float:
    """Standard deviation of the intervals (SDQT / SDRT), in ms.

    Sample SD with 1/(N-1) (Schmidt 2018, Eq. 17).
    """
    intervals_ms = np.asarray(intervals_ms, dtype=float)
    return float(np.std(intervals_ms, ddof=1))


def qtvi(intervals_ms: np.ndarray, rr_ms: np.ndarray) -> float:
    """QT variability index (Berger 1997): log10 of normalized QT var over normalized RR var.

    QTVi = log10( (QT_var / QT_mean^2) / (RR_var / RR_mean^2) )
    """
    intervals_ms = np.asarray(intervals_ms, dtype=float)
    rr_ms = np.asarray(rr_ms, dtype=float)
    qt_var = np.var(intervals_ms, ddof=0)
    rr_var = np.var(rr_ms, ddof=0)
    qt_mean = np.mean(intervals_ms)
    rr_mean = np.mean(rr_ms)
    qtvn = qt_var / qt_mean ** 2
    rrvn = rr_var / rr_mean ** 2
    if rrvn == 0.0:
        return float("nan")
    return float(np.log10(qtvn / rrvn))


def compute_metrics(
    intervals_ms: np.ndarray,
    t_amps_uv: np.ndarray,
    rr_ms: np.ndarray,
    apply_rejection: bool = True,
) -> dict:
    """Reject beats, then compute SDQT and QTVi over the accepted beats.

    Args:
        intervals_ms: Per-beat QT (i2DSW) or RT (neurokit2) intervals, in ms.
        t_amps_uv: Per-beat T-wave amplitudes, in uV (for beat rejection).
        rr_ms: RR intervals in ms (for QTVi). One per accepted beat.
        apply_rejection: Whether to apply the T-amplitude beat rejection.

    Returns:
        {"sdqt": float, "qtvi": float, "qt_mean": float, "rr_mean": float,
         "sdnn": float, "t_amp_median": float, "n_beats": int,
         "n_rejected": int, "n_implausible": int, "n_t_amp_nan": int}

    Raises:
        ValueError: If fewer than 2 beats remain after rejection.
    """
    intervals_ms = np.asarray(intervals_ms, dtype=float)
    t_amps_uv = np.asarray(t_amps_uv, dtype=float)
    rr_ms = np.asarray(rr_ms, dtype=float)

    if apply_rejection:
        keep = reject_beats(intervals_ms, t_amps_uv)
        implausible = ((intervals_ms < _QT_MIN_MS) | (intervals_ms > _QT_MAX_MS))
        n_implausible = int((implausible & ~np.isnan(intervals_ms)).sum())
    else:
        keep = ~np.isnan(intervals_ms)
        n_implausible = 0

    iv = intervals_ms[keep]
    rr = rr_ms[keep]
    n_rejected = int((~keep).sum())

    if len(iv) < 2:
        raise ValueError(f"fewer than 2 beats remain after rejection ({len(iv)})")

    return {
        "sdqt": sdqt(iv),
        "qtvi": qtvi(iv, rr),
        "qt_mean": float(np.mean(iv)),
        "rr_mean": float(np.mean(rr)),
        # SDNN over the accepted beats, the HRV counterpart to SDQT
        "sdnn": float(np.std(rr, ddof=0)),
        # median |T amplitude| over the accepted beats
        "t_amp_median": float(np.nanmedian(np.abs(t_amps_uv[keep]))),
        "n_beats": int(len(iv)),
        "n_rejected": n_rejected,
        "n_implausible": n_implausible,
        # beats without a measurable T amplitude (no T peak, or iso_level failed)
        "n_t_amp_nan": int(np.isnan(t_amps_uv).sum()),
    }