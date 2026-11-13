"""preprocessing.py: shared steps for both algorithms (neurokit2 and i2DSW).

Turns a raw Record into aligned, extracted beats ready for QT measurement.
Mirrors Schmidt 2018, Section 4.1 (General framework).
"""

import neurokit2 as nk
import numpy as np
from scipy.signal import butter, sosfiltfilt

#for Woody TDE
_ALIGN_WINDOW_MS = 100.0 # as Schmidt2014 mentions
_ALIGN_MAX_SHIFT_MS = 20.0
_ALIGN_MAX_ITER = 5

#for iso level calculation
_ISO_WINDOW_MS = 20.0     # Schmidt2018 width of the flat search window
_ISO_FALLBACK_MS = 100.0  # PQ span searched when the P offset is unavailable

# "own" is the tuned high-pass, "orini" the band-pass of Orini 2023.
# Both configurations filter before R-peak detection.
FILTER_CONFIGS = {
    "own":   dict(filter_cutoff=0.6, filter_order=2, filter_highcut=None),
    "orini": dict(filter_cutoff=0.5, filter_order=2, filter_highcut=20.0),
}

def highpass(signal: np.ndarray, fs: float, cutoff=0.6, order: int = 2,
             highcut: float | None = None) -> np.ndarray:

    """Butterworth high-pass filter to remove baseline wander.

    With ``highcut`` set, a band-pass is applied instead (Orini 2023, see
    FILTER_CONFIGS).

    Args:
        signal: 1-D ECG signal in mV.
        fs: Sampling frequency in Hz.
        cutoff: Cutoff frequency in Hz (lower edge when ``highcut`` is given).
        order: Filter order.
        highcut: Upper cutoff in Hz, or None for a plain high-pass.

    Returns:
        Filtered signal, same shape as input.
    """
    if np.isnan(signal).any():
        raise ValueError("signal contains NaN; handle gaps before filtering")

    if highcut is None:
        sos = butter(order, cutoff, btype="high", fs=fs, output="sos")
    else:
        sos = butter(order, [cutoff, highcut], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, signal)

def detect_qrs(signal: np.ndarray, fs: float) -> np.ndarray:
    """Detect QRS complexes with neurokit2's default detector; returns R-peak sample indices."""
    _, info = nk.ecg_peaks(signal, sampling_rate=int(fs))
    return np.asarray(info["ECG_R_Peaks"], dtype=int)

def _normalized_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation between two equal-length segments, -1.0 without variance."""
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt(np.dot(a, a) * np.dot(b, b)))
    if denom <= 0.0:
        return -1.0
    return float(np.dot(a, b) / denom)

def align_beats(signal, fs, r_peaks,
                window_ms=_ALIGN_WINDOW_MS,
                max_shift_ms=_ALIGN_MAX_SHIFT_MS,
                max_iter=_ALIGN_MAX_ITER):
    """Time delay estimation: remove R-peak jitter by iterative alignment.

    Improved Woody's Method (Cabasson/Meste 2008, Eq. 11; Schmidt 2018). Each
    beat is shifted to maximize the unnormalized cross-correlation between its
    window and the mean of all other windows, with that reference updated
    immediately. Repeats until the shifts stabilize or max_iter is reached.
    The window is 100 ms wide (Schmidt 2014), the search is bounded to
    +-max_shift_ms, edge beats are excluded, delays stay integer, and the
    zero-mean gauge is applied once at the end.

    Returns: (r_peaks_aligned, correlations, shifts)
    """
    signal = np.asarray(signal, dtype=float)
    r_peaks = np.asarray(r_peaks, dtype=int)

    half = int(round(0.5 * window_ms / 1000.0 * fs))
    max_shift = int(round(max_shift_ms / 1000.0 * fs))
    offsets = np.arange(-half, half + 1)

    shifts = np.zeros(r_peaks.size, dtype=int)
    correlations = np.full(r_peaks.size, np.nan)

    # Beat nur nutzbar, wenn Fenster + Suchbereich ins Signal passen
    usable = ((r_peaks - half - max_shift) >= 0) & ((r_peaks + half + max_shift) < len(signal))
    if usable.sum() < 2:
        return r_peaks, correlations, shifts

    for _ in range(max_iter):
        windows = np.stack([signal[r + s + offsets] for r,s in zip(r_peaks[usable], shifts[usable])])
        window_sum = windows.sum(axis=0)
        n_usable = usable.sum()
        updated = shifts.copy()
        for pos, i in enumerate(np.flatnonzero(usable)):
            loo_reference = (window_sum - windows[pos]) / (n_usable - 1)
            candidates = [float(np.dot(signal[r_peaks[i] + s + offsets], loo_reference))
                          for s in range(-max_shift, max_shift + 1)]
            new_shift = int(np.argmax(candidates)) - max_shift
            updated[i] = new_shift

            new_window = signal[r_peaks[i] + new_shift + offsets]
            window_sum += new_window - windows[pos]
            windows[pos] = new_window

        if np.array_equal(updated, shifts):
            break
        shifts = updated

    windows = np.stack([signal[r + s + offsets] for r, s in zip(r_peaks[usable], shifts[usable])])
    window_sum = windows.sum(axis=0)
    n_usable = usable.sum()
    for pos, i in enumerate(np.flatnonzero(usable)):
        loo_reference = (window_sum - windows[pos]) / (n_usable - 1)
        correlations[i] = _normalized_correlation(signal[r_peaks[i] + shifts[i] + offsets], loo_reference)

    # zero-mean gauge, applied once at the end
    shifts[usable] -= int(round(shifts[usable].mean()))

    return r_peaks + shifts, correlations, shifts


def extract_beats(signal: np.ndarray,
                  fs: float,
                  r_peaks: np.ndarray
                  )-> list[tuple[np.ndarray, int]]:
    """Cut one PT segment per beat, P-onset to T-end (Schmidt 2018).

    Window for beat i (R at t_QRS,i):
        start = t_QRS,i - 370ms
        end   = t_QRS,i+1 - 240ms           if mean RR >= 720ms
                t_QRS,i + (2/3)*mean RR     otherwise

    Returns a list of (beat_segment, start_offset) tuples:
    - start_offset is the index in `signal` where the segment begins
    - beat_segment is the actual segment
    """
    rr_mean = np.mean(np.diff(r_peaks)) / fs  # mean RR in seconds
    pre = int(round(0.370 * fs))              # 370 ms before R

    if rr_mean >= 0.720: # slow heartbeat
        starts = r_peaks[:-1] - pre
        ends = r_peaks[1:] - int(round(0.240 * fs))
    else: # fast heartbeat
        starts = r_peaks - pre
        ends = r_peaks + int(round((2 / 3) * rr_mean * fs))

    beats = []
    for start, end in zip(starts, ends):
        if start < 0 or end > len(signal):
            continue
        if end > start:
            beats.append((signal[start:end], start))
    return beats

def normalized_manhattan_distance(a_uv, b_uv, time_ms) -> float:
    """Normalized Manhattan distance between two aligned waveforms, in uV/ms.

    Mean absolute amplitude difference, normalized by the duration of the
    compared segment (Schmidt 2014, Eq. 20). Used with a threshold of 30 uV/ms
    to exclude dissimilar beats from the template (template.py) and of
    100 uV/ms to reject beats the warping could not adapt to (qtv_i2dsw.py).

    Args:
        a_uv, b_uv: Amplitude arrays in uV, already on a common time grid.
        time_ms: The shared time grid in ms.

    Returns:
        Distance in uV/ms, or NaN if the segment has no positive duration.
    """
    a_uv = np.asarray(a_uv, dtype=float)
    b_uv = np.asarray(b_uv, dtype=float)
    time_ms = np.asarray(time_ms, dtype=float)
    span_ms = float(time_ms[-1] - time_ms[0]) if time_ms.size > 1 else 0.0
    if span_ms <= 0:
        return float("nan")
    return float(np.sum(np.abs(a_uv - b_uv))) / span_ms


def iso_level(amplitude, time_ms, q_on_idx, p_off_idx=None,
              window_ms: float = _ISO_WINDOW_MS,
              fallback_ms: float = _ISO_FALLBACK_MS) -> float:
    """Isoelectric level of one beat: median of the flattest segment before Q-onset.

    Median of the flattest 20 ms window between the end of the P wave and the
    Q onset (Schmidt 2018, Sec. 4.4), flatness measured as the standard
    deviation within the window. Used for beat rejection in
    metrics.reject_beats.

    Args:
        amplitude: Amplitude values.
        time_ms: Time coordinate per sample, same length as ``amplitude``.
        q_on_idx: Index of the Q onset (window end).
        p_off_idx: Index of the P offset (window start). If missing or NaN,
            the search falls back to the ``fallback_ms`` before the Q onset.

    Returns:
        The isoelectric level, or NaN if the segment is too short to evaluate.
    """
    amplitude = np.asarray(amplitude, dtype=float)
    time_ms = np.asarray(time_ms, dtype=float)

    if q_on_idx is None or not np.isfinite(q_on_idx):
        return float("nan")
    q_on_idx = int(q_on_idx)

    if p_off_idx is None or not np.isfinite(p_off_idx):
        start = int(np.searchsorted(time_ms, time_ms[q_on_idx] - fallback_ms, side="left"))
    else:
        start = int(p_off_idx)
    start = max(start, 0)
    if q_on_idx - start < 2:
        return float("nan")

    # slide a full-width window across the PQ segment, keep the flattest one
    best_sd, best_median = np.inf, float("nan")
    for i in range(start, q_on_idx):
        j = int(np.searchsorted(time_ms, time_ms[i] + window_ms, side="left"))
        if j > q_on_idx:
            break  # a full window no longer fits before the Q onset
        if j - i < 2:
            continue
        sd = float(np.std(amplitude[i:j]))
        if sd < best_sd:
            best_sd, best_median = sd, float(np.median(amplitude[i:j]))

    if not np.isfinite(best_median):  # PQ segment shorter than one window
        best_median = float(np.median(amplitude[start:q_on_idx]))
    return best_median


def preprocess_common(record, apply_filter: bool = True, filter_order=2, filter_cutoff=0.6,
                      filter_highcut: float | None = None,
                      stats: dict | None = None):
    """Shared preprocessing for both QTV paths: filter, detect QRS, align.

    apply_filter: Whether to filter at all.
    filter_highcut: Upper band edge in Hz, or None for a plain high-pass.
    stats: If a dict is passed, it is filled with QRS-detection and alignment
        diagnostics (n_r_peaks, align shift statistics, correlation quality).

    Returns:
        (sig, r_peaks): the signal and aligned R-peak sample indices.
    """
    fs = record.fs
    sig = (highpass(record.signal, fs, cutoff=filter_cutoff, order=filter_order,
                    highcut=filter_highcut)
           if apply_filter else record.signal)
    r_peaks = detect_qrs(sig, fs)
    r_peaks, correlations, shifts = align_beats(sig, fs, r_peaks)

    if stats is not None:
        # beats align_beats could not shift (signal edge) carry NaN correlation
        unusable = int(np.isnan(correlations).sum())
        shift_ms = np.abs(shifts) * (1000.0 / fs)
        usable_shift = shift_ms[~np.isnan(correlations)]
        stats.update(
            n_r_peaks=int(r_peaks.size),
            n_align_unusable=unusable,
            align_shift_mean_ms=float(usable_shift.mean()) if usable_shift.size else float("nan"),
            align_shift_max_ms=float(usable_shift.max()) if usable_shift.size else float("nan"),
            align_corr_mean=float(np.nanmean(correlations)) if correlations.size else float("nan"),
        )
    return sig, r_peaks

def preprocess(record, apply_filter: bool = True):
    """Run the full i2DSW preprocessing chain: shared steps + beat extraction."""
    sig, r_peaks = preprocess_common(record, apply_filter=apply_filter)
    beats = extract_beats(sig, record.fs, r_peaks)
    return beats
