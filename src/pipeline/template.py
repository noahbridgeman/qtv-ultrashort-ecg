"""template.py: beat template construction for the i2DSW QTV pipeline.

The template is the mean of the mutually similar beat segments (Schmidt 2014,
Sec. III-B; Schmidt 2018, Sec. 4.1), excluding beats whose normalized Manhattan
distance (Eq. 20) to the mean waveform exceeds a threshold that is raised in
steps of 20 until at least half the beats survive. build_template() returns
(template_2d, q_on_index, t_end_index, p_off_index, t_peak_index,
template_stats), where template_2d is an (N, 2) array of [time_ms relative to
the R-peak, amplitude_uV] and the indices are row positions into it. Delineation
is delegated to neurokit2's wavelet delineator (Schmidt 2018, Sec. 4.2).
"""

import numpy as np
import neurokit2 as nk

# Schmidt 2014, Eq. 20 and the surrounding template-construction rules
_MIN_KEPT_FRACTION = 0.50   # escalate while fewer than half the beats survive
_THRESHOLD_STEP_UV = 20.0   # threshold increment per escalation step


def build_template(
    beats: list[tuple[np.ndarray, int]],
    fs: float,
    max_distance_uv: float = 30.0, # Schmidt 2014 starting value, escalated if needed
) -> tuple[np.ndarray, int, int]:
    """Build an i2DSW-compatible template beat from a list of PT segments.

    All beats are resampled to the median beat length, then beats whose distance
    to the median waveform exceeds the exclusion threshold are dropped before
    averaging.

    Args:
        beats: Output of extract_beats(), a list of (segment, start_offset).
        fs: Sampling frequency in Hz, for the ms time axis.
        max_distance_uv: Starting exclusion threshold in uV (Schmidt 2014, Eq. 20),
            raised in 20 uV steps while fewer than 50% of beats remain.

    Returns:
        (template_2d, q_on_index, t_end_index, p_off_index, t_peak_index,
        template_stats), where p_off_index (P offset, iso-level window) and
        t_peak_index (RT interval) may be None, and template_stats reports the
        escalated threshold in uV and how many extracted beats survived it.

    Raises:
        ValueError: If ``beats`` is empty.
    """
    if not beats:
        raise ValueError("beats list is empty")

    segments = [seg for seg, _ in beats]
    target_len = int(np.median([len(s) for s in segments]))

    # physical time axis (ms) for each beat, relative to the R-peak (t = 0)
    pre_samples = int(round(0.370 * fs))
    ms_per_sample = 1000.0 / fs
    beat_t_ms = [(np.arange(len(s)) - pre_samples) * ms_per_sample for s in segments]

    # common bounding time grid; interpolate every beat onto it, then scale to uV
    t_start = max(t[0] for t in beat_t_ms)
    t_end = float(np.median([t[-1] for t in beat_t_ms]))
    time_ms = np.linspace(t_start, t_end, target_len)
    resampled_uv = np.stack([
        np.interp(time_ms, t, s) for t, s in zip(beat_t_ms, segments)
    ]) * 1000.0  # mV -> uV

    # exclude beats whose normalized Manhattan distance in uV/ms
    # (Schmidt 2014, Eq. 20) to the median waveform is too large
    median_waveform = np.median(resampled_uv, axis=0)
    span_ms = float(time_ms[-1] - time_ms[0])
    mad = np.sum(np.abs(resampled_uv - median_waveform), axis=1) / span_ms

    # raise the threshold in steps of 20 uV/ms while fewer than half the beats
    # survive (Schmidt 2014)
    threshold_uv = float(max_distance_uv)
    keep = mad <= threshold_uv
    while keep.mean() < _MIN_KEPT_FRACTION:
        threshold_uv += _THRESHOLD_STEP_UV
        keep = mad <= threshold_uv
    template_stats = {
        "threshold_uv": threshold_uv,
        "n_kept": int(keep.sum()),
        "n_total": len(segments),
    }

    amplitude_uv = resampled_uv[keep].mean(axis=0)

    template_2d = np.column_stack([time_ms, amplitude_uv])
    q_on_index, t_end_index, p_off_index, t_peak_index = _estimate_fiducials(template_2d)
    return template_2d, q_on_index, t_end_index, p_off_index, t_peak_index, template_stats


def build_template_raw(
    beats: list[tuple[np.ndarray, int]],
    fs: float,
    max_distance_uv: float = 30.0,
) -> tuple[np.ndarray, dict]:
    """Like build_template, without fiducial estimation; returns (template_2d, stats)."""
    if not beats:
        raise ValueError("beats list is empty")

    segments = [seg for seg, _ in beats]
    target_len = int(np.median([len(s) for s in segments]))

    pre_samples = int(round(0.370 * fs))
    ms_per_sample = 1000.0 / fs
    beat_t_ms = [(np.arange(len(s)) - pre_samples) * ms_per_sample for s in segments]

    t_start = max(t[0] for t in beat_t_ms)
    t_end_ms = float(np.median([t[-1] for t in beat_t_ms]))
    time_ms = np.linspace(t_start, t_end_ms, target_len)
    resampled_uv = np.stack([
        np.interp(time_ms, t, s) for t, s in zip(beat_t_ms, segments)
    ]) * 1000.0

    median_waveform = np.median(resampled_uv, axis=0)
    span_ms = float(time_ms[-1] - time_ms[0])
    mad = np.sum(np.abs(resampled_uv - median_waveform), axis=1) / span_ms

    threshold_uv = float(max_distance_uv)
    keep = mad <= threshold_uv
    while keep.mean() < _MIN_KEPT_FRACTION:
        threshold_uv += _THRESHOLD_STEP_UV
        keep = mad <= threshold_uv

    stats = {"threshold_uv": threshold_uv, "n_kept": int(keep.sum()), "n_total": len(segments)}
    amplitude_uv = resampled_uv[keep].mean(axis=0)
    return np.column_stack([time_ms, amplitude_uv]), stats


def _estimate_fiducials(template_2d: np.ndarray, n_tiles: int = 12) -> tuple[int, int, int | None, int | None]:
    """Q-onset, T-end, P-offset and T-peak row indices on the n_tiles-tiled template, None when unresolved."""
    time_ms = template_2d[:, 0]
    amp = template_2d[:, 1]
    n = len(amp)

    # rate of the template's own time grid, not the record's fs
    fs_eff = int(round(1000.0 / (time_ms[1] - time_ms[0])))

    tiled = np.tile(amp, n_tiles)
    _, info = nk.ecg_peaks(tiled, sampling_rate=fs_eff)
    r_peaks = np.asarray(info["ECG_R_Peaks"], dtype=int)
    if len(r_peaks) < 3:
        raise ValueError("fiducial estimation failed: too few R-peaks on tiled template")

    _, waves = nk.ecg_delineate(
        tiled, rpeaks=r_peaks, sampling_rate=fs_eff, method="dwt"
    )
    q_rel = np.asarray(waves["ECG_R_Onsets"], dtype=float) - r_peaks
    t_rel = np.asarray(waves["ECG_T_Offsets"], dtype=float) - r_peaks
    p_rel = np.asarray(waves["ECG_P_Offsets"], dtype=float) - r_peaks
    tp_rel = np.asarray(waves["ECG_T_Peaks"], dtype=float) - r_peaks
    if np.all(np.isnan(q_rel)) or np.all(np.isnan(t_rel)):
        raise ValueError("fiducial estimation failed: delineation returned no valid points")

    r_idx = int(np.argmin(np.abs(time_ms)))  # R-peak = template origin (t = 0)
    q_on_index = int(np.clip(round(r_idx + np.nanmedian(q_rel)), 0, n - 1))
    t_end_index = int(np.clip(round(r_idx + np.nanmedian(t_rel)), 0, n - 1))

    # P offset is optional: iso_level() falls back to a fixed span before Q
    if np.all(np.isnan(p_rel)):
        p_off_index = None
    else:
        p_off_index = int(np.clip(round(r_idx + np.nanmedian(p_rel)), 0, n - 1))
        if p_off_index >= q_on_index:
            p_off_index = None

    if q_on_index >= t_end_index:
        raise ValueError("fiducial estimation failed: q_on >= t_end")

    # T peak is optional: without it RT is left unmeasured
    if np.all(np.isnan(tp_rel)):
        t_peak_index = None
    else:
        t_peak_index = int(np.clip(round(r_idx + np.nanmedian(tp_rel)), 0, n - 1))
        if not (q_on_index < t_peak_index < t_end_index):
            t_peak_index = None

    return q_on_index, t_end_index, p_off_index, t_peak_index