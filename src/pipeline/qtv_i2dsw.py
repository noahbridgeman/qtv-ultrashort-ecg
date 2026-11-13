"""qtv_i2dsw.py: i2DSW QTV path (Schmidt 2014/2018).

Builds one template per record (template.py) and warps it onto every beat
waveform via Kammler's execute_i2DSW(). Per-beat QT (Q-onset to T-end) is read
off the warped template, RT (R-peak to T-peak, Orini 2023) alongside. Returns
the shared metrics contract (intervals_ms, t_amps_uv, rr_ms, info) consumed by
metrics.compute_metrics().
"""

import numpy as np

from src.kammler.i2DSW import AVAILABLE as I2DSW_AVAILABLE, Optimizer
from src.pipeline.preprocessing import (preprocess_common, extract_beats, iso_level,
                                        normalized_manhattan_distance)
from src.pipeline.template import build_template

_PRE_R_S = 0.370          # beat window starts 370 ms before R (Schmidt 2018)
_ITER_STEP_MAX = 4
_BREAKING_FLAVOUR = "2018_paper"  # breaking conditions per Schmidt 2018; see i2DSW.py for alternatives
_MAX_ADAPTATION_DISTANCE = 100.0  # uV/ms; Schmidt 2014 beat-rejection threshold (Eq. 20)


def _adaptation_distance(warped_template: np.ndarray, waveform_2d: np.ndarray) -> float:
    """Residual distance (uV/ms) between a warped template and its beat (Schmidt 2014, Eq. 20)."""
    t_time, t_amp = warped_template[:, 0], warped_template[:, 1]
    w_time, w_amp = waveform_2d[:, 0], waveform_2d[:, 1]
    mask = (w_time >= t_time.min()) & (w_time <= t_time.max())
    if mask.sum() < 2:
        return float("nan")
    interpolated = np.interp(w_time[mask], t_time, t_amp)
    return normalized_manhattan_distance(interpolated, w_amp[mask], w_time[mask])


def _beat_to_waveform(seg: np.ndarray, fs: float) -> np.ndarray:
    """Convert one beat segment (mV) into i2DSW's [time_ms relative to R, amplitude_uV] format."""
    pre_samples = int(round(_PRE_R_S * fs))
    time_ms = (np.arange(len(seg)) - pre_samples) * (1000.0 / fs)
    return np.column_stack([time_ms, seg * 1000.0])  # mV -> uV


def _t_amplitude(warped_template: np.ndarray, q_on_i: int, t_end_i: int,
                 p_off_i: int | None = None, t_peak_i: int | None = None) -> float:
    """T-wave amplitude (uV) relative to the per-beat isoelectric level (Schmidt 2018, Sec. 4.5)."""
    time = warped_template[:, 0]
    amp = warped_template[:, 1]
    baseline = iso_level(amp, time, q_on_i, p_off_i)
    if not np.isfinite(baseline):
        return np.nan

    if t_peak_i is not None:
        return float(amp[t_peak_i] - baseline)

    r_i = int(np.argmin(np.abs(time)))          # R-peak = t = 0
    lo = max(r_i + 1, q_on_i)
    st_t = amp[lo:t_end_i + 1]
    if st_t.size == 0:
        return np.nan
    return float(st_t[np.argmax(np.abs(st_t - baseline))] - baseline)


def compute_qt_intervals(
    record,
    apply_filter: bool = True,
    filter_cutoff: float = 0.6,
    filter_order: int = 2,
    filter_highcut: float | None = None,
    _return_intermediates: bool = False,
):
    """Per-beat QT intervals, T amplitudes and RR intervals via i2DSW.

    Args:
        record: A Record (signal in mV, plus fs).
        apply_filter: Passed to preprocess_common.
        filter_highcut: Upper band edge in Hz, or None for a plain high-pass.
            See preprocessing.FILTER_CONFIGS.
        _return_intermediates: If True, info additionally carries template_2d,
            q_on_index, t_end_index, warped_templates.

    Returns:
        (qt_ms, t_amps_uv, rr_ms, info): aligned per-beat arrays plus an info
        dict with the template exclusion threshold, the number of beats that
        formed the template, and the number of failed warpings. Beats where
        warping fails are NaN in qt_ms/t_amps_uv and excluded downstream by
        metrics.reject_beats.

    Raises:
        ValueError: If fewer than 2 beats are extracted.
        NotImplementedError: If the i2DSW implementation is not installed.
    """
    if not I2DSW_AVAILABLE:
        raise NotImplementedError("i2DSW implementation not included; see src/kammler/i2DSW.py")
    fs = record.fs
    pre_stats: dict = {}
    sig, r_peaks = preprocess_common(
        record, apply_filter=apply_filter, filter_order=filter_order, filter_cutoff=filter_cutoff,
        filter_highcut=filter_highcut,
        stats=pre_stats,
    )
    beats = extract_beats(sig, fs, r_peaks)
    if len(beats) < 2:
        raise ValueError(f"fewer than 2 beats extracted ({len(beats)}); cannot build a template")

    template_2d, q_on_index, t_end_index, p_off_index, t_peak_index, template_stats = build_template(beats, fs)
    r_index = int(np.argmin(np.abs(template_2d[:, 0])))  # R-peak row (t = 0)

    n = len(beats)
    qt_ms = np.full(n, np.nan)
    rt_ms = np.full(n, np.nan)
    t_amps_uv = np.full(n, np.nan)
    warped_templates: list = [None] * n
    n_warp_failed = 0
    n_adapt_rejected = 0
    adapt_distances = np.full(n, np.nan)
    iter_steps = np.full(n, np.nan)

    for i, (seg, _offset) in enumerate(beats):
        waveform_2d = _beat_to_waveform(seg, fs)
        try:
            warped, q_on_i, t_end_i, _orig_dist, _final_dist, iter_step = Optimizer.execute_i2DSW(
                template_2d, waveform_2d, q_on_index, t_end_index,
                iter_step_max=_ITER_STEP_MAX,
                breaking_conditions_flavour=_BREAKING_FLAVOUR,
            )
        except Exception:
            n_warp_failed += 1
            continue  # leave NaN, dropped by metrics.reject_beats

        iter_steps[i] = iter_step

        # Schmidt 2014 beat rejection: discard beats with a poor template match
        distance = _adaptation_distance(warped, waveform_2d)
        adapt_distances[i] = distance
        if np.isfinite(distance) and distance > _MAX_ADAPTATION_DISTANCE:
            n_adapt_rejected += 1
            continue  # leave NaN; dropped downstream by metrics.reject_beats

        qt_ms[i] = warped[t_end_i, 0] - warped[q_on_i, 0]
        # RT is read at the template's T-peak row; execute_i2DSW does not resample
        if t_peak_index is not None:
            rt_ms[i] = warped[t_peak_index, 0] - warped[r_index, 0]
        t_amps_uv[i] = _t_amplitude(warped, q_on_i, t_end_i, p_off_index, t_peak_index)
        warped_templates[i] = warped

    # RR interval (ms) per beat: difference to the previous R-peak
    r = np.asarray(r_peaks, dtype=float)[:n]
    rr_ms = np.full(n, np.nan)
    rr_ms[1:] = np.diff(r) / fs * 1000.0
    rr_ms[0] = rr_ms[1] if n > 1 else np.nan

    info = dict(template_stats)
    info.update(pre_stats)
    info["n_beats_raw"] = n
    # the template is delineated once per record: all beats or none
    info["n_iso_fallback"] = n if p_off_index is None else 0
    info["n_warp_failed"] = n_warp_failed
    info["n_adapt_rejected"] = n_adapt_rejected
    info["adapt_distance_median"] = float(np.nanmedian(adapt_distances)) if n else float("nan")
    info["adapt_distance_max"] = float(np.nanmax(adapt_distances)) if n else float("nan")
    info["iter_step_median"] = float(np.nanmedian(iter_steps)) if n else float("nan")
    info["rt_ms"] = rt_ms
    if _return_intermediates:
        info.update({
            "template_2d": template_2d,
            "q_on_index": q_on_index,
            "t_end_index": t_end_index,
            "warped_templates": warped_templates,
        })
    return qt_ms, t_amps_uv, rr_ms, info