"""qtv_neurokit.py: toolbox-baseline QTV path using neurokit2 (Makowski 2021).

Derives per-beat QT (QRS onset to T-end) and RT (R-peak to T-peak) intervals,
T amplitudes and RR intervals from neurokit2's standard delineation. Returns the
shared metrics contract (intervals_ms, t_amps_uv, rr_ms, info) consumed by
metrics.compute_metrics().
"""

import neurokit2 as nk
import numpy as np

from src.pipeline.preprocessing import preprocess_common, iso_level

def compute_qt_intervals(record, apply_filter: bool = True, filter_cutoff=0.6, filter_order=2,
                         filter_highcut=None,
                         method: str = "dwt", _return_intermediates: bool = False):
    """Per-beat QT and RT intervals, T amplitudes and RR intervals from neurokit2.

    Args:
        record: A Record (signal in mV, plus fs).
        apply_filter: Passed to preprocess_common.
        filter_highcut: Upper band edge in Hz, or None for a plain high-pass.
            See preprocessing.FILTER_CONFIGS.
        method: neurokit2 delineation method, "dwt" or "cwt".
        _return_intermediates: If True, info additionally carries sig, r_peaks,
            t_peaks, t_offsets, q_onsets.

    Returns:
        (qt_ms, t_amps_uv, rr_ms, info): aligned per-beat arrays plus an info
        dict holding the RT series and the delineation failure counts. Beats with
        missing delineation are NaN in qt_ms/t_amps_uv and excluded downstream by
        metrics.reject_beats.
    """
    fs = record.fs
    pre_stats: dict = {}
    sig, r_peaks = preprocess_common(record, apply_filter=apply_filter, filter_order=filter_order,
                                    filter_cutoff=filter_cutoff, filter_highcut=filter_highcut,
                                    stats=pre_stats)

    _, waves = nk.ecg_delineate(sig, rpeaks=r_peaks, sampling_rate=int(fs), method=method)
    t_offsets = np.asarray(waves["ECG_T_Offsets"], dtype=float)
    t_peaks = np.asarray(waves["ECG_T_Peaks"], dtype=float)
    q_onsets = np.asarray(waves["ECG_R_Onsets"], dtype=float)  # QRS onset
    p_offsets = np.asarray(waves["ECG_P_Offsets"], dtype=float)  # window start for the iso level

    r = np.asarray(r_peaks, dtype=float)
    n = min(len(r), len(t_offsets), len(t_peaks), len(q_onsets), len(p_offsets))
    r, t_offsets, t_peaks, q_onsets = r[:n], t_offsets[:n], t_peaks[:n], q_onsets[:n]
    p_offsets = p_offsets[:n]

    # QT interval (ms): QRS onset to T-offset
    qt_ms = (t_offsets - q_onsets) / fs * 1000.0

    # RT interval (ms): R-peak to T-peak (Orini 2023)
    rt_ms = (t_peaks - r) / fs * 1000.0

    # T amplitude (uV): T-peak minus the per-beat isoelectric level (Schmidt 2018, Eq. 16)
    time_ms_full = np.arange(len(sig)) * (1000.0 / fs)
    t_amps_uv = np.array([
        (sig[int(tp)] - iso_level(sig, time_ms_full, q_on, p_off)) * 1000.0
        if not np.isnan(tp) else np.nan
        for tp, q_on, p_off in zip(t_peaks, q_onsets, p_offsets)
    ])

    # RR interval (ms) per beat: difference to the previous R-peak
    rr_ms = np.full(n, np.nan)
    rr_ms[1:] = np.diff(r) / fs * 1000.0
    rr_ms[0] = rr_ms[1] if n > 1 else np.nan  # first beat has no preceding RR

    info = {
        "rt_ms": rt_ms,
        # beats where delineation could not resolve an endpoint
        "n_delineate_failed": int(np.isnan(qt_ms).sum()),
        "n_qonset_failed": int(np.isnan(q_onsets).sum()),
        # beats where iso_level() fell back to a fixed window (no P offset)
        "n_iso_fallback": int(np.isnan(p_offsets).sum()),
        "n_total": int(n),
        # number of delineated beats, logged under the i2DSW path's key
        "n_beats_raw": int(n),
    }
    info.update(pre_stats)

    if _return_intermediates:
        info.update({"sig": sig, "r_peaks": r, "t_peaks": t_peaks, "t_offsets": t_offsets,
                    "q_onsets": q_onsets})
    return qt_ms, t_amps_uv, rr_ms, info
