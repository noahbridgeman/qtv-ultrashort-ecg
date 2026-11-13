"""Synthetic ECG dataset for the zero-QTV robustness test.

A clean signal of identical, evenly spaced beats has zero QT variability by
construction, so any QTV a pipeline reports on it is artefact (Schmidt 2018,
Sec. 4.6; Baumert 2012). The signal is one clean beat repeated with np.tile,
with the T-wave amplitude optionally scaled by k in [0.1, 1.0] and one of three
distortions applied, all anchored to the T-wave amplitude of the clean beat
(Baumert 2012, Sec. 2). The T-wave window is derived analytically from the
ecgsyn model (McSharry 2003), as implemented in neurokit2's ecg_simulate.
"""

from __future__ import annotations
import numpy as np
import neurokit2 as nk
from scipy.signal import find_peaks
from src.pipeline.loaders import Record
from typing import Iterator

_DISTORTIONS = {"gaussian", "baseline", "amplitude_mod"}

_T_ANGLE_DEG = 100.0    # T-Winkel, ecgsyn
_T_SIGMA_DEG = 22.9         # b_T = 0.4 rad, McSharry Tab. I
_T_WINDOW_SIGMAS = 3.0      # +/- 3 sigma umfasst die Gauss-Kurve praktisch vollstaendig

# noise parameters (Baumert 2012, Sec. 2)
_WANDER_FREQ_HZ = 0.3
_NOISE_FRACTION = 0.03
_AM_FREQ_HZ = 0.3
_AM_DEPTH = 0.7


# Baumert's k-sweep: ten T-wave amplitudes from minimum to maximum
_K_VALUES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
#0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,


def make_synthetic_record(
    n_beats: int = 60,
    fs: int = 1000,
    hr: int = 60,
    distortion: str | None = None,
    k: float = 1.0,
    random_state: int = 67,
) -> Record:
    """Generate a synthetic ECG Record with zero QT variability.

    Args:
        n_beats: Number of identical beats.
        fs: Sampling frequency in Hz.
        hr: Constant heart rate in BPM.
        distortion: One of ``"gaussian"``, ``"baseline"``, ``"amplitude_mod"``,
            or ``None`` for a clean signal.
        k: T-wave amplitude scaling in [0.1, 1.0]; 1.0 leaves the beat
            unchanged. The QRS complex is never scaled.
        random_state: Seed for all random operations.

    Returns:
        A ``Record`` with ``lead="ii"``, ``group=None``, ``subject_id=None``.

    Raises:
        ValueError: If ``distortion`` or ``k`` is out of range.
    """
    if distortion is not None and distortion not in _DISTORTIONS:
        raise ValueError(f"distortion must be one of {_DISTORTIONS} or None, got {distortion!r}")
    if not 0.1 <= k <= 1.0:
        raise ValueError(f"k must be in [0.1, 1.0], got {k}")

    # one clean beat, T-wave scaled by k, then repeated -> byte-identical beats
    if hr != 60:
        raise ValueError(f"only hr=60 is supported, got hr={hr}")
    beat = _extract_one_beat(fs, hr, random_state)
    t_amp = _t_amplitude(beat)
    if k != 1.0:
        beat = _scale_t_wave(beat, k)
    signal = np.tile(beat, n_beats)

    # apply the distortion, if any, to the signal
    if distortion is not None:
        rng = np.random.default_rng(random_state)
        if distortion == "gaussian":
            signal = _add_gaussian(signal, t_amp, rng)
        elif distortion == "baseline":
            signal = _add_baseline(signal, t_amp, fs, rng)
        elif distortion == "amplitude_mod":
            signal = _add_amplitude_mod(signal, fs, rng)

    tag = distortion or "clean"
    record_id = f"synth_{tag}_k{k:g}"

    return Record(
        signal=signal, fs=fs, lead="ii", record_id=record_id, group=None, subject_id=None,
    )


def load_synthetic(
    fs: int = 1000, hr: int = 60, n_beats: int = 500, random_state: int = 67,
) -> Iterator[Record]:
    """Lazily yield the full synthetic robustness set.

    Ten T-wave amplitudes (k = 0.1 ... 1.0), each as a clean signal plus one per
    distortion: 10 clean + 10 x 3 distorted = 40 records.

    Yields:
        ``Record`` objects, clean first then distorted, in ascending k order.
    """
    for k in _K_VALUES:
        yield make_synthetic_record(n_beats=n_beats, fs=fs, hr=hr,
                                    distortion=None, k=k, random_state=random_state)
        for dist in ("gaussian", "baseline", "amplitude_mod"):
            yield make_synthetic_record(n_beats=n_beats, fs=fs, hr=hr,
                                        distortion=dist, k=k, random_state=random_state)


def _extract_one_beat(fs: int, hr: int, random_state: int) -> np.ndarray:
    """Second beat of a short ecgsyn stretch, cut R-peak to R-peak and detrended."""
    rr = int(round(fs * 60.0 / hr))
    sig = nk.ecg_simulate(
        duration=4 * 60.0 / hr, sampling_rate=fs, heart_rate=hr,
        heart_rate_std=0.0, noise=0.0, method="ecgsyn", random_state=random_state,
    )
    sig = np.asarray(sig, dtype=float)
    height = 0.6 * sig.max()
    peaks, _ = find_peaks(sig, distance=rr * 0.7, height=height)
    beat = sig[peaks[1]:peaks[1] + rr]
    return beat - np.linspace(0.0, beat[-1] - beat[0], len(beat))



def _t_window_samples(beat: np.ndarray) -> tuple[int, int]:
    """T-wave window in samples: the +/- 3 sigma support of the model's T Gaussian, 31 deg to 169 deg."""
    rr = len(beat)
    half = _T_WINDOW_SIGMAS * _T_SIGMA_DEG
    lo = int((_T_ANGLE_DEG - half) / 360.0 * rr)
    hi = int((_T_ANGLE_DEG + half) / 360.0 * rr)
    return lo, hi


def _scale_t_wave(beat: np.ndarray, k: float) -> np.ndarray:
    """Scale the T-wave amplitude by ``k`` relative to the line between the window edges."""
    lo, hi = _t_window_samples(beat)
    base = np.linspace(beat[lo], beat[hi - 1], hi - lo)
    out = beat.copy()
    out[lo:hi] = base + (beat[lo:hi] - base) * k
    return out


def _t_amplitude(beat: np.ndarray) -> float:
    """T-wave amplitude of the clean beat, relative to the same baseline."""
    lo, hi = _t_window_samples(beat)
    base = np.linspace(beat[lo], beat[hi - 1], hi - lo)
    seg = beat[lo:hi] - base
    return float(np.max(seg))


def _add_gaussian(signal: np.ndarray, t_amp: float, rng: np.random.Generator) -> np.ndarray:
    """White Gaussian noise, SD = 3% of the T-wave amplitude (Baumert 2012, Sec. 2.1)."""
    sigma = _NOISE_FRACTION * t_amp
    return signal + rng.normal(0.0, sigma, size=signal.shape)


def _add_baseline(signal: np.ndarray, t_amp: float, fs: float, rng: np.random.Generator) -> np.ndarray:
    """Sinusoidal baseline wander, amplitude = T amplitude, 0.3 Hz (Baumert 2012, Sec. 2.2)."""
    t = np.arange(len(signal)) / fs
    phase = rng.uniform(0.0, 2.0 * np.pi)
    wander = t_amp * np.sin(2.0 * np.pi * _WANDER_FREQ_HZ * t + phase)
    return signal + wander


def _add_amplitude_mod(signal: np.ndarray, fs: float, rng: np.random.Generator) -> np.ndarray:
    """Sinusoidal amplitude modulation, 0.3 Hz, depth 0.7, mean 1 (Baumert 2012, Sec. 2.3)."""
    lo, hi = _t_window_samples(signal)
    iso = 0.5 * (signal[lo] + signal[hi - 1])

    t = np.arange(len(signal)) / fs
    phase = rng.uniform(0.0, 2.0 * np.pi)
    modulator = 1.0 + _AM_DEPTH * np.sin(2.0 * np.pi * _AM_FREQ_HZ * t + phase)
    return iso + (signal - iso) * modulator
