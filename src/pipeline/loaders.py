
"""Dataset loading for the QTV pipeline.

Every loader returns a ``Record`` with a fixed interface, so the rest of the
pipeline stays dataset-independent.

Loaders: load_ptb / load_ptb_record (PTB), load_kammler_record /
load_kammler_synthetic (Baumert 2012),
load_sjlife_record + sjlife_pairs (SJLIFE).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
from functools import lru_cache

import h5py
import numpy as np
import wfdb


@dataclass
class Record:
    """Single ECG record with a uniform interface for the QTV pipeline.

    Attributes:
        signal: 1-D ECG array for the chosen lead, in mV.
        fs: Sampling frequency in Hz.
        lead: Name of the analysis lead, lower-cased (e.g. ``"ii"``).
        record_id: Unique record identifier (file stem).
        group: Diagnostic class ``"MI"``, ``"healthy"``, or ``"other"``; ``None`` if unknown.
        subject_id: Patient/subject identifier used for de-duplication across records.
    """

    signal: np.ndarray
    fs: float
    lead: str
    record_id: str
    group: str | None = None        # "MI" / "healthy" / "other"
    subject_id: str | None = None    # for later deduplication

    def __repr__(self) -> str:
        n = self.signal.shape[0]
        return (f"Record({self.record_id}, {self.lead}, {self.fs}Hz, "
                f"{n} samples / {n/self.fs:.0f}s, group={self.group})")

#for PTB
def _group_from_header(comments: list[str]) -> str | None:
    """Map a PTB header's "Reason for admission" to "MI", "healthy" or "other", None if absent."""
    for line in comments:
        low = line.lower()
        if "reason for admission" in low:
            if "healthy control" in low:
                return "healthy"
            if "myocardial infarction" in low:
                return "MI"
            return "other"
    return None

def load_ptb_record(record_path: str | Path, lead: str = "ii") -> Record:
    """Load one PTB record from a base path (without file extension).

    Args:
        record_path: Base path to the record, without extension (e.g. ``patient001/s0010_re``).
        lead: ECG lead to extract, case-insensitive. Defaults to ``"ii"``.

    Returns:
        A ``Record`` populated with the signal, metadata, and group label.

    Raises:
        ValueError: If ``lead`` is not present in the record.
    """
    record_path = Path(record_path)
    rec = wfdb.rdrecord(str(record_path))
    #print(vars(rec).keys())

    lead_names = [n.lower() for n in rec.sig_name] #does the requested lead exist?
    if lead.lower() not in lead_names:
        raise ValueError(f"Lead '{lead}' not in {record_path.name}: {rec.sig_name}")

    signal = np.asarray(rec.p_signal[:, lead_names.index(lead.lower())], dtype=float) #1d array for the lead

    return Record(
        signal=signal,
        fs=rec.fs,
        lead=lead.lower(),
        record_id=record_path.name,
        group=_group_from_header(rec.comments),
        subject_id=record_path.parent.name,
    )

def load_ptb(root: str | Path, lead: str = "ii", groups: tuple[str, ...] = ("MI", "healthy")) -> Iterator[Record]:
    """Lazily yield PTB records found under ``root``.

    Walks ``root`` recursively for ``*.hea`` files, skipping records whose group
    is not in ``groups`` or that are missing the requested lead.

    Args:
        root: Root directory of the PTB dataset.
        lead: ECG lead to extract, case-insensitive. Defaults to ``"ii"``.
        groups: Diagnostic groups to include. Defaults to ``("MI", "healthy")``.

    Yields:
        ``Record`` objects in sorted path order.
    """
    for hea in sorted(Path(root).rglob("*.hea")):
        base = hea.with_suffix("")
        header = wfdb.rdheader(str(base))  # text-only, avoids reading the .dat
        if _group_from_header(header.comments) not in groups:
            continue
        if lead.lower() not in [n.lower() for n in header.sig_name]:
            continue
        yield load_ptb_record(base, lead=lead)


#for Kammler synthetic (Baumert 2012 original, via Schmidt/Kammler)
_KAMMLER_FS = 1000.0
_KAMMLER_START = 540  # skip Kammler's own leading padding/settling region
_KAMMLER_DISTORTIONS = {"N": "gaussian", "B": "baseline", "M": "amplitude_mod"}

# quanta -> mV, approximated from Baumert 2012's 957-quanta R-peak at k = 1.0
_KAMMLER_QUANTA_TO_MV = 1 / 1000


def load_kammler_record(
    root: str | Path, k_index: int, distortion_letter: str, duration_s: float = 15.0,
) -> Record:
    """Load a duration-matched slice of one Kammler/Baumert 2012 waveform file.

    Each ``E{k_index:02d}{letter}.mat`` is a continuous 360 s signal: one beat
    at T-wave amplitude k, repeated 500 times, then corrupted by one Baumert
    2012 distortion. k = k_index / 10 (e01=k0.1 ... e10=k1.0), QRS unscaled.

    Args:
        root: Directory containing the ``E*.mat`` waveform files.
        k_index: 1-10, selects ``e{k_index:02d}``.
        distortion_letter: One of ``"N"`` (gaussian), ``"B"`` (baseline),
            ``"M"`` (amplitude_mod).
        duration_s: Length of the slice taken from the 360 s original,
            starting after the leading padding region.

    Returns:
        A ``Record`` (``lead="ii"``, ``group=None``, ``subject_id=None``),
        matching synthetic.py. ``record_id`` is ``kammler_{tag}_k{k:g}``.

    Raises:
        ValueError: If ``distortion_letter`` or ``k_index`` is out of range.
    """
    if distortion_letter not in _KAMMLER_DISTORTIONS:
        raise ValueError(f"distortion_letter must be one of {list(_KAMMLER_DISTORTIONS)}, "
                        f"got {distortion_letter!r}")
    if not 1 <= k_index <= 10:
        raise ValueError(f"k_index must be in 1..10, got {k_index}")

    path = Path(root) / f"E{k_index:02d}{distortion_letter}.mat"
    with h5py.File(path, "r") as f:
        raw = np.asarray(f["data1000"][0], dtype=float)

    n_samples = int(round(duration_s * _KAMMLER_FS))
    signal_mv = raw[_KAMMLER_START:_KAMMLER_START + n_samples] * _KAMMLER_QUANTA_TO_MV

    k = k_index / 10.0
    tag = _KAMMLER_DISTORTIONS[distortion_letter]
    return Record(
        signal=signal_mv,
        fs=_KAMMLER_FS,
        lead="ii",
        record_id=f"kammler_{tag}_k{k:g}",
        group=None,
        subject_id=None,
    )


def load_kammler_synthetic(root: str | Path, duration_s: float = 15.0) -> Iterator[Record]:
    """Lazily yield all 30 Kammler/Baumert 2012 records, duration-matched.

    Distorted recordings only, no clean variant (Schmidt 2018, Fig. 5).

    Yields:
        ``Record`` objects, ascending k, in ``("N", "B", "M")`` order per k.
    """
    for k_index in range(1, 11):
        for letter in ("N", "B", "M"):
            yield load_kammler_record(root, k_index, letter, duration_s=duration_s)


#for SJLIFE (Paired Clinical and Apple Watch ECG Repository, St. Jude Lifetime Cohort)
_SJLIFE_CSV_NAME = "shared_paired_data_243.csv"
_SJLIFE_SOURCES = ("apple", "clinical")
_SJLIFE_FS = {"apple": 512.0, "clinical": 500.0}
_SJLIFE_SUBDIR = {"apple": "AppleECGs_full_243", "clinical": "ClinicalECGs_full_243"}
_SJLIFE_PREFIX = {"apple": "apple_ecg", "clinical": "clinical_ecg"}
_SJLIFE_CLINICAL_LEADS = ("i", "ii", "iii", "avr", "avl", "avf",
                          "v1", "v2", "v3", "v4", "v5", "v6")

# -> mV: apple exports uV, clinical holds raw GE MUSE quanta at 4.88 uV/LSB
_SJLIFE_TO_MV = {"apple": 1e-3, "clinical": 4.88e-3}


@dataclass(frozen=True)
class SJLifePair:
    """One subject's two recordings, with the metadata the CSV carries for them.

    Attributes:
        subject_id: ``sudoID`` from the CSV, shared by both recordings of the
            subject and used as ``Record.subject_id`` for either source.
        index: Numeric id of the subject's two ``.npy`` files, read from the CSV
            and offset by one from the subject number (``PID_<n>`` ->
            ``*_ecg_<n+1>.npy``).
        sex: ``gender_x``.
        race: ``RestingECG_PatientDemographics_Race_``.
        ethnicity: ``ethnic``.
        age: Age at the clinical ECG, in years.
        clinical_hr: Ventricular rate from the clinical report, in bpm.
        apple_hr: Average rate reported for the Apple recording, in bpm.
        minutes_between: Absolute time between the two recordings, in minutes,
            ranging from ~2 min to ~23 h across the cohort.
    """

    subject_id: str
    index: int
    sex: str
    race: str
    ethnicity: str
    age: float | None
    clinical_hr: float | None
    apple_hr: float | None
    minutes_between: float | None


def _sjlife_float(value: str) -> float | None:
    """Parse a CSV cell as float, or None when it is blank or non-numeric."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _sjlife_path(root: str | Path, source: str, index: int) -> Path:
    """Path to one recording, rebuilt from the source and the file index."""
    return Path(root) / _SJLIFE_SUBDIR[source] / f"{_SJLIFE_PREFIX[source]}_{index}.npy"


@lru_cache(maxsize=4)
def _sjlife_manifest(root: str) -> tuple[SJLifePair, ...]:
    """Parse the pairing CSV once per process."""
    pairs = []
    with (Path(root) / _SJLIFE_CSV_NAME).open(newline="") as f:
        for row in csv.DictReader(f):
            index = int(Path(row["apple_loc_ECG_243"]).stem.rsplit("_", 1)[1])
            if int(Path(row["clinical_loc_ECG_243"]).stem.rsplit("_", 1)[1]) != index:
                raise ValueError(f"{row['sudoID']}: apple and clinical file index differ")
            pairs.append(SJLifePair(
                subject_id=row["sudoID"],
                index=index,
                sex=row["gender_x"],
                race=row["RestingECG_PatientDemographics_Race_"],
                ethnicity=row["ethnic"],
                age=_sjlife_float(row["age_at_ecg"]),
                clinical_hr=_sjlife_float(
                    row["RestingECG_OriginalRestingECGMeasurements_VentricularRate_"]),
                apple_hr=_sjlife_float(row["ecgAvgHR"]),
                minutes_between=_sjlife_float(row["Minutes_ABS_Btw_Clinical_Apple"]),
            ))
    return tuple(pairs)


def sjlife_pairs(root: str | Path, sources=_SJLIFE_SOURCES) -> list[SJLifePair]:
    """Subjects whose recordings are present on disk, in CSV order.

    Args:
        root: Directory holding the CSV and the two recording folders.
        sources: Sources that must be present for a subject to be returned.

    Returns:
        One ``SJLifePair`` per usable subject.

    Raises:
        ValueError: If a source name is unknown.
    """
    for source in sources:
        if source not in _SJLIFE_SOURCES:
            raise ValueError(f"Unknown source '{source}', expected one of {_SJLIFE_SOURCES}")
    return [p for p in _sjlife_manifest(str(root))
            if all(_sjlife_path(root, s, p.index).exists() for s in sources)]


def load_sjlife_record(root: str | Path, subject_id: str, source: str,
                       lead: str = "ii") -> Record:
    """Load one subject's recording from one of the two paired sources.

    The two sources are not duration- or lead-matched: the Apple Watch trace is
    30 s of single-lead I at 512 Hz, the clinical one 10 s of 12 leads at
    500 Hz. Neither sampling rate is stored in the data. Amplitudes are scaled
    to mV (see ``_SJLIFE_TO_MV``).

    Args:
        root: Directory holding the CSV and the two recording folders.
        subject_id: ``sudoID`` of the subject, e.g. ``"PID_<n>"``.
        source: ``"apple"`` or ``"clinical"``.
        lead: Clinical lead to extract, case-insensitive; defaults to ``"ii"``
            as elsewhere in the pipeline. Ignored for ``"apple"``, which is
            single-lead and always returns lead I.

    Returns:
        A ``Record`` with the signal in mV. ``subject_id`` is the ``sudoID``,
        identical for a subject's two recordings, ``record_id`` carries the
        source. ``group`` is ``None``, the CSV holds no diagnostic label.

    Raises:
        ValueError: If ``source`` or ``lead`` is unknown, ``subject_id`` is not
            in the CSV, or the stored array does not have the expected shape.
    """
    if source not in _SJLIFE_SOURCES:
        raise ValueError(f"Unknown source '{source}', expected one of {_SJLIFE_SOURCES}")

    pair = next((p for p in _sjlife_manifest(str(root)) if p.subject_id == subject_id), None)
    if pair is None:
        raise ValueError(f"subject '{subject_id}' not in {_SJLIFE_CSV_NAME}")

    raw = np.asarray(np.load(_sjlife_path(root, source, pair.index)), dtype=float)

    if source == "apple":
        if raw.ndim != 1:
            raise ValueError(f"{subject_id}: expected a 1-D apple signal, got {raw.shape}")
        signal, lead_name = raw, "i"
    else:
        leads = raw[0] if raw.ndim == 3 and raw.shape[0] == 1 else raw
        if leads.ndim != 2 or leads.shape[0] != len(_SJLIFE_CLINICAL_LEADS):
            raise ValueError(f"{subject_id}: expected 12 clinical leads, got {raw.shape}")
        lead_name = lead.lower()
        if lead_name not in _SJLIFE_CLINICAL_LEADS:
            raise ValueError(f"Lead '{lead}' not in {_SJLIFE_CLINICAL_LEADS}")
        signal = leads[_SJLIFE_CLINICAL_LEADS.index(lead_name)]

    return Record(
        signal=signal * _SJLIFE_TO_MV[source],
        fs=_SJLIFE_FS[source],
        lead=lead_name,
        record_id=f"sjlife_{source}_{pair.subject_id}",
        group=None,
        subject_id=pair.subject_id,
    )