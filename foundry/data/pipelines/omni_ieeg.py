"""Pipeline for processing the Omni-iEEG dataset into brainset HDF5 recordings.

Omni-iEEG (https://github.com/Omni-iEEG) ships a BIDS-*like* layout
(``sub-*/ses-*/ieeg/*_ieeg.edf`` + companion ``_channels.tsv``/``.json``), but
it is missing files that strict BIDS requires (``dataset_description.json``,
per-run ``_ieeg.json`` sidecars, ``events.tsv``). ``mne_bids`` and
``torch_brain.utils.bids`` (used by e.g. ``NeurosoftPipeline``) refuse to
treat such a directory as a BIDS root, so this pipeline discovers and reads
recordings the same way the upstream ``omni_ieeg`` package's own
``DataFilter``/``sample_usage.py`` do: glob for EDF files and read
``participants.tsv``/``*_channels.tsv`` directly, then load each EDF with
``mne.io.read_raw_edf``.

Each ``*_ieeg.edf`` file is treated as one recording (this matches the
granularity of the dataset's own official split in
``derivatives/datasplit/final_split.csv``, which assigns train/test per EDF
file, not per session). No task-specific target extraction is wired up here:
channel-level ``soz``/``resection``/``anatomical``/``good`` labels and
patient-level metadata (``outcome``, ``dataset_name``, ``task_name``) are
stored as plain attributes on the ``Data`` object for downstream Foundry
target extractors to consume.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pandas as pd

try:
    import mne

    MNE_AVAILABLE = True
except ImportError:
    mne = None
    MNE_AVAILABLE = False

from torch_brain.data import (
    ArrayDict,
    BrainsetDescription,
    Data,
    DeviceDescription,
    Interval,
    SessionDescription,
    SubjectDescription,
    serialize_fn_map,
)
from torch_brain.pipeline import BrainsetPipeline
from torch_brain.utils.mne import (
    extract_channels,
    extract_measurement_date,
    extract_signal,
)

_SEX_CODE_TO_LABEL = {"1": "male", "0": "female"}
_UNKNOWN_CODES = {"-1", "nan", "none", ""}


class OmniIEEGPipeline(BrainsetPipeline):
    """Processes Omni-iEEG's raw EDF recordings into per-recording HDF5 files."""

    brainset_id: str = "omni_ieeg"
    modality = "ieeg"

    #: Line noise frequency to notch-filter out, matching the upstream
    #: ``omni_ieeg.utils.utils_edf.read_raw`` preprocessing.
    notch_freq: float = 60.0

    @classmethod
    def get_manifest(cls, raw_dir: Path, args: Optional[object]) -> pd.DataFrame:
        if not raw_dir.exists():
            raise FileNotFoundError(f"Raw directory '{raw_dir}' does not exist.")

        participants = _load_participants(raw_dir)
        split_by_edf_relpath = _load_official_split(raw_dir)

        manifest_rows = []
        for patient_id in participants["participant_id"]:
            patient_dir = raw_dir / patient_id
            if not patient_dir.exists():
                continue

            for edf_path in sorted(patient_dir.glob("*/ieeg/*_ieeg.edf")):
                recording_id = edf_path.name[: -len("_ieeg.edf")]
                edf_relpath = str(edf_path.relative_to(raw_dir))
                manifest_rows.append({
                    "recording_id": recording_id,
                    "patient_id": patient_id,
                    "edf_relpath": edf_relpath,
                    "split": split_by_edf_relpath.get(edf_relpath),
                })

        if not manifest_rows:
            raise ValueError(f"No iEEG recordings found under {raw_dir}")

        return pd.DataFrame(manifest_rows).set_index("recording_id")

    def download(self, manifest_item) -> dict:
        self.update_status("DOWNLOADING")

        edf_path = self.raw_dir / manifest_item.edf_relpath
        if not edf_path.exists():
            raise FileNotFoundError(f"EDF file not found: {edf_path}")

        return {
            "recording_id": manifest_item.Index,
            "patient_id": manifest_item.patient_id,
            "edf_path": edf_path,
            "split": manifest_item.split,
        }

    def process(self, download_output: dict) -> None:
        if not MNE_AVAILABLE:
            raise ImportError(
                "OmniIEEGPipeline requires the `mne` package. Install it with "
                "`pip install mne`."
            )

        recording_id = download_output["recording_id"]
        patient_id = download_output["patient_id"]
        edf_path = download_output["edf_path"]
        split = download_output["split"]

        self.processed_dir.mkdir(exist_ok=True, parents=True)
        store_path = self.processed_dir / f"{recording_id}.h5"
        if store_path.exists() and not getattr(self.args, "reprocess", False):
            self.update_status("Already Processed")
            return None

        self.update_status("Loading metadata")
        participants = _load_participants(self.raw_dir)
        prow_matches = participants[participants["participant_id"] == patient_id]
        if len(prow_matches) == 0:
            raise ValueError(f"Patient {patient_id!r} not found in participants.tsv")
        prow = prow_matches.iloc[0]

        channels_path = edf_path.with_name(
            edf_path.name.replace("_ieeg.edf", "_channels.tsv")
        )
        if not channels_path.exists():
            self.update_status(f"Skipping {recording_id}: no channels.tsv")
            return None
        channels_df = pd.read_csv(channels_path, sep="\t")

        self.update_status("Reading EDF")
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        raw.notch_filter(self.notch_freq, n_jobs=1, notch_widths=2, verbose=False)

        self.update_status("Extracting signal")
        signal = extract_signal(raw)

        self.update_status("Building channels")
        channels = _build_channels(raw, channels_df)

        self.update_status("Extracting annotations")
        annotations = _extract_annotations(raw)

        try:
            meas_date = extract_measurement_date(raw)
        except Exception:
            meas_date = None

        self.update_status("Building Data object")
        brainset_description = BrainsetDescription(
            id=self.brainset_id,
            origin_version="0.0.1",
            derived_version="1.0.0",
            source="https://github.com/Omni-iEEG",
            description=(
                "Multi-center intracranial EEG dataset for epilepsy research, "
                "with channel-level seizure-onset-zone (soz) and resection "
                "annotations."
            ),
        )
        subject_description = SubjectDescription(
            id=patient_id,
            species="human",
            sex=_sex_from_code(prow.get("sex")),
        )
        session_description = SessionDescription(
            id=recording_id,
            recording_date=meas_date,
        )
        device_description = DeviceDescription(
            id=recording_id,
            recording_tech=_none_if_unknown(prow.get("methods")),
        )

        data = Data(
            brainset=brainset_description,
            subject=subject_description,
            session=session_description,
            device=device_description,
            ieeg=signal,
            channels=channels,
            annotations=annotations,
            domain=signal.domain,
            dataset_name=str(prow.get("dataset", "")),
            task_name=_task_from_recording_id(recording_id),
            outcome=int(prow.get("outcome", -1)),
            split=split if isinstance(split, str) else "unassigned",
        )

        self.update_status("Storing")
        with h5py.File(store_path, "w") as file:
            data.to_hdf5(file, serialize_fn_map=serialize_fn_map)


def _load_participants(raw_dir: Path) -> pd.DataFrame:
    participants_path = raw_dir / "participants.tsv"
    if not participants_path.exists():
        raise FileNotFoundError(f"participants.tsv not found in {raw_dir}")
    participants = pd.read_csv(participants_path, sep="\t")
    participants["participant_id"] = participants["participant_id"].astype(str)
    return participants


def _load_official_split(raw_dir: Path) -> dict[str, str]:
    split_path = raw_dir / "derivatives" / "datasplit" / "final_split.csv"
    if not split_path.exists():
        return {}
    split_df = pd.read_csv(split_path)
    return dict(zip(split_df["edf_name"], split_df["split"]))


def _build_channels(raw: "mne.io.BaseRaw", channels_df: pd.DataFrame) -> ArrayDict:
    """Merge MNE-derived channel metadata with Omni-iEEG's channels.tsv labels."""
    channels_df = channels_df.copy()
    channels_df["name"] = channels_df["name"].astype(str)
    by_name = channels_df.drop_duplicates(subset="name").set_index("name")

    channels = extract_channels(raw)
    ch_ids = channels.id.astype(str)
    n = len(ch_ids)

    soz = np.full(n, -1, dtype=np.int8)
    resection = np.full(n, -1, dtype=np.int8)
    # `anatomical` is a categorical region label (e.g. "fusiform") when known,
    # and the string "-1" when unknown -- unlike soz/resection/good it is not
    # a numeric code, see derivatives/datasplit/anatomical_mapping.csv.
    anatomical = ["-1"] * n
    good = np.full(n, -1, dtype=np.int8)
    sampling_frequency = np.full(n, np.nan, dtype=np.float64)

    missing = []
    for i, ch_id in enumerate(ch_ids):
        if ch_id not in by_name.index:
            missing.append(ch_id)
            continue
        row = by_name.loc[ch_id]
        soz[i] = int(row.get("soz", -1))
        resection[i] = int(row.get("resection", -1))
        anatomical[i] = str(row.get("anatomical", "-1"))
        good[i] = int(row.get("good", -1))
        sampling_frequency[i] = float(row.get("sampling_frequency", np.nan))

    if missing:
        warnings.warn(
            f"{len(missing)}/{n} channel(s) not found in channels.tsv: {missing}",
            stacklevel=2,
        )

    channels.soz = soz
    channels.resection = resection
    channels.anatomical = np.array(anatomical, dtype=str)
    channels.good = good
    channels.sampling_frequency = sampling_frequency
    return channels


def _extract_annotations(raw: "mne.io.BaseRaw") -> Interval:
    annot = raw.annotations
    if annot is None or len(annot) == 0:
        return Interval(start=np.array([]), end=np.array([]))

    onset = np.asarray(annot.onset, dtype=np.float64)
    duration = np.asarray(annot.duration, dtype=np.float64)
    description = np.asarray(annot.description, dtype=object).astype(str)

    return Interval(
        start=onset,
        end=onset + duration,
        description=description,
    )


def _task_from_recording_id(recording_id: str) -> str:
    match = re.search(r"task-([A-Za-z0-9]+)", recording_id)
    return match.group(1) if match else ""


def _sex_from_code(value) -> Optional[str]:
    if value is None:
        return None
    return _SEX_CODE_TO_LABEL.get(str(value).strip())


def _none_if_unknown(value) -> Optional[str]:
    if value is None:
        return None
    value = str(value).strip()
    return None if value.lower() in _UNKNOWN_CODES else value