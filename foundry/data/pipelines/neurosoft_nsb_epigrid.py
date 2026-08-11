"""Pipeline for processing the Neurosoft nsb-epigrid-v1 human iEEG dataset.

Wraps ``auditorydecoding.NeurosoftPipeline`` (which discovers/reads recordings
via ``torch_brain.utils.bids``, requiring a strict BIDS root) with two
nsb-epigrid-v1-specific pieces:

- ``load_recordings`` is patched to keep a recording's own ``meas_date``
  instead of raising ``KeyError`` when a sidecar is missing the non-standard
  ``OriginalRecordingTimestamp`` field. Some nsb-epigrid-v1 recordings (e.g.
  Comparator/Test acquisitions) were curated without that field.
- the train/test split configuration for this cohort (see
  ``auditorydecoding``'s ``SPLITS_README.md`` for the schema).
"""

import warnings
from datetime import datetime, timezone

import numpy as np
from auditorydecoding import NeurosoftPipeline
from auditorydecoding.data import neurosoft_pipeline as _nsp


def _load_recordings_tolerant(raw_dir, recording_ids, modality):
    """Same as ``neurosoft_pipeline.load_recordings``, except it keeps a
    recording's own ``meas_date`` instead of raising ``KeyError`` when its
    sidecar is missing the non-standard ``OriginalRecordingTimestamp`` field.
    """
    session = {}
    for recording_id in recording_ids:
        bids_path = _nsp.build_bids_path(raw_dir, recording_id, modality)
        raw = _nsp.read_raw_bids(
            bids_path,
            on_ch_mismatch="reorder",
            verbose="CRITICAL",
        )

        if not raw.annotations or len(raw.annotations) == 0:
            if "Baseline" in recording_id:
                warnings.warn(
                    f"No annotations found in baseline recording {recording_id}. Adding baseline annotations."
                )
                _nsp._add_baseline_annotations(raw)
            else:
                warnings.warn(
                    f"No annotations found in recording {recording_id}. Skipping."
                )
                continue
        else:
            if "rest" not in np.unique(raw.annotations.description):
                _nsp._add_rest_annotations(raw)

        timestamp = _nsp.load_json_sidecar(bids_path).get(
            "OriginalRecordingTimestamp"
        )
        if timestamp is None:
            warnings.warn(
                f"{recording_id}: sidecar has no OriginalRecordingTimestamp, "
                "keeping the recording's own meas_date."
            )
        else:
            meas_date = datetime.fromisoformat(timestamp)
            if meas_date.tzinfo is None:
                meas_date = meas_date.replace(tzinfo=timezone.utc)
            raw.set_meas_date(meas_date)

        _nsp._verify_baseline_annotations(raw)
        session[recording_id] = raw

    return _nsp._sort_recordings(session)


_nsp.load_recordings = _load_recordings_tolerant


class NsbEpigridV1Pipeline(NeurosoftPipeline):
    """Processes the Neurosoft nsb-epigrid-v1 human iEEG dataset."""

    brainset_id = "neurosoft_nsb-epigrid-v1"

    split_config = {
        "test_subjects": {"sub-02"},
        "test_subject_early_sessions": {
            "sub-02": {"ses-01", "ses-02"},
        },
        "intersubject_subjects": [
            "sub-01",
            "sub-02",
            "sub-04",
        ],
        "intersession_sessions": {
            "sub-01": ["ses-01", "ses-02"],
            "sub-03": ["ses-02"],
            "sub-04": ["ses-01", "ses-02"],
        },
        "intersession_train_ratio": 0.7,
    }

    skip_sessions = []
