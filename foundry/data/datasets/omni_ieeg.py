from __future__ import annotations

from pathlib import Path
from typing import Callable, Literal, Optional

from torch_brain.datasets import Dataset, MultiChannelDatasetMixin

VALID_SPLITS = ("train", "test", "unassigned")


class OmniIEEGDataset(MultiChannelDatasetMixin, Dataset):
    """Foundry dataset for Omni-iEEG.

    Loads recordings processed by :class:`foundry.data.pipelines.OmniIEEGPipeline`
    (run via ``uv run brainsets prepare --local pipelines/omni_ieeg``).
    Each recording is a single BIDS-style iEEG run (one EDF file).

    No task-specific target extraction is wired up here and channel labels are
    exposed as plain attributes (``data.channels.soz``, etc.) for a Foundry
    target extractor to consume for a specific downstream task.
    """

    def __init__(
        self,
        root: str,
        dirname: str = "omni_ieeg",
        recording_ids: Optional[list[str]] = None,
        transform: Optional[Callable] = None,
        **kwargs,
    ):
        super().__init__(
            dataset_dir=Path(root) / dirname,
            recording_ids=recording_ids,
            transform=transform,
            namespace_attributes=["session.id", "subject.id", "channels.id"],
            **kwargs,
        )

    def get_recording_ids_for_split(
        self, split: Literal["train", "test", "unassigned"]
    ) -> list[str]:
        """Recording IDs assigned to `split` in Omni-iEEG's official split.

        Splits come from the dataset's own
        ``derivatives/datasplit/final_split.csv`` (train/test per EDF file).
        Recordings absent from that file are assigned ``"unassigned"``.
        """
        if split not in VALID_SPLITS:
            raise ValueError(
                f"split must be one of {VALID_SPLITS}, got {split!r}."
            )
        return [
            rid
            for rid in self.recording_ids
            if str(self.get_recording(rid).split) == split
        ]
