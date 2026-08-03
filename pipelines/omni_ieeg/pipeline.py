"""CLI entrypoint for processing Omni-iEEG with `brainsets prepare`.

Run from the repo root, using the active project environment (so that the `foundry` package is importable):
uv run brainsets prepare --local pipelines/omni_ieeg --use-active-env --raw data/raw_dir --processed-dir data/processed

`data/raw_dir` must contain a folder or symlink named `omni_ieeg` (matching
`brainset_id` below) pointing at the Omni-iEEG BIDS-like raw data, e.g.:

ln -s Omni-iEEG_dataset data/raw_dir/omni_ieeg
"""

from foundry.data.pipelines import OmniIEEGPipeline


class Pipeline(OmniIEEGPipeline):
    brainset_id = "omni_ieeg"
