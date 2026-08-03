"""CLI entrypoint for processing nsb-epigrid-v1 with `brainsets prepare`.

Run from the repo root, using the active project environment (so that the
`foundry` package is importable):
uv run brainsets prepare --local pipelines/neurosoft_nsb-epigrid-v1 --use-active-env --raw-dir data/raw --processed-dir data/processed

`--raw-dir` must contain a folder or symlink named `neurosoft_nsb-epigrid-v1`
(matching `brainset_id` below) pointing at the nsb-epigrid-v1 BIDS raw data.
"""

from foundry.data.pipelines import NsbEpigridV1Pipeline


class Pipeline(NsbEpigridV1Pipeline):
    brainset_id = "neurosoft_nsb-epigrid-v1"
