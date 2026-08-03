# Neurosoft Epilepsy — Setup

## Resources

- [Foundry](https://github.com/dianamatata/neurosoft_epilepsy): data structure
- [auditorydecoding](https://github.com/Neurosoft-Bioelectronics/auditorydecoding.git) (`alex/monkeys` branch)
- [Omni-iEEG](https://github.com/Omni-iEEG)
- [Globus data transfer docs (CSCS)](https://docs.cscs.ch/storage/transfer/)
- [CSCS filesystem docs](https://docs.cscs.ch/storage/filesystems/)
- The project uses the [BIDS (Brain Imaging Data Structure)](https://bids.neuroimaging.io/) format for organizing neurophysiology data.


## Setting up the working environment

See also: [CSCS uenv/Python docs](https://docs.cscs.ch/build-install/python/#uenv)

Log in CSCS and Update certificate (renew it daily):

```sh
cscs-key sign
```

---

**1. Fetch the code**

```sh
cd /capstor/scratch/cscs/davalos
git clone --branch feat/omni-ieeg_integration --single-branch https://github.com/dianamatata/neurosoft_epilepsy
cd neurosoft_epilepsy
```

**2. Start a uenv with a view**

```sh
uenv repo create
uenv image find prgenv-gnu
uenv image pull prgenv-gnu-openmpi/26.3:v1
uenv start --view=default prgenv-gnu-openmpi/26.3:v1
```

**3. Fix the Python version**

```sh
unset PYTHONPATH
export PYTHONUSERBASE="$(dirname "$(dirname "$(which python)")")"
```

**4. Create the venv with `uv`** (pointing at your `pyproject.toml`'s folder)

```sh
uv venv --python $(which python) --system-site-packages --seed --relocatable --link-mode=copy .venv
source .venv/bin/activate
```

**5. Sync dependencies from `pyproject.toml`**

```sh
uv sync
uv pip install ipykernel
```

---

### Adding Omni-iEEG repository

```bash
git clone https://github.com/Omni-iEEG/Omni-iEEG.git
cd Omni-iEEG
uv pip install -e .
```

### Adding Omni-iEEG dataset

```bash
python omni_ieeg/dataloader/download_dataset.py \
  --output_dir /Users/avalos/Documents/Programming/neurosoft_epilepsy/data/raw_dir
```

### Adding Neurosoft nsb-epigrid-v1 dataset

```bash
cp -r /capstor/store/cscs/swissai/a0091/sdsc/nsb-epigrid-v1 \
  /capstor/scratch/cscs/davalos/data/raw_dir/neurosoft_nsb-epigrid-v1
```

### Copying data

```bash
scp -r clariden:/capstor/scratch/cscs/davalos/data/processed/omni_ieeg/sub-openieegDetroit006_ses-01_task-sleep.h5 \
  /Users/avalos/Documents/Programming/neurosoft_epilepsy/data/processed/omni_ieeg/sub-openieegDetroit006_ses-01_task-sleep.h5

scp -r clariden:/capstor/scratch/cscs/davalos/data/raw/omni_ieeg/derivatives \
  /Users/avalos/Documents/Programming/neurosoft_epilepsy/data/raw/omni_ieeg/derivatives
```

> Note: instead of copying, the dataset folder can be symlinked to mimic the local setup, e.g. `ln -s /mydata/aqvpa/shared/audio data`.

---

### Switching branch / verifying the environment

```bash
rm uv.lock
uv sync --locked
uv run pytest
```

---

### Adding auditorydecoding repository

**Clone Repo.  Install dependencies and configure `brainsets`. The `brainsets` config file specifies settings for data processing.**

```bash
git clone https://github.com/Neurosoft-Bioelectronics/auditorydecoding.git
cd auditorydecoding
uv sync
uv run brainsets config
```

This prompts for:
- `raw_dir` — location of your BIDS/raw data
- `processed_dir` — location for processed outputs

** Process the data**

Call `uv run brainsets prepare` This will:
- Validate the raw BIDS dataset
- Process iEEG recordings
- Extract relevant features and metadata
- Generate `.h5` files ready for model training

## Project structure and goals

**[Neurosoft-Bioelectronics/auditorydecoding](https://github.com/Neurosoft-Bioelectronics/)**

`auditorydecoding` have a pipeline `NeurosoftPipeline`/`NeurosoftDataset` which is Neurosoft-specific.
One layer down, in `torch_brain` itself (a dependency of `auditorydecoding`), we can find the commands to process BIDS-iEEG datasets into HDF5, and the runtime loader to read them :

- `torch_brain.pipeline.BrainsetPipeline` — the manifest/download/process/HDF5-storage skeleton
- `torch_brain.utils.bids` — `fetch_ieeg_recordings`, `build_bids_path`, `extract_channels`, `load_participants_tsv`, etc. (generic BIDS-iEEG helpers)
- `torch_brain.data.Data`/`Interval`/`IrregularTimeSeries` — the HDF5-serializable containers
- `torch_brain.datasets.Dataset` + `MultiChannelDatasetMixin` — the runtime loader

---

## Transforming Omni-iEEG to HDF5

```bash
export UV_CACHE_DIR=/capstor/scratch/cscs/davalos/.cache/uv

uv run --frozen brainsets prepare -v --local pipelines/omni_ieeg --use-active-env \
    --raw-dir /capstor/scratch/cscs/davalos/data/raw \
    --processed-dir /capstor/scratch/cscs/davalos/data/processed
```

- The pipeline skips any `.h5` that already exists unless `--reprocess` is passed.
- This processes all 464 discovered recordings from the 27GB raw dataset.
- Each `.h5` stores the signal as `float64`, so a 67MB sleep EDF becomes a ~282MB `.h5`.


## Transforming Omni-iEEG and neurosoft_nsb-epigrid-v1 to HDF5

```bash
# TODO: run in a specific job with allocated ressources? why is it not working?
uv run --frozen brainsets prepare -v --local pipelines/neurosoft_nsb-epigrid-v1 --use-active-env \
    --raw-dir /capstor/scratch/cscs/davalos/data/raw \
    --processed-dir /capstor/scratch/cscs/davalos/data/processed
```

---

## On the cluster

```bash
git pull
uv sync --all-groups
```

---

## DONE
- Write a new `OmniIEEGPipeline(BrainsetPipeline)` (analogous to `NeurosoftPipeline`, reusing the generic `torch_brain` BIDS helpers) that processes Omni-iEEG's raw BIDS+EDF data into per-session HDF5 files, plus an `OmniIEEGDataset(MultiChannelDatasetMixin, Dataset)` to load them, and a thin Foundry wrapper in `foundry/data/datasets/omni_ieeg.py`.
- Plot the Omni-iEEG data.
- Check the percentage of Omni-iEEG annotated data.

## STEPS

- reproduce results with PyHFO 
- https://github.com/Omni-iEEG/Omni-iEEG/tree/master/omni_ieeg/event_model/pyhfo_classification


## Status: what's in place

**Pipeline** (`foundry/data/pipelines/omni_ieeg.py` + thin CLI entry `pipelines/omni_ieeg/pipeline.py`)
- Since Omni-iEEG is BIDS-like but missing `dataset_description.json`, `_ieeg.json`, and `events.tsv`, it discovers/reads recordings the same way `sample_usage.py`'s `DataFilter` does (glob + `participants.tsv`/`*_channels.tsv`), rather than via `mne_bids`.
- One HDF5 recording per EDF file (matching the granularity of the dataset's own `final_split.csv`), with channel-level `soz`/`resection`/`anatomical`/`good` labels merged onto MNE's channel list, plus patient-level `outcome`/`dataset_name`/`task_name`/official split.
- No task-specific target extraction is baked in — labels are exposed as plain attributes for a future Foundry target extractor.
- `data/raw_dir/omni_ieeg` is symlinked to `Omni-iEEG_dataset` so the pipeline's `brainset_id` resolves correctly.

**Dataset** (`foundry/data/datasets/omni_ieeg.py`, `OmniIEEGDataset`) — registered in `foundry/data/datasets/__init__.py`.

Caught and fixed while smoke-testing on 3 recordings across different centers (Multicenter, openieeg, hup):
- The `anatomical` channel field isn't a numeric code like `soz`/`resection` — it's a region-name string (e.g. `"fusiform"`) or `"-1"` when unknown. Fixed the pipeline to store it as a string field instead of crashing on `int()`.
- Confirmed `hup` recordings can have bipolar-derivation channel names (e.g. `"F3-F4"`) that don't match `channels.tsv` — handled gracefully with a warning instead of a crash.

**Data quirk worth knowing:** the local `data/raw_dir` is a partial snapshot — `participants.tsv` references 328 patients but only 201 have folders on disk (464 EDFs discovered), and `final_split.csv` has 1026 rows vs 482 EDFs on disk. Not a pipeline bug, just an incomplete local download.

**Notebook** (`notebooks/omni_ieeg_overview.ipynb`) — dataset composition and annotation coverage as percentages (e.g. 14.9% of patients have raw event annotations, 87.8% have SOZ labels), the official split, and a signal plot for one SOZ channel.

**Processed** 3 demo recordings into `data/processed/omni_ieeg/` (~480MB) so the notebook runs out of the box. Running the full pipeline over all ~464 recordings (27GB raw) takes a long time — run separately when ready:

**Not done yet:** `nsb-epigrid-v1` has no Foundry dataset/pipeline — a separate task per the TODO notes above.                                                        





 I want to load with the foundry structure (check foundry repository, and subfolder datasets), the dataset:
- neurosoft_epilepsy/data/raw_dir/nsb-epigrid-v1
helper functions might be found here: /Users/avalos/Documents/Programming/neurosoft_epilepsy/foundry/data/datasets/neurosoft.py 

I want to create a notebook in notebooks which load both datasets and plot one example data for each (1 patient 1 channel for instance).
I want to base this notebook on the example here: neurosoft_epilepsy/auditorydecoding/notebooks/raw_data_visualization.ipynb

The key pair (e.g. ~/.ssh/cscs-key) — generated once, and does not need to be run every time your signed key expires.
The signed certificate — this is what's short-lived. By default, keys (certificates) are valid for 1 day, and you need to generate or sign a new one when needed for continued access.
To sign an existing public key: `cscs-key sign`

davalos@clariden-ln004:/capstor/scratch/cscs/eymericboyer/data/nsb-epigrid-v1> cd sub-01
bash: cd: sub-01: Permission denied >> TODO ask why


