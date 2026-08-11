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
# uv pip install -e .
```

**Known packaging bug in the upstream repo:** `Omni-iEEG/setup.py` uses `setuptools.find_packages()`, which only discovers a package if it has an `__init__.py`. 
The top-level `omni_ieeg/` package (and `channel_model/`,`exploratory_model/`) ship without one, so `find_packages()` finds nothing, 
the editable install ends up with an empty package mapping, and `import omni_ieeg` fails with `ModuleNotFoundError` even though `pip install -e .` reported success. (Scripts run from *inside* `Omni-iEEG/`)


Fix once per fresh clone, before installing:
```bash
touch Omni-iEEG/omni_ieeg/__init__.py \
      Omni-iEEG/omni_ieeg/channel_model/__init__.py \
      Omni-iEEG/omni_ieeg/exploratory_model/__init__.py
```
Then install (or re-install after the fix):
```bash
uv pip install -e ./Omni-iEEG
```
Verify:
```bash
python -c "from omni_ieeg.dataloader.download_dataset import download_and_extract"
```

### Adding Omni-iEEG dataset

Use the following command to download the dataset from the huggingface dataset. Please note that our full dataset is very large, around 150GB.

```bash
python omni_ieeg/dataloader/download_dataset.py \
  --output_dir /Users/avalos/Documents/Programming/neurosoft_epilepsy/data/raw_dir
```
To download only annotated data :
```bash
uv run /Users/avalos/Documents/Programming/neurosoft_epilepsy/pipelines/omni_ieeg/download_annotated_omni_ieeg.py
```

### Verifying the annotated data transfer

An annotated recording needs three things on disk under `data/raw/omni_ieeg` (see `_load_hfo_events` in `foundry/data/pipelines/omni_ieeg.py`): 
- the raw folder itself,
- `derivatives/hfo/<...>.csv` (auto-detected HFO candidates)
- `derivatives/hfo_annotation/{train,test}/<edf_name>.parquet` (doctor labels) 

**1. Which annotated participants are missing entirely** (dry-run of the download script, no download performed):
```bash
uv run pipelines/omni_ieeg/download_annotated_omni_ieeg.py --dry-run
```

**2. Which annotated EDFs are missing their HFO derivatives**:
```bash
uv run /Users/avalos/Documents/Programming/neurosoft_epilepsy/pipelines/omni_ieeg/download_annotated_omni_ieeg.py  --dry-run 
```

`notebooks/omni_ieeg_overview.ipynb` reports the resulting annotation coverage as percentages once the data is in place.

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

Note: instead of copying, the dataset folder can be symlinked to mimic the local setup, e.g. `ln -s /mydata/aqvpa/shared/audio data`.

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

RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 uv run --frozen brainsets prepare -v --local pipelines/omni_ieeg --use-active-env \
    --raw-dir /capstor/scratch/cscs/davalos/data/raw \
    --processed-dir /capstor/scratch/cscs/davalos/data/processed
```

**Why `RAY_ENABLE_UV_RUN_RUNTIME_ENV=0` is required:** 
since the driver is launched via `uv run --frozen`, Ray auto-detects this and tries to replicate the exact `uv run` environment on every worker, packaging up the working directory and re-running `uv sync --frozen` + `uv run` inside an isolated copy per worker (time intensive)
That's unnecessary here (`--use-active-env` already means "just use the environment I'm standing in") and actively broken: the packaged copy excludes `.venv` (Ray's own hardcoded default), so the re-provisioned `uv run --frozen` has neither an environment nor a lockfile and every worker crashes on startup. 
- .venv is excluded by Ray, So each worker would still do a full uv sync --frozen from scratch into a fresh venv: very slow
- Not excluding .venv (RAY_OVERRIDE_RUNTIME_ENV_DEFAULT_EXCLUDES='') means packaging and copying your multi-GB .venv (torch, ray, mne, ...) for every worker startup: very slow
- Dropping --frozen avoids the immediate crash but lets uv freely re-resolve dependencies per worker, risking a different resolved version than what the driver is actually running
Setting `RAY_ENABLE_UV_RUN_RUNTIME_ENV=0` disables this auto-replication so workers just inherit the driver's already-active interpreter directly.

- The pipeline skips any `.h5` that already exists unless `--reprocess` is passed but there is a possibility that corrupted h5 are created, and need to cleane these before rerunning. 
- This processes all 594 discovered recordings from the raw dataset.
- Each `.h5` stores the signal as `float64`, so a 67MB sleep EDF becomes a ~282MB `.h5`.

### If some recordings show `FAILED`

The parallel run only prints `<recording_id>: FAILED` — the real traceback goes to the actor's stderr, which isn't captured anywhere reachable (in local Ray mode it streams to whichever terminal/`srun` step launched it, not to a log file). To get the real error for one failing recording, rerun it directly in-process (no Ray, so exceptions print normally):

```bash
RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 uv run --frozen --active python -m torch_brain.pipeline.runner \
  pipelines/omni_ieeg/pipeline.py \
  --raw-dir=/capstor/scratch/cscs/davalos/data/raw \
  --processed-dir=/capstor/scratch/cscs/davalos/data/processed \
  --single=<recording_id>
```

**Watch out:** a crash during `process()` happens *after* `h5py.File(store_path, "w")` has already created the file and after most groups have been written — so a failed recording can leave a large, corrupt-but-existing `.h5` behind (multi-GB partial files, in practice). Since the "already processed" check only tests file *existence*, delete these before rerunning or they'll be silently skipped as done.

`scripts/find_corrupt_h5.py` scans a processed directory, opens every `.h5`, and flags anything unreadable/truncated or missing top-level groups that every other file in the directory has (a majority-vote reference schema. no pipeline-specific knowledge hardcoded, so it works for any brainset's output). Read-only and lightweight enough to run on the login node:

```bash
uv run --frozen --active python scripts/find_corrupt_h5.py /capstor/scratch/cscs/davalos/data/processed/omni_ieeg
# add --delete to remove the broken files it finds
```

```bash
rm /capstor/scratch/cscs/davalos/data/processed/omni_ieeg/<recording_id>.h5
# then rerun the normal brainsets prepare command — it skips everything else and only reprocesses the missing ones
```

Two real bugs were found and fixed this way (both were `TypeError: Object dtype dtype('O') has no native HDF5 equivalent` from `torch_brain`'s `Interval.to_hdf5`, which only knows how to serialize unicode (`'U'`) string arrays, not raw Python-object (`'O'`) arrays):
- `foundry/data/pipelines/omni_ieeg.py`'s `_extract_annotations` defaulted the `description` field to an object-dtype array for recordings with zero embedded MNE annotations — fixed with an explicit `dtype=str`.
- `auditorydecoding/data/neurosoft_pipeline.py`'s `extract_*_trials` functions built `recording_id` arrays with `dtype=object` — same fix. `auditorydecoding` is a pinned git dependency (see `pyproject.toml`), so the installed `.venv` copy needed patching separately from the local `auditorydecoding/` checkout, and either fix will be wiped by the next `uv sync` unless pushed upstream.

### HFO doctor-annotation / detector-candidate mismatch

`_load_hfo_events` in `foundry/data/pipelines/omni_ieeg.py` merges auto-detected HFO candidates (`derivatives/hfo/<recording>.csv`) with doctor-annotated labels (`derivatives/hfo_annotation/{train,test}/<recording>.parquet`) by walking a per-detector pointer into the candidates, in the same order the doctor labels appear — assuming there are always at least as many candidates as doctor-labeled rows per detector. That assumption is violated for a handful of recordings (found for Zurich's `ste` detector and a couple of Detroit `mni` rows), which previously crashed with `StopIteration`. `scripts/audit_hfo_detector_shortage.py` audits every recording's candidate-vs-doctor-label counts per detector and writes `scripts/hfo_detector_shortage_report.csv` — run it after any raw-data changes to check the assumption still holds:

```bash
RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 uv run --frozen --active python scripts/audit_hfo_detector_shortage.py
```

The fix in `_load_hfo_events` keeps only as many doctor-labeled rows per detector as there are candidates available (dropping the unresolvable tail), rather than hardcoding a specific site or detector — this covers today's known shortages and any new ones a future data update might introduce.

**Note:** the HFO-merging feature was added to the pipeline after most of the 594 recordings were already processed, and `process()`'s existence-based skip means those older `.h5` files do **not** include HFO-merged annotations. Delete a recording's `.h5` before rerunning if you need it to reflect the current merging logic.


## Transforming neurosoft_nsb-epigrid-v1 to HDF5
Same `RAY_ENABLE_UV_RUN_RUNTIME_ENV=0` requirement as Omni-iEEG above. Recommendation: for brainsets prepare runs (data download/processing — no GPU needed), grab an interactive compute-node allocation instead of running on the login node:
```bash
salloc -A a0091 -p debug -N 1 -c 4 -t 01:00:00 --uenv-passthrough=use # Account A, nodes N, cpus C, time limit t, and take the user environment to the job
squeue -u davalos # get the jobid_nbr
srun --jobid=jobid_nbr --overlap bash -c 'RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 uv run --frozen brainsets prepare -v --local pipelines/neurosoft_nsb-epigrid-v1 --use-active-env --raw-dir /capstor/scratch/cscs/davalos/data/raw --processed-dir /capstor/scratch/cscs/davalos/data/processed' 2>&1 | tail -60
```
- overlaps: let this step run alongside/overlapping with other steps already using this allocation
- bash -c '...' with ... the command to be executed
debug caps at 1h30 (fine for the small 13-item neurosoft manifest); use -p normal -t 12:00:00 for the larger omni_ieeg-scale runs. This dedicates a full node to you, avoids login-node contention, and matches how this repo already does things (configs/hydra/launcher/slurm_cscs.yaml uses the same a0091 account for training jobs).
kill job when done:
- If you're still inside the salloc shell: just type exit
- If you want to cancel it from elsewhere (e.g., another login-node terminal), first find the job ID with squeue -u davalos, then run scancel <jobid>.
check your current pending jobs: `squeue -u davalos 2>&1`
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

davalos@clariden-ln004:/capstor/scratch/cscs/eymericboyer/data/nsb-epigrid-v1cd sub-01
bash: cd: sub-01: Permission denied >TODO ask why


