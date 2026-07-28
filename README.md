# Set up

## Ressources
- Foundry at https://github.com/dianamatata/neurosoft_epilepsy to keep this data structure
- https://github.com/Neurosoft-Bioelectronics/auditorydecoding.git@alex/monkeys
- data transfert with globus: https://docs.cscs.ch/storage/transfer/
- documentation CSCS: https://docs.cscs.ch/storage/filesystems/
- get Omni iEEG: https://github.com/Omni-iEEG


update certificate:
`cscs-key sign`


### Setting up the working environment

```sh
# Fetch code
cd /capstor/scratch/cscs/davalos
git clone --branch feat/omni-ieeg_integration --single-branch https://github.com/dianamatata/neurosoft_epilepsy
cd neurosoft_epilepsy
```
https://docs.cscs.ch/build-install/python/#uenv

1. Start a uenv with a view (prgenv-gnu-openmpi/26.3:v1)
```sh
uenv repo create
uenv image find prgenv-gnu
uenv image pull prgenv-gnu-openmpi/26.3:v1
uenv start --view=default prgenv-gnu-openmpi/26.3:v1
```

2. Set up the environment so Python behaves predictably
```sh
unset PYTHONPATH
export PYTHONUSERBASE="$(dirname "$(dirname "$(which python)")")"
```

3. Create the venv with uv, pointing at your pyproject.toml's folder

```sh
uv venv --python $(which python) --system-site-packages --seed --relocatable --link-mode=copy .venv
source .venv/bin/activate
```

4. Now sync your actual dependencies from pyproject.toml
```sh
uv sync
uv pip install ipykernel
```

# add omnieeg

```bash
git clone https://github.com/Omni-iEEG/Omni-iEEG.git
cd Omni-iEEG
uv pip install -e .
```
Get the data:

`python omni_ieeg/dataloader/download_dataset.py --output_dir /Users/avalos/Documents/Programming/neurosoft_epilepsy/data/raw_dir
`
# copy data

```bash
cp -r /capstor/store/cscs/swissai/a0091/sdsc/nsb-epigrid-v1 /capstor/scratch/cscs/davalos/data/raw_dir/neurosoft_nsb-epigrid-v1

scp -r davalos@clariden-ln002:/capstor/scratch/cscs/davalos/data/processed/omni_ieeg/sub-openieegDetroit051_ses-01_task-sleep.h5 /Users/avalos/Documents/Programming/neurosoft_epilepsy/data/processed/sub-openieegDetroit051_ses-01_task-sleep.h5 
```
#no: Mimic the local setup by symlinking the dataset folder ln -s /mydata/aqvpa/shared/audio data

# Switch branch, Prepare the Python virtual environment, check Pytest
```bash
rm uv.lock
uv sync --locked
uv run pytest
```

# add auditorydecoding


1. Clone the repository:
```bash
git clone https://github.com/Neurosoft-Bioelectronics/auditorydecoding.git
cd auditorydecoding
```

2. Install dependencies using uv:
3. Before processing data, you need to configure the brainsets module. The brainsets configuration file specifies settings for data processing. Initialize and configure brainsets using the interactive command:
```bash
uv sync
uv run brainsets config
```

This command will prompt you to set the `raw_dir` (location of your BIDS/raw data) and `processed_dir` (location where processed outputs will be stored).

### Processing the Data

Once brainsets is configured, prepare the Neurosoft minipigs 2026 dataset using the following command:

```bash
uv run brainsets prepare --local pipelines/neurosoft_minipigs_2026  --raw <path to the BIDS data>
```

This command will:
- Validate the raw BIDS dataset
- Process iEEG recordings
- Extract relevant features and metadata
- Generate `.h5` files that are ready for model training

## Dataset Structure

The project uses the [BIDS (Brain Imaging Data Structure)](https://bids.neuroimaging.io/) format for organizing neurophysiology data.



# Project structure and Goals

**Neurosoft-Bioelectronics/auditorydecoding**
auditorydecoding doesn't provide a generic "any BIDS dataset" pipeline; its NeurosoftPipeline/NeurosoftDataset are Neurosoft-specific (hardcoded to on_vs_off/acoustic_stim trial extraction). 
What is generic and reusable lives one layer down, in torch_brain itself (a dependency of auditorydecoding):
  - torch_brain.pipeline.BrainsetPipeline — the manifest/download/process/HDF5-storage skeleton                                               
  - torch_brain.utils.bids — fetch_ieeg_recordings, build_bids_path, extract_channels, load_participants_tsv, etc. (generic BIDS-iEEG helpers)
  - torch_brain.data.Data/Interval/IrregularTimeSeries — the HDF5-serializable containers                                                     
  - torch_brain.datasets.Dataset + MultiChannelDatasetMixin — the runtime loader  


# Transform omi-eeg and neurosoft_nsb-epigrid-v1  to hdf5 files

```bash
export UV_CACHE_DIR=/capstor/scratch/cscs/davalos/.cache/uv.

  uv run --frozen brainsets prepare --local pipelines/omni_ieeg --use-active-env \
      --raw-dir /capstor/scratch/cscs/davalos/data/raw_dir \
      --processed-dir /capstor/scratch/cscs/davalos/data/processed

  uv run --frozen brainsets prepare --local pipelines/neurosoft_nsb-epigrid-v1 --use-active-env \
      --raw-dir /capstor/scratch/cscs/davalos/data/raw_dir \
      --processed-dir /capstor/scratch/cscs/davalos/data/processed
```
  - the pipeline skips any .h5 that already exists unless --reprocess is passed. 
  - this will process all 464 discovered recordings from the 27GB raw dataset
  - each .h5 stores the signal as float64, so e.g. that 67MB sleep EDF became a 282MB .h5; expect the
  full processed set to run well into the multi-GB range


# on cluster
git pull
uv sync --all-groups



https://github.com/Neurosoft-Bioelectronics/
Reproduce results?
Dont even need it?
encoder/cnn/which model?
 routine for loading h5 files? 
h5 files need to load them, pre written routines that i can borrow
nwb files instead of hd5 adapted to neurosoft?  they are hd5 files structure as a directory


/capstor/scratch/cscs/eymericboyer/data/omni-ieeg/


### TODO:
- write a new OmniIEEGPipeline(BrainsetPipeline) (analogous to NeurosoftPipeline, reusing the generic torch_brain BIDS helpers) that processes Omni-iEEG's raw BIDS+EDF into per-session HDF5 files, 
- plus an OmniIEEGDataset(MultiChannelDatasetMixin, Dataset) to load them, then a thin Foundry wrapper in foundry/data/datasets/omni_ieeg.py                             

- Understand why not all the scripts are present in the github and i cannot track changes, and add it to the next commit?
- As there are many packages: foundry and Omni-EEG that each have their uv lock and readme, can i get one global uv lock for the whole project? or should i keep them separate?

**Omni-iEEG**
Plot the data
Check the percentage of annotated data

 Summary of what's in place:                                                                           
                                                                                                                                                    
  Pipeline (foundry/data/pipelines/omni_ieeg.py + thin CLI entry pipelines/omni_ieeg/pipeline.py)                                                   
  - Since Omni-iEEG is BIDS-like but missing dataset_description.json/_ieeg.json/events.tsv, it discovers/reads recordings the same way             
  sample_usage.py's DataFilter does (glob + participants.tsv/*_channels.tsv), not via mne_bids.                                                     
  - One HDF5 recording per EDF file (matches the granularity of the dataset's own final_split.csv), with channel-level soz/resection/anatomical/good
  labels merged onto MNE's channel list, plus patient-level outcome/dataset_name/task_name/official split.                                          
  - No task-specific target extraction baked in — labels are exposed as plain attributes for a future Foundry target extractor.                     
  - Symlinked data/raw_dir/omni_ieeg -> Omni-iEEG_dataset so the pipeline's brainset_id resolves correctly.                                         
                                                                                                                                                    
  Dataset (foundry/data/datasets/omni_ieeg.py, OmniIEEGDataset) — registered in foundry/data/datasets/__init__.py.                                  
                                                                                                                                                    
  Caught and fixed while smoke-testing on 3 recordings across different centers (Multicenter, openieeg, hup): the anatomical channel field isn't a  
  numeric code like soz/resection — it's a region name string (e.g. "fusiform") or "-1" when unknown, so I fixed the pipeline to store it as a      
  string field instead of crashing on int(). Also confirmed hup recordings can have bipolar-derivation channel names (e.g. "F3-F4") that don't match
  channels.tsv — handled gracefully with a warning, not a crash.                                                                                    
                                                                                                                                                    
  Data quirk worth knowing: your local data/raw_dir is a partial snapshot — participants.tsv references 328 patients but only 201 have folders on   
  disk (464 EDFs discovered), and final_split.csv has 1026 rows vs 482 EDFs on disk. Not a bug in the pipeline, just incomplete local download.     
                                                                                                                                                    
  Notebook (notebooks/omni_ieeg_overview.ipynb) — dataset composition, and per your request, annotation coverage as percentages (e.g. 14.9% of      
  patients have raw event annotations, 87.8% have SOZ labels), the official split, and a signal plot for one SOZ channel.                           
                                                                                                                                                    
  Processed 3 demo recordings into data/processed/omni_ieeg/ (~480MB) so the notebook runs out of the box. Running the full pipeline over all ~464  
  recordings (27GB raw) would take a long time — that's a separate command for you to run when ready:                                               
  uv run brainsets prepare --local pipelines/omni_ieeg --use-active-env --raw data/raw_dir --processed-dir data/processed                           
                                                                                                                                                    
  Not done yet: nsb-epigrid-v1 has no Foundry dataset/pipeline — that's a separate task per your README notes.                                      
                                                            





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


/Users/avalos/Documents/Programming/neurosoft_epilepsy/

# Foundry

Foundry is a modular brain data experimentation framework designed for flexible neuroscience research. It provides composable building blocks (tokenizers, embeddings, backbones, readouts) and keeps the core minimal so you can focus on your experiments instead of glue code.

**Use Foundry when you want to:**
- Train neural network models on neural data (e.g., EEG, iEEG)
- Experiment with different architectures and tokenization strategies
- Run sweeps across hyperparameters on local or cluster (SLURM) environments
- Log experiments and track results with Weights & Biases or CSV

---

## Quick start

Get up and running in 3 steps:

### 1. Install dependencies

```bash
# Requires Python ≥ 3.11
# Install uv if you don't have it: pip install uv

uv sync
```

This installs Foundry and all required dependencies. Several are downloaded from git (`brainsets`, `torch-brain`, `temporaldata`), so you need internet access.

### 2. Prepare your data

Place processed neural data in `./data/processed/` (or override the path in configs).

### 3. Set up environment variables (optional but recommended for logging)

Create a `.env` file in the project root with your credentials. See **Environment setup** section below for details.

### 4. Run a training experiment

```bash
uv run python main.py experiment=tokenizer_explore/poyo_ajile_sweep
```

That's it! Your model will train and log results to Weights & Biases by default (if credentials are set up). Outputs and checkpoints go to `./outputs` (or to the `SCRATCH` directory if set).

---

## Repository structure

```
foundry/
├── data/                    # Data loading: datasets and datamodules
│   ├── datasets/            # Raw dataset implementations
│   ├── datamodules/         # Lightning datamodules for train/val/test splits
│   └── transforms/          # Preprocessing and patching utilities
├── models/
│   ├── embeddings/          # Token/patch embedding layers
│   ├── backbones/           # Core model architectures (Perceiver, etc.)
│   └── poyo_eeg.py          # Reference neural/brain signal model implementation
├── training/                # Lightning modules and training logic
│   └── module.py            # Main training module for neural networks
├── tools/                   # Utility scripts
│   └── stage_data.py        # Copy/compress data for SLURM cluster
├── callbacks.py             # PyTorch Lightning callbacks
├── config_resolvers.py      # Custom Hydra config helpers
└── core.py                  # Shared utilities

configs/
├── config.yaml              # Root config (composes all groups below)
├── experiment/              # Pre-configured experiment combinations
│   ├── tokenizer_explore/            # Tokenizer-focused experiments
│   ├── neurosoft/                    # Neurosoft project experiments
│   └── ...
├── data/                    # Data configuration by dataset group
│   ├── ajile/
│   ├── neurosoft_minipigs/
│   └── ...
├── model/                   # Model configuration
├── module/                  # Training loop configuration
├── trainer/                 # PyTorch Lightning trainer settings
├── logger/                  # Logging backends (wandb, csv)
├── profiling/               # Performance profiling settings
└── hydra/launcher/          # Cluster job submission (SLURM)

main.py                       # Training entrypoint (Hydra + PyTorch Lightning)
profile_training.py           # Profiling entrypoint (same config system)
pyproject.toml               # Dependencies and project metadata
uv.lock                      # Locked dependency versions (for reproducibility)

tests/                        # Unit tests (pytest)
```

---

## Prerequisites

- **Python 3.11+** (required by PyTorch and core dependencies)
- **uv** (package manager; much faster than pip)
  - Install: `pip install uv`
  - Learn more: [astral.sh/uv](https://docs.astral.sh/uv/)
- **Internet access** (to download git-sourced dependencies during install)

---

## Environment setup

### Standard install (all you need to train models)

```bash
cd /path/to/Foundry
uv sync
```

This reads `pyproject.toml` and `uv.lock` to install Foundry and all dependencies in an isolated virtual environment. It's fast and reproducible.

### Development install (includes testing and linting tools)

```bash
uv sync --group dev
```

This adds `pytest`, `ruff`, and `pre-commit` for development workflows.

### Verify installation

```bash
uv run python -c "import foundry; print(foundry.__version__)"
```

If this succeeds, you're ready to go.

### Setting up Weights & Biases (WandB) for logging

By default, Foundry logs experiments to [Weights & Biases](https://wandb.ai/), a platform for tracking and visualizing machine learning experiments.

**Step 1: Create a WandB account**

1. Go to [https://wandb.ai/](https://wandb.ai/) and sign up for a free account
2. Once logged in, navigate to [https://wandb.ai/authorize](https://wandb.ai/authorize) to generate an API key

**Step 2: Create a `.env` file with your credentials**

Create a file named `.env` in the project root directory (same level as `main.py`) with the following content:

```bash
# .env file - NEVER share this publicly as it contains your API key!
WANDB_API_KEY=<your_api_key_from_wandb>
WANDB_ENTITY=<your_username_or_team_name>
```

Replace:
- `<your_api_key_from_wandb>` with your actual WandB API key
- `<your_username_or_team_name>` with your WandB username or team name

**⚠️ Important security note:** The `.env` file contains sensitive credentials and should **NEVER** be committed to git or publicly shared. It's already listed in `.gitignore`, but double-check if you use other version control systems.

**Step 3: Verify setup**

The first time you run Foundry, it will automatically read the `WANDB_API_KEY` from your `.env` file and authenticate. You should see your experiment appear on the [WandB dashboard](https://wandb.ai/).

**If you prefer not to use WandB**, you can disable it and use CSV logging instead:

```bash
uv run python main.py experiment=tokenizer_explore/poyo_ajile_sweep logger=csv
```

---

## How configuration works

Foundry uses **Hydra**, a configuration framework that lets you compose settings from YAML files without editing code. The config system works by merging defaults in layers:

1. **Root config** ([`configs/config.yaml`](configs/config.yaml)) defines the base structure with required groups:
   - `experiment` (mandatory) — the experiment name you must specify
   - `data`, `model`, `module`, `trainer`, `logger`, `profiling`, `hydra/launcher`

2. **Experiments** ([`configs/experiment/`](configs/experiment/)) combine multiple groups into a ready-to-run setup.
   - Example: `experiment=tokenizer_explore/poyo_ajile_sweep` picks `model=poyo_eeg`, `data=ajile/singlesess`, and SLURM launcher settings.

3. **Config groups** ([`configs/data/`](configs/data/), [`configs/model/`](configs/model/), etc.) define options within each category.

4. **Command-line overrides** let you tweak settings without editing files:
   ```bash
   uv run python main.py experiment=tokenizer_explore/poyo_ajile_sweep data.root=./my_data model.hidden_dim=256
   ```

### Key config groups

| Group | Location | Purpose |
|-------|----------|---------|
| `experiment` | `configs/experiment/` | Pre-composed experiment combinations (which data, model, logger, etc.) |
| `data` | `configs/data/` | Dataset and dataloader settings |
| `model` | `configs/model/` | Model architecture and hyperparameters |
| `module` | `configs/module/` | Training loop (optimizer, loss, learning rate) |
| `trainer` | `configs/trainer/` | PyTorch Lightning trainer settings |
| `logger` | `configs/logger/` | Logging backend (WandB, CSV) |
| `hydra/launcher` | `configs/hydra/launcher/` | Job submission (local or SLURM cluster) |

---

## Running experiments

### Local training (single run)

```bash
uv run python main.py experiment=tokenizer_explore/poyo_ajile_sweep
```

**What happens:**
1. Hydra composes the config from `experiment=tokenizer_explore/poyo_ajile_sweep` and all linked config groups
2. Foundry loads data from `./data/processed/` (or your configured `data.root`)
3. The model trains on GPU (if available) and logs metrics to Weights & Biases by default
4. Checkpoints and logs save to `./outputs/` (or `$SCRATCH/runs/` if the `SCRATCH` environment variable is set)

**Output structure:**
```
./outputs/runs/
├── POYO_AJILE_SWEEP/                    # experiment group
│   └── ajile_poyo_sweep_bs32_lr0.001/   # run name (from config)
│       ├── checkpoints/
│       │   ├── last.ckpt                # last checkpoint (auto-resume from here if preempted)
│       │   └── best-*-*.ckpt            # best checkpoint by validation metric
│       └── .hydra/
│           └── config.yaml              # saved config snapshot
```

### Multi-run sweeps (hyperparameter grid)

To run multiple configurations in parallel (e.g., trying different batch sizes and learning rates):

```bash
uv run python main.py experiment=tokenizer_explore/poyo_ajile_sweep -m
```

The `-m` flag enables **multirun mode**. Hydra will:
1. Parse the sweep parameters from the experiment config (e.g., `batch_size: choice(32, 64, 128)`)
2. Generate all combinations (3 batch sizes × 3 learning rates = 9 runs)
3. Run each one sequentially (or in parallel on SLURM)

Each run gets its own output folder under the experiment group.

**View results on the WandB dashboard:**

Once your experiment starts running and you have set up your `.env` file with WandB credentials, you can view:
- Real-time training curves and metrics
- Compare runs across different experiments
- Download final artifacts and model weights
- Access logs and system information

---

## Common errors and fixes

### Error: `experiment: ???` is mandatory

**Problem:** You ran `uv run python main.py` without specifying an experiment.

**Fix:** Always pass an experiment:
```bash
uv run python main.py experiment=tokenizer_explore/poyo_ajile_sweep
```

### Error: No data found at `./data/processed/`

**Problem:** Foundry can't find your processed dataset.

**Fix:** You have two options:

1. **Override the data path** in your command:
   ```bash
   uv run python main.py experiment=tokenizer_explore/poyo_ajile_sweep data.root=/path/to/my/data
   ```

2. **Create a symbolic link** to your processed data location:
   ```bash
   ln -s /path/to/your/processed/data ./data/processed
   ```
   This is convenient if your data lives elsewhere—just point the symlink to the actual location, and Foundry will find it at `./data/processed/`.

### Error: W&B offline / API key not found

**Problem:** You're using the default `logger=wandb` but your WandB credentials aren't set up properly.

**Fix:**
- Set up your `.env` file with `WANDB_API_KEY` as described in the **Environment setup** section above, or
- Switch to CSV logging:
  ```bash
  uv run python main.py experiment=tokenizer_explore/poyo_ajile_sweep logger=csv
  ```

### Error: GPU out of memory

**Problem:** Model + batch size exceeds GPU memory.

**Fix:**
- Reduce batch size:
  ```bash
  uv run python main.py experiment=tokenizer_explore/poyo_ajile_sweep hyperparameters.batch_size=32
  ```
- Or train on CPU (slow but works):
  ```bash
  uv run python main.py experiment=tokenizer_explore/poyo_ajile_sweep trainer.accelerator=cpu
  ```

### Error: Checkpoint not found during SLURM restart

**Problem:** Job was preempted and `last.ckpt` is missing.

**Fix:** Foundry logs a warning and starts from scratch. This is expected behavior. If you want to manually resume from an old checkpoint:
```bash
uv run python main.py experiment=tokenizer_explore/poyo_ajile_sweep run.resume_if_checkpoint_exists=true
```

### Error: New run with the same name resumes old WandB run

**Problem:** You want multiple independent runs with identical `run.name`.

**Fix:** Leave `run.resume_wandb_if_name_matches=false` (the default). If you do want same-name runs to resume the same WandB history, enable it explicitly:
```bash
uv run python main.py experiment=tokenizer_explore/poyo_ajile_sweep run.resume_wandb_if_name_matches=true
```

---

## Contributing to Foundry

If you want to contribute code changes to Foundry, please follow these quality checks before submitting a pull request:

### Run unit tests

```bash
uv run pytest
```

This runs all tests in the `tests/` folder to verify your changes don't break existing functionality.

### Check code style

```bash
uv run ruff check .
```

This checks for code style violations. The linter is enforced in CI, so please fix any issues it flags.

### Or automatically format code

```bash
uv run ruff format .
```

This automatically formats your code to match the project's style standards. Run this before committing to ensure consistent formatting.

### Install pre-commit hooks (recommended)

```bash
uv sync --group dev
pre-commit install
```

This automatically runs linting and formatting checks before each commit, catching issues early.

### Development workflow

1. Create a feature branch from `main`
2. **Add unit tests** for any new code in the `tests/` folder
   - All new features and bug fixes must include corresponding tests
   - Tests should cover the main functionality and edge cases
   - Run `uv run pytest` to verify your tests pass
3. Make your changes and run the checks above
4. Ensure all tests pass and style checks are clean
5. Submit a pull request with a clear description of your changes

Thank you for contributing!
