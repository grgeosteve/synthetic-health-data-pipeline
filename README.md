# synthetic-health-data-pipeline
End-to-end synthetic health data pipeline: generation (CART & CTGAN), and evaluation across fidelity, utility, and disclosure risk.

## Status

Environment setup complete (Python 3.11 + R/renv). Exploratory data analysis complete.
Project statement and data dictionary complete. Decision log in progress, updated as the pipeline is built.
Dataset preparation complete. Low-fidelity synthetic data generation complete. CART and CTGAN generation,
and evaluation not started.

## Setup

*Disclaimer: This environment setup has been tested on Ubuntu 24.04. System dependency instructions for macOS and Windows are provided based on official requirements for uv and CRAN.*

### Python (3.11, CPU-only, via uv)

**1. Install uv (if not already installed)**
 * **macOS/Linux:** `curl -LsSf https://astral.sh/uv/install.sh | sh`
 * **Windows:** `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`

**2. Create the environment and install dependencies**
```bash
uv sync
```

**3. Run commands inside the environment**
```bash
uv run pytest
uv run jupyter lab
```

`uv run` executes inside the project environment without activation. To activate
it in the current shell instead:
```bash
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

The environment is CPU-only by design: no GPU, CUDA, or vendor-specific backend
is required, keeping the pipeline reproducible on any machine.

### R (via renv)

System build dependencies are required to compile package dependencies from source. Install
the requirements for your operating system before opening R:

**Ubuntu/Debian (Tested OS)**
```bash
sudo apt install -y r-base-dev gfortran liblapack-dev libblas-dev
```

**macOS**
1. Install the Xcode Command Line Tools:
```bash
sudo xcode-select --install
```
2. Install the GNU Fortran compiler. Download the official universal installer package matching your R version
(`gfortran-14.2-universal.pkg` for R ≥ 4.5, `gfortran-12.2-universal.pkg` for R 4.3-4.4) from the R-macos GitHub
releases. Do not use Homebrew for this, as it will conflict with CRAN binaries.

**Windows**

Download and install the **Rtools** toolchain that corresponds to your specific R version (e.g., Rtools44 for R 4.4) directly from the
CRAN website. For the latest version of R at the point of development (R 4.6.1) the compatible version is Rtools45.
Proceed with the default installation settings. No terminal commands are required. R detects it automatically.

Then, from an R session in the repository root:

```r
install.packages("renv")   # one-time
renv::restore()            # installs synthpop and dependencies from renv.lock
```

The project R library is isolated via renv (`renv/library/`, backed by
renv's user-level cache). Nothing installs into the system R library.

### Exploratory analysis
`notebooks/eda.ipynb` documents the exploratory data analysis of the raw Stroke Prediction dataset.
It examines variable distributions, associations, missingness, class imbalance and rare-record disclosure risk,
and records the preprocessing decisions that drive the pipeline.

The dataset is the [Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset/data) by fedesoriano (Kaggle). Licence: "Data files © Original Authors". It is not redistributed in this repository.
Download from the source and review the terms on the Kaggle page before use.

To run it:

1. Complete the Python setup above (see [Setup](#setup))
2. Download the dataset and extract it. Place `healthcare-dataset-stroke-data.csv` into `data/raw/`.
3. Launch the notebook:
```bash
uv run jupyter lab notebooks/eda.ipynb
```

Or, with the environment activated:

```bash
jupyter lab notebooks/eda.ipynb
```

4. Run all cells: **Kernel -> Restart & Run All**

### Dataset preparation

`src/prepare_dataset.py` loads the raw dataset, applies the decisions
recorded in the EDA (drop `id`, drop the single `Other` gender row), and
writes stratified train, test, and optional validation splits to
`data/processed/`. Splits are seeded and reproducible, and existing output
files are never overwritten silently.
Stratification is verified at runtime, each split's positive rate on the
stratification column must stay within 0.005 of the overall rate, or the
script raises an error.

Run it with:

```bash
uv run python src/prepare_dataset.py
```

Pass `--config-path` to use a config file other than `configs/config.yaml`:

```bash
uv run python src/prepare_dataset.py --config-path configs/my_config.yaml
```

### Low-fidelity synthetic data (L2 - univariate)

`src/generate_low_fidelity.py` synthesises a low-fidelity dataset from
`data/processed/train.csv`. Each column is fit and sampled independently, numeric
columns via a KDE estimate, and categorical and binary columns from the observed frequency table,
so no inter-variable relationships are preserved. Missingness is reproduced per column at its
observed rate independently. Column groupings are validated against the config file (`configs/config.yaml`)
before generation begins, and generation fails clearly if the config and the data disagree.

Run it with:

```bash
uv run python src/generate_low_fidelity.py
```

Pass `--config-path` to use a config file other than `configs/config.yaml`:

```bash
uv run python src/generate_low_fidelity.py --config-path configs/my_config.yaml
```

Output is written to `data/synthetic/low_fidelity.csv`.

## Testing

```bash
uv run pytest
```

Runs the full suite: `tests/test_config.py` (config loading and validation),
`tests/test_io_utils.py` (shared file loading and writing),
`tests/test_column_utils.py` (column-schema validation),
`tests/test_prepare_dataset.py` (data loading, decisions, splitting, and
output writing), and `tests/test_generate_low_fidelity.py` (structural checks
on the low-fidelity generator).

## Documentation

`docs/project_statement.md` states the project's purpose, fidelity levels,
and disclosure risk requirements, written before generation work began.

`docs/data_dictionary.md` documents every variable's type, role, and description.

`docs/decision_log.md` records engineering and tooling decisions made while
building the pipeline. Data decisions are recorded in `notebooks/eda.ipynb`.

