# synthetic-health-data-pipeline
End-to-end synthetic health data pipeline: generation (CART & CTGAN), and evaluation across fidelity, utility, and disclosure risk.

## Status

Environment setup complete (Python 3.11 + R/renv). Exploratory data analysis complete.
Project statement, data dictionary, and decision log complete.
Dataset preparation complete. Low-fidelity univariate, high-fidelity multivariate (synthpop, CTGAN),
and differentially private (PAC-Synth) synthetic data generation complete.
Evaluation harness complete across all three axes.

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

### CTGAN synthetic data (L3 - multivariate)

`src/generate_ctgan.py` synthesises a multivariate (L3) dataset from
`data/processed/train.csv` using CTGAN (SDV implementation). Column types are declared explicitly
from `configs/config.yaml` rather than relying on auto-detection, keeping the generation consistent
with the rest of the pipeline. CTGAN was used with SDV's parameter defaults deliberately, as a comparison
against the CART-based synthpop generator. Hyper-parameter tuning is strongly advised for best performance.
However, the purpose of this project is to demonstrate the whole synthetic generation process, and not to maximise 
the performance of any specific generator. Missingness is reproduced natively by CTGAN.

Run it with:

```bash
uv run python src/generate_ctgan.py
```

Pass `--config-path` to use a config file other than `configs/config.yaml`:

```bash
uv run python src/generate_ctgan.py --config-path configs/my_config.yaml
```

Output is written to `data/synthetic/ctgan.csv`.

### synthpop (CART) synthetic data (L3 - multivariate)

`src/generate_synthpop.R` synthesises a multivariate (L3) dataset from
`data/processed/train.csv` using synthpop's CART method. Numeric columns are cast to numeric
and binary or categorical columns to factor before generation based on the column type configuration
within `configs/config.yaml`, so synthpop treats them as classification targets rather than continuous
variables. Types are restored after generation to match the real data.
By default synthpop generates column data one at a time, in the sequence they appear in the data.
A custom visit sequence is defined in `configs/config.yaml`, moving `heart_disease`
directly before the target variable (`stroke`), so its generation is conditioned
on every other predictor, including `avg_glucose_level`, `bmi`, and `smoking_status`.
A config without this information will work as well.

Run it with:

```bash
Rscript src/generate_synthpop.R
```

Pass `--config-path` to use a config file other than `configs/config.yaml`:

```bash
Rscript src/generate_synthpop.R --config-path configs/my_config.yaml
```

Output is written to `data/synthetic/synthpop.csv`.

### Differentially private synthetic data (L3 - DP)

`src/generate_dp.py` synthesises a multivariate, differentially private dataset
from `data/processed/train.csv` using SmartNoise's PAC-Synth. PAC-Synth was
used in place of the stronger marginal methods (AIM, MST) because released
smartnoise-synth is currently incompatible with both the old and new
`private-pgm` (`mbi`) APIs. Continuous columns are discretised with fixed,
public clinical bounds (no privacy budget spent learning them) and mapped
back to bin midpoints after sampling. One file is written per epsilon.

Run it with:

```bash
uv run python src/generate_dp.py --epsilons 0.5 1.0 3.0 8.0
```

Output is written to `data/synthetic/dp_eps{epsilon}.csv`. These files are
not committed to the repository.

### Cross-generator compatibility check

`src/check_cross_language_compatibility.py` verifies that every synthetic file matches `data/processed/train.csv`
in dtype, and for every column declared binary or categorical in `configs/config.yaml`, the distinct values
in the synthetic datasets are a subset of the real dataset column values.

Run it after generating any synthetic file:

```bash
uv run python src/check_cross_language_compatibility.py
```

Exits with a non-zero status and reports the exact column and value if any file fails. It prints `PASSED` otherwise.

### Evaluation

`src/evaluate_synthetic.py` evaluates one synthetic file against
`data/processed/train.csv`/`test.csv` across fidelity, utility, and privacy.

Fidelity: univariate (KS, TVD, Hellinger), multivariate (typed association
matrix, named clinical checks), population detection AUC, missingness
comparison, plausibility checks.

Utility: TSTR vs TRTR (ROC-AUC, PR-AUC, F1). PR-AUC is the headline metric
given ~5% stroke prevalence.

Privacy: DCR via SDMetrics, and Anonymeter's three attacks (singling out,
linkability, inference), using `test.csv` as the control set.

Run it with:

```bash
uv run python src/evaluate_synthetic.py --synthetic-path data/synthetic/synthpop.csv --name synthpop
```

`--skip-privacy` skips DCR and the Anonymeter attacks. `--skip-attacks`
runs DCR only, skipping the Anonymeter attacks, since these can be slow or
intractable on heavily-noised (low-epsilon DP) synthetic data.

Results accumulate in `outputs/results.csv`, one row per generator.

Distribution and association-structure figures are written to
`outputs/figures/` per generator (`{name}_distributions.png`,
`{name}_associations.png`).

## Testing

### Pytest (Python)

```bash
uv run pytest
```

Runs the full suite: `tests/test_config.py` (config loading and validation),
`tests/test_io_utils.py` (shared file loading and writing),
`tests/test_column_utils.py` (column-schema validation),
`tests/test_prepare_dataset.py` (data loading, decisions, splitting, and
output writing),
`tests/test_generate_low_fidelity.py` (structural checks on the low-fidelity generator),
`tests/test_generate_ctgan.py` (structural checks on the CTGAN generator),
`tests/test_generate_dp.py` (structural checks on the DP generator),
and `tests/test_evaluate_synthetic.py` (structural checks on every metric in the evaluation harness).

### testthat (R)

```bash
Rscript -e 'testthat::test_file("tests/testthat/test-generate_synthpop.R")'
```

### Compatibility

```bash
uv run python src/check_cross_language_compatibility.py
```

## Documentation

`docs/project_statement.md` states the project's purpose, fidelity levels,
and disclosure risk requirements, written before generation work began.

`docs/data_dictionary.md` documents every variable's type, role, and description.

`docs/decision_log.md` records engineering and tooling decisions made while
building the pipeline. Data decisions are recorded in `notebooks/eda.ipynb`.

