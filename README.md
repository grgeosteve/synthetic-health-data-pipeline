# synthetic-health-data-pipeline
End-to-end synthetic health data pipeline: generation (CART & CTGAN), and evaluation across fidelity, utility, and disclosure risk.

## Status

Environment setup complete (Python 3.11 + R/renv). Exploratory data analysis complete.
Pipeline implementation not started.

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
