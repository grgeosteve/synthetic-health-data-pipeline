# Decision Log

Engineering and tooling decisions for this pipeline, in the order made. Data decisions
are recorded in `notebooks/eda.ipynb`, not duplicated here.

## 1. Migrated to uv's project workflow, PyTorch pinned to CPU
**Context.** PyPI's default torch wheel is a CUDA build, inconsistent with a
GPU-free, reproducible-anywhere environment. Pinning it via a manual install
step complicates documentation and reproduction of the project, and `requirements.txt`
directives didn't work reliably under uv.

**Decision.** Moved dependencies into `pyproject.toml`, with torch pinned to
a dedicated CPU-only index via uv's source-pinning mechanism. `uv.lock` is
the committed, resolved lockfile. `requirements.txt` was removed.

**Consequence.** Setup is one command (`uv sync`). A transitive dependency
also pulled in a broken old `pyyaml` version during this migration, fixed
with a direct dependency and a forced override.

## 2. Extracted shared CSV loading and writing logic into src/io_utils.py

**Context.** `load_raw` and `write_outputs` in `prepare_dataset.py` implemented
file loading and writing logic which will be needed in synthetic data generation,
and synthetic data evaluation scripts.

**Decision.** Extracted the full versions with comprehensive error handling into
`src/io_utils.py` as `load_csv()` and `write_csv()`, and updated existing consumers to
use it instead of their own copies.

**Consequence.** File-loading and file-writing error handling is defined once. 
Consumers lost no behaviour, and every subsequent processing script will not
duplicate logic.

## 3. write_csv writes an explicit "NA" token instead of a blank field

**Context.** Pandas' to_csv default writes missing values as a blank field.
The files will also be read by R (synthpop). Base R's read.csv
treats a blank field as NA automatically for numeric columns, but not for
character columns, where it imports as a literal empty string unless
na.strings is configured to include it. This project's only column with
missingness (bmi) is numeric, so the blank-field default is not currently
causing a problem, but the categorical synthesis logic already anticipates
categorical missingness occurring later.

**Decision.** write_csv now writes missing values as the literal token
"NA" (na_rep="NA"), R's own default na.strings value, rather than relying
on blank-field interpretation matching across both languages' parsers by
default.

**Consequence.** Tested the change in `write_csv` is handled correctly,
and that when re-read by pandas it is correctly handled as a missing value
(dtype and null positions preserved). This removes a silent-corruption risk for any future
categorical column with missingness, at no cost to the current numeric-only
case.
