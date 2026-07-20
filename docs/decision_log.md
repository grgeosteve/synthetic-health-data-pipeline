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

## 4. CTGAN generator: metadata encoded explicitly

**Context.** SDV can auto-detect each column's type when building
metadata for CTGAN, but auto-detection risks silently diverging from how
`config.columns` classifies each column elsewhere in the pipeline.

**Decision.** The column configuration in `configs/config.yaml` is treated
as a project-wide ground truth to enforce appropriate processing by all methods.
Column types (`sdtype`) are set explicitly from `config.columns`, not SDV's auto-detection.

**Consequence.** This generator cannot silently diverge from how the rest
of the pipeline classifies each column.

## 5. CTGAN generator: defaults kept untuned

**Context.** CTGAN exposes tunable hyperparameters. Tuning them would
likely improve fidelity, but risks the comparison against synthpop
becoming a comparison against a specifically-tuned CTGAN, not a fair
default-vs-default read of the two methods.

**Decision.** CTGAN is run at SDV's library defaults, deliberately
untuned.

**Consequence.** The comparison against synthpop is default-vs-default.
Hyperparameter tuning is out of scope, the goal of this project is to
demonstrate the full generation and evaluation process, not to maximise
any single generator's performance.

## 6. Explicitly define CART's variable visit sequence 

**Context.** synthpop's CART synthesis conditions each variable on every
variable generated before it, so the visit sequence determines each
column's available predictors. Confirmed against Nowok, Raab and Dibben (Administrative
Data Research Centre, Scotland), "Synthetic data in practice, software,
applications and challenges" (RSS, 2017), that this is synthpop's own
documented default behaviour.
The dataset's raw column order happens to
place `stroke` last (maximum available predictors) and resolves every
named EDA relationship (age with bmi, hypertension, and heart_disease,
ever_married with work_type, work_type and ever_married with
smoking_status) in the correct direction, each predictor appearing before
the column it predicts. However, `bmi`, `avg_glucose_level` and `smoking_status`
are ordered *after* `heart_disease` by default, even though they have been shown in the
literature to be cardiovascular risk factors.

**Decision.** Explicitly define visit sequence in `configs/config.yaml`, moving `heart_disease`
directly before the target variable (`stroke`).

**Consequence.** `heart_disease` generation is conditioned on all the potentially contributing
risk factors, and before the target `stroke`. This way conditional generation preserves potential
causal relationships. Output column order is explicitly restored to match the input
(`result$syn[, names(data)]`) as a defensive guarantee, however 
in practice, `synthpop::syn()` already preserves input column order regardless of the visit sequence.

## 7. Cross-generator compatibility check enforced by both d-types and values
**Context.** All generators, whether written in Python or R, load and write in CSV files.
Evaluating the generated datasets requires compatibility on types and values, while the different
languages have inherently different type inference and type-casting mechanisms.
Additionally, checking only d-type equality between the synthetic files and the real data (`train.csv`),
does not catch potential value substitutions. These could potentially happen silently by corrupting
a binary column by casting the values from 0 / 1, to other integers, such as 1 / 2,
which would preserve the categorical status, and d-type (still integer), but the column would contain
wrong values.

**Decision.** Add an explicit cross compatibility check across the generated datasets and the real
`train.csv`, comparing both d-type, and for every categorical and binary column declared in `configs/config.yaml`.
The check verifies that that every synthetic dataset contains matching d-types to the original dataset,
and that the values are a subset of the original's (no unexpected values present).

**Consequence.** Verified against a deliberately corrupted file, the value-set check flags it and names the
exact column and value. This is run after any generator produces a new file.

