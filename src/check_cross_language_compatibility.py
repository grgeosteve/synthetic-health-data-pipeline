"""check_cross_language_compatibility.py. Enforces that every synthetic and
prepared CSV in the pipeline is dtype-compatible when read by pandas,
regardless of the generator or language used in the generation. This is a compatibility
contract."""

import sys
from pathlib import Path

import pandas as pd

from src.config import load_config


def check_compatibility(
    reference_path: Path, candidate_paths: list[Path], categorical_cols: list[str]
) -> list[str]:
    """Compare every candidate file's dtypes and category values against
    the reference file.

    Dtype comparison cannot catch a factor-code substitution bug
    (e.g. binary values silently swapped from 0/1 to 1/2), since the
    corrupted column is still a valid integer column, just with the wrong
    values. Categorical columns are checked to enforce column value set is
    a subset of the reference data values.

    Args:
        reference_path (Path): The file whose dtypes and values are the
            source of truth (train.csv).
        candidate_paths (Path): Files to check against the reference.
        categorical_cols (list[str]): Binary and categorical column names
            (from config.columns) used for the value check.
    Returns:
        list[str]: One message per mismatch found. Empty if fully compatible.
    """
    reference = pd.read_csv(reference_path)
    errors = []

    for path in candidate_paths:
        if not path.exists():
            errors.append(f"{path.name}: file does not exist, skipped")
            continue

        candidate = pd.read_csv(path)

        missing_cols = set(reference.columns) - set(candidate.columns)
        extra_cols = set(candidate.columns) - set(reference.columns)
        if missing_cols:
            errors.append(f"{path.name}: missing columns {sorted(missing_cols)}")
        if extra_cols:
            errors.append(f"{path.name}: unexpected extra columns {sorted(extra_cols)}")

        for col in reference.columns:
            if col not in candidate.columns:
                continue
            ref_dtype = str(reference[col].dtype)
            cand_dtype = str(candidate[col].dtype)
            if ref_dtype != cand_dtype:
                errors.append(
                    f"{path.name}: column '{col}' dtype mismatch, "
                    f"train.csv has {ref_dtype}, {path.name} has {cand_dtype}"
                )

            if col in categorical_cols:
                ref_values = set(reference[col].dropna().unique())
                cand_values = set(candidate[col].dropna().unique())
                unexpected = cand_values - ref_values
                if unexpected:
                    errors.append(
                        f"{path.name}: column '{col}' contains values not present "
                        f"in train.csv: {sorted(unexpected, key=str)}. "
                        f"train.csv's actual values are: {sorted(ref_values, key=str)}. "
                    )

    return errors


def main() -> None:
    config = load_config()
    reference_path = config.paths.processed_dir / "train.csv"

    candidate_paths = sorted(config.paths.synthetic_dir.glob("*.csv"))

    categorical_cols = (config.columns.binary or []) + (config.columns.categorical or [])
    errors = check_compatibility(reference_path, candidate_paths, categorical_cols)

    if errors:
        print("CROSS-LANGUAGE COMPATIBILITY CHECK: FAILED\n")
        for p in errors:
            print(f"  - {p}")
        sys.exit(1)
    else:
        print("CROSS-LANGUAGE COMPATIBILITY CHECK: PASSED")
        print(f"All files match {reference_path.name}'s dtypes exactly.")


if __name__ == "__main__":
    main()
