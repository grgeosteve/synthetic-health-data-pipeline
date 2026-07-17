"""column_utils.py. Shared validation of a DataFrame's columns against the
config's declared column groupings (numeric, binary, categorical), so a
mismatch between the data and the config is caught early with a clear
message, or when a column is declared in more than one group."""

from collections import Counter

import pandas as pd

from src.config import Columns


def validate_column_config(data: pd.DataFrame, columns: Columns) -> None:
    """Validate that the config's column groupings exactly match the data.

    Every column in the data must be declared in exactly one of numeric,
    binary, or categorical, and every declared column must be
    present in the data.

    Args:
        data (pd.DataFrame): The dataset to check.
        columns (Columns): The column-type groupings from config.

    Raises:
        ValueError: If any data column is undeclared, or any declared
            column is missing from the data, or any column is declared in more
            than one group.
    """
    all_declared = list(columns.numeric or []) + list(columns.binary or []) + list(columns.categorical or [])
    group_counts = Counter(all_declared)

    declared = set(all_declared)
    actual = set(data.columns)

    undeclared = actual - declared
    stale = declared - actual
    duplicated = sorted(col for col, count in group_counts.items() if count > 1)

    if undeclared or stale or duplicated:
        errors = []
        if undeclared:
            errors.append(
                f"present in data but not declared in config.columns: {sorted(undeclared)}"
            )
        if stale:
            errors.append(
                f"declared in config.columns but not present in data: {sorted(stale)}"
            )
        if duplicated:
            errors.append(
                f"declared in more than one column group: {duplicated}"
            )
        raise ValueError(
            "Column configuration does not match the data. " + " ".join(errors)
        )
