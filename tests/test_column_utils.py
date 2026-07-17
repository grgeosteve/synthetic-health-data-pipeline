import pandas as pd
import pytest

from src.column_utils import validate_column_config
from src.config import Columns


@pytest.fixture
def columns():
    return Columns(
        numeric=["age", "bmi"],
        binary=["stroke"],
        categorical=["gender"],
    )


@pytest.fixture
def matching_data():
    return pd.DataFrame({
        "gender": ["Male", "Female"],
        "age": [30, 40],
        "bmi": [22.0, 25.0],
        "stroke": [0, 1],
    })


def test_validate_column_config_passes_on_match(matching_data, columns):
    # should not raise
    validate_column_config(matching_data, columns)


def test_validate_column_config_raises_on_undeclared_data_column(matching_data, columns):
    data = matching_data.copy()
    data["undeclared_column"] = [1, 2]

    with pytest.raises(ValueError, match="not declared in config.columns"):
        validate_column_config(data, columns)


def test_validate_column_config_raises_on_stale_config_column(matching_data):
    columns = Columns(
        numeric=["age", "bmi"],
        binary=["stroke"],
        categorical=["gender", "nonexistent_col"],
    )

    with pytest.raises(ValueError, match="not present in data"):
        validate_column_config(matching_data, columns)


def test_validate_column_config_raises_on_column_in_multiple_groups(matching_data):
    """A column declared in more than one group is not caught by a plain
    union check, since set union silently deduplicates. This is worse than
    a validation gap: in generate(), the later group's synthesis silently
    overwrites the earlier one for that column, with no error at all."""
    columns = Columns(
        numeric=["age", "bmi"],
        binary=["stroke"],
        categorical=["gender", "age"],
    )

    with pytest.raises(ValueError, match="more than one column group"):
        validate_column_config(matching_data, columns)


def test_validate_column_config_reports_both_problems_together(matching_data):
    data = matching_data.copy()
    data["undeclared_column"] = [1, 2]

    columns = Columns(
        numeric=["age", "bmi"],
        binary=["stroke"],
        categorical=["gender", "nonexistent_col"],
    )

    with pytest.raises(ValueError) as exc_info:
        validate_column_config(data, columns)

    message = str(exc_info.value)
    assert "not declared in config.columns" in message
    assert "not present in data" in message