from pathlib import Path

import pandas as pd
import pytest

from src.io_utils import load_csv, write_csv


@pytest.fixture
def sample_df():
    """Provides a sample dataframe for processing and splitting tests."""
    data = {
        "id": list(range(1, 21)),
        "gender": ["Male", "Female", "Other", "Male", "Female", "Male", "Female", "Male", "Female", "Male"] * 2,
        "bmi": [20.1, 22.4, 25.0, None, 21.2, 28.5, 23.1, 24.4, 26.0, 27.1] * 2,
        "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1] * 2
    }
    return pd.DataFrame(data)

@pytest.fixture
def csv_file(tmp_path, sample_df):
    """Writes the sample dataframe to a temporary CSV file."""
    path = tmp_path / "raw_data.csv"
    sample_df.to_csv(path, index=False)
    return path

# --- Tests for load_csv ---

def test_load_csv_success(csv_file):
    df = load_csv(csv_file)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 20

def test_load_csv_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_csv(Path("non_existent.csv"))

def test_load_csv_empty_file(tmp_path):
    empty_file = tmp_path / "empty.csv"
    empty_file.write_text("") # Completely empty
    with pytest.raises(pd.errors.EmptyDataError):
        load_csv(empty_file)

def test_load_csv_parser_error(tmp_path):
    bad_file = tmp_path / "bad.csv"
    # Unbalanced quotes are a reliable way to trigger a pandas.errors.ParserError
    bad_file.write_text('col1,col2\n"unclosed quote,1\n4,5,6')

    with pytest.raises(Exception): # Catch the custom Exception wrapper in load_csv
        load_csv(bad_file)

# --- Tests for write_csv ---

def test_write_csv_success(tmp_path, sample_df):
    path = tmp_path / "out.csv"
    write_csv(path, sample_df)

    assert path.exists()
    reloaded = pd.read_csv(path)
    pd.testing.assert_frame_equal(reloaded, sample_df)


def test_write_csv_creates_parent_directories(tmp_path, sample_df):
    path = tmp_path / "nested" / "deeper" / "out.csv"
    write_csv(path, sample_df)
    assert path.exists()


def test_write_csv_prevents_overwrite(tmp_path, sample_df):
    path = tmp_path / "out.csv"
    path.write_text("existing content")

    with pytest.raises(ValueError, match="already exists"):
        write_csv(path, sample_df)

    # confirm the existing file was not touched
    assert path.read_text() == "existing content"
