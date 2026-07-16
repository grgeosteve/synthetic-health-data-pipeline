from pathlib import Path

import pandas as pd
import pytest

from src.config import Split
from src.prepare_dataset import apply_decisions, load_raw, split, write_outputs


@pytest.fixture
def sample_df():
    """Provides a sample dataframe for processing and splitting tests."""
    # Increased size to 20 rows to ensure split ratios work correctly during stratification
    # 10 targets of 0, 10 targets of 1
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

# --- Tests for load_raw ---

def test_load_raw_success(csv_file):
    df = load_raw(csv_file)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 20

def test_load_raw_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_raw(Path("non_existent.csv"))

def test_load_raw_empty_file(tmp_path):
    empty_file = tmp_path / "empty.csv"
    empty_file.write_text("") # Completely empty
    with pytest.raises(pd.errors.EmptyDataError):
        load_raw(empty_file)

def test_load_raw_parser_error(tmp_path):
    bad_file = tmp_path / "bad.csv"
    # Unbalanced quotes are a reliable way to trigger a pandas.errors.ParserError
    bad_file.write_text('col1,col2\n"unclosed quote,1\n4,5,6')

    with pytest.raises(Exception): # Catch the custom Exception wrapper in load_raw
        load_raw(bad_file)

# --- Tests for apply_decisions ---

def test_apply_decisions(sample_df):
    df = apply_decisions(sample_df.copy())
    
    # 1.) ID should be gone
    assert "id" not in df.columns
    # 2.) 'Other' gender should be gone
    assert (df["gender"] == "Other").sum() == 0
    # 3.) Row count should be 18 (20 total - 2 'Other' entries from new fixture)
    assert len(df) == 18
    # Index should be reset
    assert df.index[0] == 0
    assert df.index[-1] == 17

# --- Tests for split ---

def test_split_no_val(sample_df):
    # Use 0.2 test size (80% train, 20% test)
    split_cfg = Split(test_size=0.2, stratify_col="target")
    
    train, test, val = split(sample_df, split_cfg, seed=42)
    
    assert val is None
    assert len(train) + len(test) == len(sample_df)
    assert len(test) > 0
    assert len(train) > 0

def test_split_with_val(sample_df):
    # 0.2 test, 0.2 val (60% train)
    split_cfg = Split(test_size=0.2, val_size=0.2, stratify_col="target")
    
    train, test, val = split(sample_df, split_cfg, seed=42)
    
    assert val is not None
    assert len(train) + len(test) + len(val) == len(sample_df)
    assert len(train) > 0
    assert len(test) > 0
    assert len(val) > 0

def test_split_raises_on_undersized_group():
    """A stratified group too small for the requested ratios must raise,
    not silently return an empty partition."""
    df = pd.DataFrame({
        "id": [1, 2, 3, 4, 5, 6],
        "gender": ["Male", "Female", "Other", "Male", "Female", "Male"],
        "bmi": [20.1, 22.4, 25.0, None, 21.2, 28.5],
        "target": [0, 1, 0, 1, 0, 1],
    })
    split_cfg = Split(test_size=0.2, val_size=0.2, stratify_col="target")

    with pytest.raises(ValueError, match="produced 0"):
        split(df, split_cfg, seed=42)

def test_split_raises_on_undersized_group_train():
    """Same guard, triggered on the train partition instead of test."""
    df = pd.DataFrame({
        "id": [1, 2],
        "gender": ["Male", "Female"],
        "bmi": [20.1, 22.4],
        "target": [0, 1],
    })
    split_cfg = Split(test_size=0.4, val_size=0.4, stratify_col="target")

    with pytest.raises(ValueError):
        split(df, split_cfg, seed=42)

# --- Tests for write_outputs ---

def test_write_outputs_success(tmp_path, sample_df):
    dst = tmp_path / "processed"
    train = sample_df.sample(frac=0.6)
    test = sample_df.sample(frac=0.2)
    val = sample_df.sample(frac=0.2)
    
    write_outputs(dst, train, test, val)
    
    assert (dst / "train.csv").exists()
    assert (dst / "test.csv").exists()
    assert (dst / "val.csv").exists()

def test_write_outputs_prevents_overwrite(tmp_path, sample_df):
    dst = tmp_path / "processed"
    dst.mkdir()
    (dst / "train.csv").write_text("existing content")
    
    train = sample_df.sample(frac=0.6)
    test = sample_df.sample(frac=0.2)
    
    with pytest.raises(ValueError, match="already exists"):
        write_outputs(dst, train, test, None)