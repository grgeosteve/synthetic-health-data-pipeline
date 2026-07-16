import pytest
import yaml
from pydantic import ValidationError

from src.config import Columns, Config, Split, load_config


@pytest.fixture
def valid_config_dict():
    """Returns a dictionary representing a valid configuration."""
    return {
        "seed": 24,
        "paths": {
            "raw": "data/raw/test.csv",
            "processed_dir": "data/processed",
            "synthetic_dir": "data/synthetic",
            "outputs_dir": "outputs",
        },
        "split": {
            "test_size": 0.2,
            "stratify_col": "target",
        },
        "columns": {
            "numeric": ["age"],
            "binary": ["binary_col"],
            "categorical": ["cat_col"],
        },
    }

def test_load_config_success(tmp_path, valid_config_dict):
    """Tests that a valid YAML file is loaded correctly into a Config object."""
    config_file = tmp_path / "config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(valid_config_dict, f)

    config = load_config(config_file)
    
    assert isinstance(config, Config)
    assert config.seed == 24
    assert config.split.test_size == 0.2
    assert config.columns.numeric == ("age",)

def test_load_config_file_not_found():
    """Tests that FileNotFoundError is raised when the path does not exist."""
    with pytest.raises(FileNotFoundError):
        load_config("non_existent_file.yaml")

def test_load_config_invalid_yaml(tmp_path):
    """Tests that yaml.YAMLError is raised for malformed YAML files."""
    config_file = tmp_path / "bad_yaml.yaml"
    config_file.write_text("seed: : invalid_yaml") # Invalid YAML syntax

    with pytest.raises(yaml.YAMLError):
        load_config(config_file)

def test_load_config_validation_error(tmp_path, valid_config_dict):
    """Tests that ValidationError is raised when required fields are missing."""
    # Remove a required field
    del valid_config_dict["seed"]
    
    config_file = tmp_path / "invalid_schema.yaml"
    with open(config_file, "w") as f:
        yaml.dump(valid_config_dict, f)

    with pytest.raises(ValidationError):
        load_config(config_file)

def test_split_train_size_property():
    """Tests the calculated train_size property."""
    # Case 1: Only test_size
    split_1 = Split(test_size=0.2, stratify_col="target")
    assert pytest.approx(split_1.train_size) == 0.8

    # Case 2: test_size and val_size
    split_2 = Split(test_size=0.2, val_size=0.1, stratify_col="target")
    assert pytest.approx(split_2.train_size) == 0.7

def test_split_validator_invalid_sizes():
    """Tests that the model_validator catches cases where train_size <= 0."""
    # test_size + val_size = 1.0 (exact)
    with pytest.raises(ValidationError, match="train_size must be greater than 0"):
        Split(test_size=0.5, val_size=0.5, stratify_col="target")

    # test_size + val_size > 1.0
    with pytest.raises(ValidationError, match="train_size must be greater than 0"):
        Split(test_size=0.7, val_size=0.4, stratify_col="target")

def test_split_floating_point_precision():
    """
    Tests that floating point errors (e.g., 1.0 - 0.7 - 0.3)
    are handled by the epsilon tolerance in the validator.
    """
    # In floating point, 1.0 - 0.7 - 0.3 is approx 5.55e-17
    # This should be caught by the 'train_size <= eps' check.
    with pytest.raises(ValidationError, match="train_size must be greater than 0"):
        Split(test_size=0.7, val_size=0.3, stratify_col="target")

def test_columns_at_least_one_validation():
    """Tests that Columns requires at least one element across all lists."""
    # Valid cases: at least one column exists
    Columns(numeric=["a"], binary=[], categorical=[])
    Columns(numeric=None, binary=["b"], categorical=None)
    Columns(numeric=[], binary=None, categorical=["c"])

    # Invalid cases: no columns at all
    with pytest.raises(ValidationError, match="At least one column must be defined"):
        Columns(numeric=[], binary=[], categorical=[])

    with pytest.raises(ValidationError, match="At least one column must be defined"):
        Columns(numeric=None, binary=None, categorical=None)

def test_config_extra_forbid(valid_config_dict):
    """Tests that extra fields are forbidden as per ConfigDict(extra='forbid')."""
    valid_config_dict["unexpected_field"] = "should_fail"
    
    with pytest.raises(ValidationError):
        Config.model_validate(valid_config_dict)