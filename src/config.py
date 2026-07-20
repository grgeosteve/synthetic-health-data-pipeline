from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def get_project_root(path: Path) -> Path:
    """ Dynamically resolve the project root directory.

    Args:
        path (Path): The starting path

    Raises:
        FileNotFoundError: If the project root directory cannot be resolved.

    Returns:
        Path: The project root directory 
    """
    for directory in [path, *path.parents]:
        if (directory / "pyproject.toml").exists():
            return directory

    raise FileNotFoundError("could not resolve project root. Missing 'pyproject.toml'")

PROJECT_ROOT = get_project_root(Path(__file__).resolve())
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

class Paths(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    raw: Path
    processed_dir: Path
    synthetic_dir: Path
    outputs_dir: Path

    @field_validator("raw", "processed_dir", "synthetic_dir", "outputs_dir", mode="after")
    @classmethod
    def resolve_absolute_paths(cls, v:Path) -> Path:
        """Resolve paths relative to the project root."""
        if not v.is_absolute():
            return (PROJECT_ROOT / v).resolve()
        return v.resolve()


class Split(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    test_size: float = Field(gt=0.0, lt=1.0)
    val_size: float | None = Field(default=None, gt=0.0, lt=1.0)
    stratify_col: str

    @model_validator(mode="after")
    def check_train_size(self) -> "Split":
        eps = 1e-9 # tolerance for floating-point summation error
        train_size = 1.0 - self.test_size - (self.val_size or 0.0)
        if train_size <= eps:
            raise ValueError(
                f"train_size must be greater than 0, got {train_size} "
                f"(1.0 - test_size {self.test_size} - val_size {self.val_size or 0.0})"
            )
        return self

    @property
    def train_size(self) -> float:
        return 1.0 - self.test_size - (self.val_size or 0.0)
    

class Columns(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    numeric: tuple[str, ...] | None = Field(default=None)
    binary: tuple[str, ...] | None = Field(default=None)
    categorical: tuple[str, ...] | None = Field(default=None)

    @model_validator(mode="after")
    def check_at_least_one_column(self) -> "Columns":
        total_cols = len(self.numeric or []) + len(self.binary or []) + len(self.categorical or [])
        if total_cols == 0:
            raise ValueError("At least one column must be defined across numeric, binary, and categorical.")
        return self

    @property
    def all_columns(self) -> tuple[str, ...]:
        return tuple(self.numeric or ()) + tuple(self.binary or ()) + tuple(self.categorical or ())


class Generation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    visit_sequence: list[str] | None = Field(default=None)


class Config(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    seed: int
    paths: Paths
    split: Split
    columns: Columns
    generation: Generation = Field(default_factory=Generation)

    @model_validator(mode="after")
    def check_visit_sequence_matches_columns(self) -> "Config":
        visit_sequence = self.generation.visit_sequence
        if visit_sequence is None:
            return self

        declared = list(self.columns.all_columns)

        duplicates = sorted({c for c in visit_sequence if visit_sequence.count(c) > 1})
        missing = sorted(set(declared) - set(visit_sequence))
        unexpected = sorted(set(visit_sequence) - set(declared))

        errors = []
        if duplicates:
            errors.append(f"appears more than once in visit_sequence: {duplicates}")
        if missing:
            errors.append(f"declared in config.columns but missing from visit_sequence: {missing}")
        if unexpected:
            errors.append(f"present in visit_sequence but not declared in config.columns: {unexpected}")
 
        if errors:
            raise ValueError(
                "generation.visit_sequence does not match config.columns. " + " ".join(errors)
            )
        return self


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> Config:
    """Load and validate a yaml config file into a frozen, typed Config object.

    Args:
        path (Path, optional): Path to the YAML config file. Defaults to DEFAULT_CONFIG_PATH.

    Raises:
        FileNotFoundError:          If the file is missing 
        yaml.YAMLError:             If it is not valid YAML
        pydantic.ValidationError:   If it fails the pydantic schema validation

    Returns:
        Config: the typed Config object with the config information 
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: '{path}'")
    
    with open(path) as f:
        raw = yaml.safe_load(f)

    return Config.model_validate(raw)
