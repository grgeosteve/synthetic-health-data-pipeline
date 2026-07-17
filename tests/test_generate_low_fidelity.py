import numpy as np
import pandas as pd
import pytest

from src.config import Columns
from src.generate_low_fidelity import generate, synthesise_categorical_column, synthesise_numeric_column

SEED = 24

@pytest.fixture
def columns():
    return Columns(
        numeric=["age", "bmi"],
        binary=["stroke"],
        categorical=["gender"],
    )


@pytest.fixture
def sample_df():
    rng = np.random.default_rng(SEED)
    n = 200
    return pd.DataFrame({
        "gender": rng.choice(["Male", "Female"], n),
        "age": rng.integers(1, 90, n),
        "bmi": rng.normal(28, 6, n),
        "stroke": rng.integers(0, 2, n),
    })


# --- Tests for synthesise_numeric_column ---

def test_synthesise_numeric_column_preserves_dtype_no_missing():
    col = pd.Series([10, 20, 30, 40, 50], name="age")
    rng = np.random.default_rng(SEED)
    result = synthesise_numeric_column(col, n=100, rng=rng)
    assert result.dtype == "int64"


def test_synthesise_numeric_column_preserves_dtype_with_missing():
    # explicit Int64 (nullable) construction, since a bare Python None in a
    # plain list upcasts to float64 at construction time.
    col_int_like = pd.Series([10, 20, None, 40, 50], name="age", dtype="Int64")
    rng = np.random.default_rng(SEED)
    result = synthesise_numeric_column(col_int_like, n=100, rng=rng)
    # with missingness, integer columns become nullable Int64, not float64
    assert str(result.dtype) == "Int64"


def test_synthesise_numeric_column_reproduces_approximate_missing_rate():
    values = [10.0] * 75 + [None] * 25  # 25% missing
    col = pd.Series(values, name="bmi")
    rng = np.random.default_rng(SEED)
    result = synthesise_numeric_column(col, n=2000, rng=rng)
    assert abs(result.isnull().mean() - 0.25) < 0.05


def test_synthesise_numeric_column_output_length(sample_df):
    rng = np.random.default_rng(SEED)
    result = synthesise_numeric_column(sample_df["bmi"], n=50, rng=rng)
    assert len(result) == 50


# --- Tests for synthesise_categorical_column ---

def test_synthesise_categorical_column_only_known_categories(sample_df):
    rng = np.random.default_rng(SEED)
    result = synthesise_categorical_column(sample_df["gender"], n=500, rng=rng)
    assert set(result.unique()).issubset(set(sample_df["gender"].unique()))


def test_synthesise_categorical_column_reproduces_approximate_missing_rate():
    values = ["Male"] * 60 + ["Female"] * 20 + [None] * 20  # 20% missing
    col = pd.Series(values, name="gender")
    rng = np.random.default_rng(SEED)
    result = synthesise_categorical_column(col, n=2000, rng=rng)
    assert abs(result.isnull().mean() - 0.20) < 0.05


def test_synthesise_categorical_column_output_length(sample_df):
    rng = np.random.default_rng(SEED)
    result = synthesise_categorical_column(sample_df["stroke"], n=50, rng=rng)
    assert len(result) == 50


# --- Tests for generate (structural, end to end) ---

def test_generate_same_columns_as_input(sample_df, columns):
    synthetic = generate(sample_df, columns, seed=SEED)
    assert list(synthetic.columns) == list(sample_df.columns)


def test_generate_same_row_count_as_input(sample_df, columns):
    synthetic = generate(sample_df, columns, seed=SEED)
    assert len(synthetic) == len(sample_df)


def test_generate_same_dtypes_as_input(sample_df, columns):
    synthetic = generate(sample_df, columns, seed=SEED)
    for col in sample_df.columns:
        assert str(synthetic[col].dtype) == str(sample_df[col].dtype), (
            f"dtype mismatch on '{col}': "
            f"real={sample_df[col].dtype}, synthetic={synthetic[col].dtype}"
        )


def test_generate_is_deterministic(sample_df, columns):
    a = generate(sample_df, columns, seed=SEED)
    b = generate(sample_df, columns, seed=SEED)
    pd.testing.assert_frame_equal(a, b)


def test_generate_destroys_inter_column_correlation(columns):
    # construct data with a strong, deliberate correlation
    rng = np.random.default_rng(SEED)
    n = 500
    age = rng.integers(1, 90, n)
    df = pd.DataFrame({
        "gender": rng.choice(["Male", "Female"], n),
        "age": age,
        "bmi": age * 0.3 + rng.normal(0, 1, n),  # strongly correlated with age
        "stroke": rng.integers(0, 2, n),
    })
    real_corr = df["age"].corr(df["bmi"])
    assert real_corr > 0.8  # confirm the injected correlation is genuinely strong

    synthetic = generate(df, columns, seed=SEED)
    synthetic_corr = synthetic["age"].corr(synthetic["bmi"])
    assert abs(synthetic_corr) < 0.2