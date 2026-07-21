"""Tests for the differentially private (PAC-Synth) generator.

PAC-Synth is not bit-reproducible from Python,
so there is deliberately no determinism assertion. It also is unstable on very small
inputs (DP noise can drive the record count negative), so fixtures use a
realistic row count. Checks are structural: schema, row count, dtype
preservation, public-bound compliance, spread (numeric columns must retain
real variance, not collapse to a constant, which could be the result of an
over-binned column), and that a column with no real missingness
never gains missingness from privacy suppression.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

from src.config import Columns
from src.generate_dp import NUMERIC_BOUNDS, generate_dp

warnings.filterwarnings("ignore")

SEED = 24
N = 2000  # safely above PAC-Synth's small-sample floor


@pytest.fixture
def columns():
    return Columns(numeric=["age", "bmi"], binary=["stroke"], categorical=["gender"])


@pytest.fixture
def sample_df():
    rng = np.random.default_rng(SEED)
    age = rng.uniform(1, 89, N).round(1)
    return pd.DataFrame({
        "gender": rng.choice(["Male", "Female"], N),
        "age": age,
        "bmi": (22 + 0.1 * age + rng.normal(0, 4, N)).round(1),
        "stroke": (rng.random(N) < 0.15 + 0.004 * age).astype(int),
    })


def test_generate_dp_raises_on_nonpositive_epsilon(sample_df, columns):
    with pytest.raises(ValueError, match="epsilon must be strictly positive"):
        generate_dp(sample_df, columns, epsilon=0.0, seed=SEED)


def test_generate_dp_same_columns_as_input(sample_df, columns):
    synthetic = generate_dp(sample_df, columns, epsilon=3.0, seed=SEED)
    assert list(synthetic.columns) == list(sample_df.columns)


def test_generate_dp_same_row_count_as_input(sample_df, columns):
    synthetic = generate_dp(sample_df, columns, epsilon=3.0, seed=SEED)
    assert len(synthetic) == len(sample_df)


def test_generate_dp_preserves_dtypes(sample_df, columns):
    synthetic = generate_dp(sample_df, columns, epsilon=3.0, seed=SEED)
    for col in sample_df.columns:
        assert synthetic[col].dtype == sample_df[col].dtype, col


def test_generate_dp_numeric_within_public_bounds(sample_df, columns):
    synthetic = generate_dp(sample_df, columns, epsilon=3.0, seed=SEED)
    for col in ["age", "bmi"]:
        lower, upper = NUMERIC_BOUNDS[col]
        values = synthetic[col].dropna()
        assert values.min() >= lower and values.max() <= upper, col


def test_generate_dp_numeric_columns_retain_spread(sample_df, columns):
    """At a reasonably generous epsilon, numeric columns must show real
    variance, not collapse to (near-)constant output - the exact symptom of a
    column being over-binned and dropped by PAC-Synth's marginal selection."""
    synthetic = generate_dp(sample_df, columns, epsilon=8.0, seed=SEED)
    for col in ["age", "bmi"]:
        real_std = sample_df[col].std()
        synth_std = synthetic[col].std()
        assert synth_std > 0.3 * real_std, f"{col}: synthetic std {synth_std:.2f} vs real {real_std:.2f}"


def test_generate_dp_no_spurious_missingness_in_complete_columns(sample_df, columns):
    synthetic = generate_dp(sample_df, columns, epsilon=3.0, seed=SEED)
    assert synthetic["stroke"].isna().sum() == 0
    assert synthetic["gender"].isna().sum() == 0


def test_generate_dp_reproduces_missingness_in_nullable_column(columns):
    rng = np.random.default_rng(SEED)
    age = rng.uniform(1, 89, N).round(1)
    df = pd.DataFrame({
        "gender": rng.choice(["Male", "Female"], N),
        "age": age,
        "bmi": (22 + 0.1 * age + rng.normal(0, 4, N)).round(1),
        "stroke": (rng.random(N) < 0.15).astype(int),
    })
    df.loc[rng.random(N) < 0.20, "bmi"] = np.nan  # heavy, so it survives at eps=8

    synthetic = generate_dp(df, columns, epsilon=8.0, seed=SEED)
    assert list(synthetic.columns) == list(df.columns)
    assert len(synthetic) == len(df)
    assert synthetic["bmi"].isna().sum() > 0
    assert synthetic["stroke"].isna().sum() == 0