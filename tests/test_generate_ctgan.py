import numpy as np
import pandas as pd
import pytest

from src.config import Columns
from src.generate_ctgan import build_metadata, generate

SEED = 24
TEST_EPOCHS = 3  # kept minimal so the suite runs quickly, not a quality claim


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


def test_build_metadata_sets_correct_sdtypes(sample_df, columns):
    metadata = build_metadata(sample_df, columns)
    metadata_dict = metadata.to_dict()["tables"]["table"]["columns"]
    assert metadata_dict["age"]["sdtype"] == "numerical"
    assert metadata_dict["bmi"]["sdtype"] == "numerical"
    assert metadata_dict["gender"]["sdtype"] == "categorical"
    assert metadata_dict["stroke"]["sdtype"] == "categorical"


def test_generate_same_columns_as_input(sample_df, columns):
    synthetic, _, _= generate(sample_df, columns, seed=SEED, epochs=TEST_EPOCHS)
    assert list(synthetic.columns) == list(sample_df.columns)


def test_generate_same_row_count_as_input(sample_df, columns):
    synthetic, _, _= generate(sample_df, columns, seed=SEED, epochs=TEST_EPOCHS)
    assert len(synthetic) == len(sample_df)


def test_generate_returns_loss_history_with_expected_epoch_count(sample_df, columns):
    _, losses, _= generate(sample_df, columns, seed=SEED, epochs=TEST_EPOCHS)
    assert len(losses) == TEST_EPOCHS
    assert list(losses.columns) == ["Epoch", "Generator Loss", "Discriminator Loss"]


def test_generate_is_deterministic(sample_df, columns):
    a, _, _ = generate(sample_df, columns, seed=SEED, epochs=TEST_EPOCHS)
    b, _, _ = generate(sample_df, columns, seed=SEED, epochs=TEST_EPOCHS)
    pd.testing.assert_frame_equal(a, b)


def test_generate_reproduces_approximate_missing_rate():
    rng = np.random.default_rng(SEED)
    n = 300
    df = pd.DataFrame({
        "gender": rng.choice(["Male", "Female"], n),
        "age": rng.integers(1, 90, n),
        "bmi": rng.normal(28, 6, n),
        "stroke": rng.integers(0, 2, n),
    })
    mask = rng.random(n) < 0.10  # 10% missing, deliberately injected
    df.loc[mask, "bmi"] = np.nan

    columns = Columns(numeric=["age", "bmi"], binary=["stroke"], categorical=["gender"])
    synthetic, _, _= generate(df, columns, seed=SEED, epochs=TEST_EPOCHS)

    # loose tolerance: few epochs and a small sample, this only confirms
    # missingness is reproduced at all, not that the rate is precise
    assert 0.0 < synthetic["bmi"].isnull().mean() < 0.30