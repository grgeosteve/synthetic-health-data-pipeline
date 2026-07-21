"""Structural tests for the evaluation harness.

These check shape, bounds, and known-answer behaviour of the metrics, not
exact values (which depend on the data). The Anonymeter attacks are covered by
a single guarded smoke test so the suite stays fast and does not hard-require
the optional privacy dependency.
"""
import numpy as np
import pandas as pd
import pytest

from src.config import Columns
from src.evaluate_synthetic import (
    _association_matrix,
    _association_matrix_mae,
    _correlation_ratio,
    _cramers_v,
    dcr,
    hellinger_distance,
    multivariate_fidelity,
    plausibility_checks,
    total_variation_distance,
    utility_tstr_trtr,
)


@pytest.fixture
def columns() -> Columns:
    return Columns(
        numeric=("age", "avg_glucose_level", "bmi"),
        binary=("hypertension", "heart_disease", "stroke"),
        categorical=("gender", "ever_married", "work_type", "Residence_type", "smoking_status"),
    )


@pytest.fixture
def real_frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 400
    age = rng.uniform(1, 82, n).round(1)
    return pd.DataFrame({
        "age": age,
        "avg_glucose_level": (80 + 0.5 * age + rng.normal(0, 10, n)).round(2),
        "bmi": np.where(rng.random(n) < 0.05, np.nan, rng.uniform(15, 45, n).round(1)),
        "hypertension": (rng.random(n) < 0.1).astype(int),
        "heart_disease": (rng.random(n) < 0.05).astype(int),
        "stroke": (rng.random(n) < 0.05).astype(int),
        "gender": rng.choice(["Male", "Female"], n),
        "ever_married": rng.choice(["Yes", "No"], n),
        "work_type": rng.choice(["Private", "Self-employed", "Govt_job"], n),
        "Residence_type": rng.choice(["Urban", "Rural"], n),
        "smoking_status": rng.choice(["never smoked", "smokes", "Unknown"], n),
    })


# --- univariate metric primitives -----------------------------------------
def test_hellinger_identical_is_zero(real_frame):
    assert hellinger_distance(real_frame["age"],
                              real_frame["age"],
                              is_categorical=False) == pytest.approx(0.0, abs=1e-9)


def test_hellinger_bounded(real_frame):
    other = real_frame["age"] + 1000  # disjoint support
    d = hellinger_distance(real_frame["age"], other, is_categorical=False)
    assert 0.0 <= d <= 1.0


def test_tvd_identical_is_zero(real_frame):
    assert total_variation_distance(real_frame["gender"], real_frame["gender"]) == pytest.approx(0.0)


def test_tvd_disjoint_is_one():
    a = pd.Series(["x"] * 50)
    b = pd.Series(["y"] * 50)
    assert total_variation_distance(a, b) == pytest.approx(1.0)


# --- association primitives -------------------------------------------------
def test_correlation_ratio_bounds_and_zero():
    cats = pd.Series(["a", "b"] * 50)
    constant = pd.Series([3.0] * 100)
    assert _correlation_ratio(cats, constant) == pytest.approx(0.0)


def test_correlation_ratio_perfect_separation():
    cats = pd.Series(["a"] * 50 + ["b"] * 50)
    values = pd.Series([0.0] * 50 + [1.0] * 50)
    assert _correlation_ratio(cats, values) == pytest.approx(1.0)


def test_cramers_v_bounds():
    a = pd.Series(["a", "b"] * 50)
    v = _cramers_v(a, a)
    assert 0.0 <= v <= 1.0 and v == pytest.approx(1.0, abs=1e-6)


def test_association_matrix_is_symmetric_unit_diagonal(real_frame, columns):
    m = _association_matrix(real_frame, columns)
    assert np.allclose(np.diag(m.to_numpy()), 1.0)
    assert np.allclose(m.to_numpy(), m.to_numpy().T)
    assert ((m.to_numpy() >= 0) & (m.to_numpy() <= 1)).all()


def test_identical_frames_have_zero_association_mae(real_frame, columns):
    real_assoc, synth_assoc, named = multivariate_fidelity(real_frame, real_frame, columns)
    assert _association_matrix_mae(real_assoc, synth_assoc) == pytest.approx(0.0, abs=1e-9)
    assert float(named["abs_diff"].max()) == pytest.approx(0.0, abs=1e-9)


# --- utility ----------------------------------------------------------------
def test_utility_returns_expected_keys(real_frame, columns):
    train = real_frame.iloc[:300]
    test = real_frame.iloc[300:]
    result = utility_tstr_trtr(train, test, train, columns, target="stroke", seed=0)
    assert set(result) >= {
        "roc_auc_trtr", "roc_auc_tstr", "roc_auc_gap",
        "pr_auc_trtr", "pr_auc_tstr", "pr_auc_gap", "pr_auc_baseline",
        "f1_trtr", "f1_tstr", "f1_gap",
    }
    # training and scoring on the same distribution => small gap
    if not np.isnan(result["roc_auc_gap"]):
        assert abs(result["roc_auc_gap"]) < 0.5
    if not np.isnan(result["pr_auc_gap"]):
        assert abs(result["pr_auc_gap"]) < 0.5


def test_utility_single_class_synthetic_is_nan(real_frame, columns):
    train = real_frame.iloc[:300]
    test = real_frame.iloc[300:]
    degenerate = train.copy()
    degenerate["stroke"] = 0  # only one class
    result = utility_tstr_trtr(train, test, degenerate, columns, target="stroke", seed=0)
    assert np.isnan(result["pr_auc_tstr"])


def test_utility_pr_auc_baseline_is_test_prevalence(real_frame, columns):
    train = real_frame.iloc[:300]
    test = real_frame.iloc[300:]
    result = utility_tstr_trtr(train, test, train, columns, target="stroke", seed=0)
    expected = float(test["stroke"].astype(int).mean())
    assert result["pr_auc_baseline"] == pytest.approx(expected)


# --- plausibility -----------------------------------------------------------
def test_plausibility_flags_impossible():
    bad = pd.DataFrame({
        "age": [200, 5],
        "bmi": [25, -3],
        "avg_glucose_level": [100, -1],
        "ever_married": ["No", "Yes"],
        "work_type": ["Private", "children"],
    })
    issues = plausibility_checks(bad)
    assert issues["age_out_of_range"] == 1
    assert issues["bmi_out_of_range"] == 1
    assert issues["glucose_non_positive"] == 1
    assert issues["married_child"] == 1  # the 5-year-old married row


# --- privacy: DCR (SDMetrics) -----------------------------------------------
def test_dcr_returns_protection_scores(real_frame, columns):
    train = real_frame.iloc[:300].reset_index(drop=True)
    test = real_frame.iloc[300:].reset_index(drop=True)
    # synthetic = a literal copy of train => overfit => low overfitting protection
    result = dcr(train, test, train.copy(), columns)
    assert set(result) == {"dcr_overfitting_protection", "dcr_baseline_protection"}
    assert 0.0 <= result["dcr_overfitting_protection"] <= 1.0
    assert 0.0 <= result["dcr_baseline_protection"] <= 1.0


def test_dcr_flags_overfitting_on_exact_copy(real_frame, columns):
    train = real_frame.iloc[:300].reset_index(drop=True)
    test = real_frame.iloc[300:].reset_index(drop=True)
    copied = dcr(train, test, train.copy(), columns)
    # a random re-sample of train should look less overfit than an exact copy
    resampled = train.sample(frac=1.0, replace=True, random_state=1).reset_index(drop=True)
    fresh = dcr(train, test, resampled, columns)
    assert copied["dcr_overfitting_protection"] <= fresh["dcr_overfitting_protection"] + 1e-6


# --- privacy: Anonymeter (guarded smoke test) ------------------------------
def test_adversarial_attacks_smoke(real_frame, columns):
    pytest.importorskip("anonymeter")
    from src.evaluate_synthetic import adversarial_attacks
    train = real_frame.iloc[:300].reset_index(drop=True)
    test = real_frame.iloc[300:].reset_index(drop=True)
    result = adversarial_attacks(train, test, train.copy(), n_attacks=50, seed=0)
    assert "singling_out_univariate_risk" in result
    assert "linkability_risk" in result
    assert "inference_risk_mean" in result
    # every *_risk key should have a matching *_reliable flag, success or failure
    risk_keys = {k[:-len("_risk")] for k in result if k.endswith("_risk") and "ci_" not in k}
    for base in risk_keys:
        if f"{base}_reliable" in result:  # skip aggregate keys like inference_risk_mean
            assert isinstance(result[f"{base}_reliable"], bool)
