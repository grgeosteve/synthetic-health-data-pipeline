import argparse
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from scipy.stats.contingency import association
from sdmetrics.single_table import DCRBaselineProtection, DCROverfittingProtection
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.column_utils import validate_column_config
from src.config import Columns, load_config
from src.io_utils import load_csv

QUASI_IDENTIFIERS = ["gender", "age", "ever_married", "work_type", "Residence_type"]
SECRET_COLUMNS = ["hypertension", "heart_disease", "smoking_status", "avg_glucose_level", "bmi"]


# ---------------------------------------------------------------------------
# Univariate fidelity
# ---------------------------------------------------------------------------

def hellinger_distance(real: pd.Series, synthetic: pd.Series, is_categorical: bool, bins: int = 20) -> float:
    """Compute the Hellinger distance between a real and synthetic column.

    Args:
        real (pd.Series): The real column.
        synthetic (pd.Series): The synthetic column.
        is_categorical (bool): True for binary or categorical columns,
            False for continuous numeric columns.
        bins (int): Number of quantile bins to use for continuous columns.

    Returns:
        float: Hellinger distance, 0 to 1.
    """
    if not is_categorical:
        combined = pd.concat([real, synthetic])
        edges = np.unique(np.quantile(combined.dropna(), np.linspace(0, 1, bins + 1)))
        real_props = pd.cut(real, edges, include_lowest=True).value_counts(normalize=True, sort=False, dropna=False)
        synth_props = pd.cut(synthetic, edges, include_lowest=True).value_counts(
            normalize=True, sort=False, dropna=False)
    else:
        real_props = real.value_counts(normalize=True, dropna=False)
        synth_props = synthetic.value_counts(normalize=True, dropna=False)

    categories = set(real_props.index) | set(synth_props.index)
    p = np.array([real_props.get(c, 0.0) for c in categories])
    q = np.array([synth_props.get(c, 0.0) for c in categories])
    return float(np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)))

def total_variation_distance(real: pd.Series, synthetic: pd.Series) -> float:
    real_props = real.value_counts(normalize=True, dropna=False)
    synth_props = synthetic.value_counts(normalize=True, dropna=False)
    categories = set(real_props.index) | set(synth_props.index)

    total = 0.0
    for cat in categories:
        total += abs(real_props.get(cat, 0.0) - synth_props.get(cat, 0.0))

    return total / 2

def univariate_fidelity(real: pd.DataFrame,
                        synthetic: pd.DataFrame,
                        columns: Columns) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    rows_numeric = []
    rows_categorical = []

    for col in columns.numeric or []:
        real_non_null = real[col].dropna()
        synth_non_null = synthetic[col].dropna()
        stat, _ = ks_2samp(real_non_null, synth_non_null) # Kolmogorov-Smirnov test
        hellinger = hellinger_distance(real[col], synthetic[col], is_categorical=False)
        rows_numeric.append({
            "column": col,
            "ks_distance": stat,
            "hellinger_distance": hellinger
        })

    for col in (columns.categorical or []) + (columns.binary or []):
        tvd = total_variation_distance(real[col], synthetic[col])
        hellinger = hellinger_distance(real[col], synthetic[col], is_categorical=True)
        rows_categorical.append({
            "column": col,
            "tvd": tvd,
            "hellinger_distance": hellinger
        })

    numeric_results = None
    categorical_results = None
    if rows_numeric:
        numeric_results = pd.DataFrame(rows_numeric).set_index("column")
    if rows_categorical:
        categorical_results = pd.DataFrame(rows_categorical).set_index("column")

    return numeric_results, categorical_results

# ---------------------------------------------------------------------------
# Multivariate fidelity
# ---------------------------------------------------------------------------
def _correlation_ratio(categories: pd.Series, values: pd.Series) -> float:
    """Correlation ratio between a categorical and a numeric column.

    Generalises the point-biserial correlation to more than two categories,
    so a single association measure covers numeric-vs-binary and
    numeric-vs-categorical pairs. Returns the strength in [0, 1] (0 = the
    numeric mean is identical across categories, 1 = category fully
    determines the numeric value). Rows missing either value are dropped.
    """
    df = pd.DataFrame({"cat": categories, "val": values}).dropna()
    if df.empty or df["cat"].nunique() < 2:
        return 0.0
    grand_mean = df["val"].mean()
    ss_total = float(((df["val"] - grand_mean) ** 2).sum())
    if ss_total == 0.0:
        return 0.0
    ss_between = 0.0
    for _, group in df.groupby("cat", observed=True):
        ss_between += len(group) * (group["val"].mean() - grand_mean) ** 2
    return float(np.sqrt(ss_between / ss_total))


def _cramers_v(a: pd.Series, b: pd.Series) -> float:
    """Cramer's V between two discrete columns, in [0, 1].

    Missingness is treated as its own category (dropna=False in the
    crosstab) so a difference in missing structure counts as association,
    consistent with the univariate metrics.
    """
    table = pd.crosstab(a, b, dropna=False)
    if table.shape[0] < 2 or table.shape[1] < 2:
        return 0.0
    return float(association(table.to_numpy(), method="cramer"))


def _pairwise_association(real: pd.DataFrame,
                          col_a: str,
                          col_b: str,
                          type_map: dict[str, str]) -> float:
    """Typed association *strength* in [0, 1] for one pair of columns.

    Types are taken from the config, never inferred:
      continuous x continuous  -> |Pearson r|      (linear strength)
      continuous x discrete    -> correlation ratio (eta / point-biserial)
      discrete   x discrete    -> Cramer's V
    All three are magnitudes in [0, 1], so the real and synthetic matrices
    are directly comparable element-wise. Sign is dropped deliberately.
    It shows whether dependencies are *preserved*, and mixing signed
    Pearson with unsigned Cramer's V in one matrix would be incoherent.
    """
    ta, tb = type_map[col_a], type_map[col_b]
    if ta == "continuous" and tb == "continuous":
        pair = real[[col_a, col_b]].dropna()
        if len(pair) < 2:
            return 0.0
        return float(abs(np.corrcoef(pair[col_a], pair[col_b])[0, 1]))
    if ta == "continuous" and tb == "discrete":
        return _correlation_ratio(real[col_b], real[col_a])
    if ta == "discrete" and tb == "continuous":
        return _correlation_ratio(real[col_a], real[col_b])
    return _cramers_v(real[col_a], real[col_b])


def _association_matrix(data: pd.DataFrame, columns: Columns) -> pd.DataFrame:
    """Full typed association-strength matrix over all configured columns."""
    type_map = {c: "continuous" for c in (columns.numeric or [])}
    type_map.update({c: "discrete" for c in (columns.binary or []) + (columns.categorical or [])})
    names = list(type_map)

    matrix = pd.DataFrame(np.eye(len(names)), index=names, columns=names, dtype=float)
    for i, col_a in enumerate(names):
        for col_b in names[i + 1:]:
            value = _pairwise_association(data, col_a, col_b, type_map)
            matrix.loc[col_a, col_b] = value
            matrix.loc[col_b, col_a] = value
    return matrix

# Clinical relationships called out explicitly in the plan / EDA.
NAMED_CLINICAL_PAIRS = [
    ("age", "stroke"),
    ("hypertension", "stroke"),
    ("heart_disease", "stroke"),
    ("avg_glucose_level", "stroke"),
]


def multivariate_fidelity(real: pd.DataFrame,
                          synthetic: pd.DataFrame,
                          columns: Columns) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare the real and synthetic dependency structure.

    Returns the real and synthetic typed association matrices, and a 
    table of the named clinical checks (real vs synthetic association and the
    absolute gap).
    """
    real_assoc = _association_matrix(real, columns)
    synth_assoc = _association_matrix(synthetic, columns)

    rows = []
    for a, b in NAMED_CLINICAL_PAIRS:
        r = real_assoc.loc[a, b]
        s = synth_assoc.loc[a, b]
        rows.append({
            "pair": f"{a}~{b}",
            "real": r,
            "synthetic": s,
            "abs_diff": abs(r - s),
        })
    named = pd.DataFrame(rows).set_index("pair")
    return real_assoc, synth_assoc, named


def _association_matrix_mae(real_assoc: pd.DataFrame, synth_assoc: pd.DataFrame) -> float:
    """Mean absolute difference over the upper triangle of two matrices."""
    mask = np.triu(np.ones(real_assoc.shape, dtype=bool), k=1)
    return float(np.abs(real_assoc.to_numpy()[mask] - synth_assoc.to_numpy()[mask]).mean())


# ---------------------------------------------------------------------------
# Population-level fidelity, missingness, plausibility
# ---------------------------------------------------------------------------

def _feature_preprocessor(columns: Columns, target: str) -> tuple[ColumnTransformer, list[str]]:
    """One shared preprocessing pipeline for the ML-based checks.

    Numeric  : median imputation with a missingness indicator.
    Binary   : most-frequent imputation, kept as a single 0/1 column.
    Category : most-frequent imputation then one-hot (unknown-safe).
    Fitted per training set by the caller.
    """
    numeric = [c for c in (columns.numeric or []) if c != target]
    binary = [c for c in (columns.binary or []) if c != target]
    categorical = [c for c in (columns.categorical or []) if c != target]

    numeric_pipe = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True))])
    binary_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent"))])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    transformer = ColumnTransformer([
        ("num", numeric_pipe, numeric),
        ("bin", binary_pipe, binary),
        ("cat", categorical_pipe, categorical),
    ])
    return transformer, numeric + binary + categorical


def detection_auc(real: pd.DataFrame,
                  synthetic: pd.DataFrame,
                  columns: Columns,
                  seed: int) -> float:
    """Real-vs-synthetic detection score.

    A classifier is trained to distinguish real rows from synthetic ones. Its
    cross-validated-style AUC on a held-out split is the detection score.
    0.5 indicates that the two are indistinguishable. 1.0 indicates trivial
    separation. Reported as a single population-level fidelity number.
    """
    target = "__is_real__"
    combined_features = [c for c in columns.all_columns]
    real_x = real[combined_features].copy()
    synth_x = synthetic[combined_features].copy()
    real_x[target] = 1
    synth_x[target] = 0
    data = pd.concat([real_x, synth_x], ignore_index=True)

    from sklearn.model_selection import train_test_split
    tr, te = train_test_split(data, test_size=0.3, random_state=seed, stratify=data[target])

    pre, _ = _feature_preprocessor(columns, target="__none__")
    clf = Pipeline([("pre", pre), ("rf", RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1))])
    clf.fit(tr[combined_features], tr[target])
    proba = clf.predict_proba(te[combined_features])[:, 1]
    return float(roc_auc_score(te[target], proba))

def missingness_comparison(real: pd.DataFrame, synthetic: pd.DataFrame, columns: Columns) -> pd.DataFrame:
    """Per-column missing-rate comparison (real vs synthetic)."""
    rows = []
    for col in columns.all_columns:
        r = float(real[col].isna().mean())
        s = float(synthetic[col].isna().mean())
        rows.append({"column": col, "real_na_rate": r, "synthetic_na_rate": s, "abs_diff": abs(r - s)})
    return pd.DataFrame(rows).set_index("column")

def plausibility_checks(synthetic: pd.DataFrame) -> dict[str, int]:
    """Domain out-of-range and impossible-combination counts.

    Bounds are clinical sanity limits for the Stroke dataset, kept explicit
    and conservative. Reported as counts.
    """
    issues: dict[str, int] = {}
    if "age" in synthetic:
        issues["age_out_of_range"] = int(((synthetic["age"] < 0) | (synthetic["age"] > 120)).sum())
    if "bmi" in synthetic:
        issues["bmi_out_of_range"] = int(((synthetic["bmi"] <= 0) | (synthetic["bmi"] > 100)).sum())
    if "avg_glucose_level" in synthetic:
        issues["glucose_non_positive"] = int((synthetic["avg_glucose_level"] <= 0).sum())
    if {"age", "ever_married"}.issubset(synthetic.columns):
        issues["married_child"] = int(((synthetic["age"] < 16) & (synthetic["ever_married"] == "Yes")).sum())
    if {"age", "work_type"}.issubset(synthetic.columns):
        issues["adult_labelled_children"] = int(
            ((synthetic["age"] > 18) & (synthetic["work_type"] == "children")).sum()
        )
    return issues


# ---------------------------------------------------------------------------
# Utility: Train-on-Synthetic-Test-on-Real vs Train-on-Real-Test-on-Real
# ---------------------------------------------------------------------------
def utility_tstr_trtr(train: pd.DataFrame,
                      test: pd.DataFrame,
                      synthetic: pd.DataFrame,
                      columns: Columns,
                      target: str = "stroke",
                      seed: int = 0) -> dict[str, float]:
    """TSTR vs TRTR downstream utility on the real test set.

    TRTR trains on the real train split, TSTR on the synthetic set.
    Both are scored on the real test split. The TSTR-TRTR gap is the
    utility loss: near zero means the synthetic data is as useful as
    the real data for this task.

    ROC-AUC, PR-AUC, and F1 (positive/minority class) are all reported because
    the target is rare (~5% strokes).
    """
    features = [c for c in columns.all_columns if c != target]
    prevalence = float(test[target].astype(int).mean())

    def _fit_score(train_df: pd.DataFrame) -> tuple[float, float, float]:
        if train_df[target].nunique() < 2:
            return float("nan"), float("nan"), float("nan")

        model = Pipeline([
            ("pre", _feature_preprocessor(columns, target=target)[0]),
            ("rf", RandomForestClassifier(n_estimators=300, random_state=seed,
                                          n_jobs=-1, class_weight="balanced")),
        ])
        model.fit(train_df[features], train_df[target].astype(int))

        y_true = test[target].astype(int)
        positive_col = list(model.classes_).index(1)
        proba = model.predict_proba(test[features])[:, positive_col]
        preds = model.predict(test[features])

        roc_auc = roc_auc_score(y_true, proba)
        pr_auc = average_precision_score(y_true, proba)
        f1 = f1_score(y_true, preds, pos_label=1, zero_division=0)
        return roc_auc, pr_auc, f1

    roc_auc_trtr, pr_auc_trtr, f1_trtr = _fit_score(train)
    roc_auc_tstr, pr_auc_tstr, f1_tstr = _fit_score(synthetic)

    return {
        "pr_auc_baseline": prevalence,  # random-classifier PR-AUC, for scale
        "roc_auc_trtr": roc_auc_trtr,
        "roc_auc_tstr": roc_auc_tstr,
        "roc_auc_gap": roc_auc_trtr - roc_auc_tstr,
        "pr_auc_trtr": pr_auc_trtr,
        "pr_auc_tstr": pr_auc_tstr,
        "pr_auc_gap": pr_auc_trtr - pr_auc_tstr,
        "f1_trtr": f1_trtr,
        "f1_tstr": f1_tstr,
        "f1_gap": f1_trtr - f1_tstr,
    }


# ---------------------------------------------------------------------------
# Privacy: Distance to Closest Record
# ---------------------------------------------------------------------------
def _sdv_metadata(columns: Columns) -> dict:
    """SDV single-table metadata built from the column config, for sdmetrics."""
    sdtype: dict[str, dict] = {}
    for col in (columns.numeric or []):
        sdtype[col] = {"sdtype": "numerical"}
    for col in (columns.binary or []) + (columns.categorical or []):
        sdtype[col] = {"sdtype": "categorical"}
    return {"columns": sdtype}


def dcr(train: pd.DataFrame,
        test: pd.DataFrame,
        synthetic: pd.DataFrame,
        columns: Columns) -> dict[str, float]:
    """Distance-to-Closest-Record privacy via SDMetrics (established metrics).

    Two published DCR metrics from SDV, both scored in [0, 1]
    with higher = safer:
      - DCROverfittingProtection: is the synthetic data closer to the training
        set than to a real holdout (`test`)? Low scores flag memorisation of
        the training records.
      - DCRBaselineProtection: how close is the synthetic data to the real data
        versus a random-data baseline?
    """
    metadata = _sdv_metadata(columns)
    overfitting = DCROverfittingProtection.compute(
        real_training_data=train,
        synthetic_data=synthetic,
        real_validation_data=test,
        metadata=metadata,
    )
    baseline = DCRBaselineProtection.compute(
        real_data=train,
        synthetic_data=synthetic,
        metadata=metadata,
    )
    return {
        "dcr_overfitting_protection": float(overfitting),
        "dcr_baseline_protection": float(baseline),
    }

# ---------------------------------------------------------------------------
# Privacy: Anonymeter adversarial attacks
# ---------------------------------------------------------------------------
def adversarial_attacks(train: pd.DataFrame,
                        test: pd.DataFrame,
                        synthetic: pd.DataFrame,
                        n_attacks: int = 500,
                        seed: int = 0) -> dict[str, float]:
    """Anonymeter's three attacks, using the real test set as the control.

    - Singling out
    - Linkability
    - Inference
    Each risk is in [0, 1] with a 95% confidence interval
    """
    import time

    from anonymeter.evaluators import InferenceEvaluator, LinkabilityEvaluator, SinglingOutEvaluator

    results: dict[str, float] = {}

    def _progress(label: str, start: float) -> None:
        print(f"    [{label}] done in {time.time() - start:.1f}s", flush=True)

    def _risk_with_reliability(ev) -> tuple:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            risk = ev.risk()
        reliable = not any("cannot be trusted" in str(w.message) for w in caught)
        return risk, reliable

    # --- Singling out (univariate + multivariate) ---
    for mode in ("univariate", "multivariate"):
        print(f"  [singling_out:{mode}] starting (n_attacks={n_attacks})...", flush=True)
        start = time.time()
        try:
            ev = SinglingOutEvaluator(ori=train, syn=synthetic, control=test,
                                      n_attacks=n_attacks)
            ev.evaluate(mode=mode)
            risk, reliable = _risk_with_reliability(ev)
            results[f"singling_out_{mode}_risk"] = float(risk.value)
            results[f"singling_out_{mode}_ci_low"] = float(risk.ci[0])
            results[f"singling_out_{mode}_ci_high"] = float(risk.ci[1])
            results[f"singling_out_{mode}_reliable"] = reliable
            _progress(f"singling_out:{mode}", start)
        except Exception as e:  # one attack failing must not abort
            warnings.warn(f"singling_out ({mode}) failed: {e}")
            results[f"singling_out_{mode}_risk"] = float("nan")
            results[f"singling_out_{mode}_reliable"] = False

    # --- Linkability across two quasi-identifier partitions ---
    start = time.time()
    half = len(QUASI_IDENTIFIERS) // 2
    aux_cols = (QUASI_IDENTIFIERS[:half], QUASI_IDENTIFIERS[half:])
    try:
        print("  [linkability] starting...", flush=True)
        ev = LinkabilityEvaluator(ori=train, syn=synthetic, control=test,
                                  n_attacks=n_attacks, aux_cols=aux_cols, n_neighbors=10)
        ev.evaluate(n_jobs=-1)
        risk, reliable = _risk_with_reliability(ev)
        results["linkability_risk"] = float(risk.value)
        results["linkability_ci_low"] = float(risk.ci[0])
        results["linkability_ci_high"] = float(risk.ci[1])
        _progress("linkability", start)
    except Exception as e:
        warnings.warn(f"linkability failed: {e}")
        results["linkability_risk"] = float("nan")
        results["linkability_reliable"] = False

    # --- Inference on each sensitive attribute ---
    inference_risks = []
    for secret in SECRET_COLUMNS:
        if secret not in train.columns:
            continue
        print(f"    [inference:{secret}] starting...", flush=True)
        start = time.time()
        aux = [c for c in QUASI_IDENTIFIERS if c != secret]
        regression = secret in {"avg_glucose_level", "bmi", "age"}
        try:
            ev = InferenceEvaluator(ori=train, syn=synthetic, control=test,
                                    aux_cols=aux, secret=secret, regression=regression,
                                    n_attacks=n_attacks)
            ev.evaluate(n_jobs=-1)
            risk, reliable = _risk_with_reliability(ev)
            value = float(risk.value)
            results[f"inference_{secret}_risk"] = value
            results[f"inference_{secret}_reliable"] = reliable
            inference_risks.append(value)
            _progress(f"inference:{secret}", start)
        except Exception as e:
            warnings.warn(f"inference ({secret}) failed: {e}")
            results[f"inference_{secret}_risk"] = float("nan")
            results[f"inference_{secret}_reliable"] = False

    results["inference_risk_mean"] = float(np.nanmean(inference_risks)) if inference_risks else float("nan")
    results["inference_risk_max"] = float(np.nanmax(inference_risks)) if inference_risks else float("nan")

    unreliable = [k[:-len("_reliable")] for k, v in results.items()
                 if k.endswith("_reliable") and v is False]
    if unreliable:
        print(f"  UNRELIABLE (attack ~= baseline, per anonymeter): {unreliable}", flush=True)

    return results


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _distribution_figures(real: pd.DataFrame, synthetic: pd.DataFrame,
                          columns: Columns, name: str, figures_dir: Path) -> None:
    numeric = list(columns.numeric or [])
    if not numeric:
        return
    fig, axes = plt.subplots(1, len(numeric), figsize=(5 * len(numeric), 4))
    axes = np.atleast_1d(axes)
    for ax, col in zip(axes, numeric):
        sns.kdeplot(real[col].dropna(), ax=ax, label="real", fill=True, alpha=0.3)
        sns.kdeplot(synthetic[col].dropna(), ax=ax, label="synthetic", fill=True, alpha=0.3)
        ax.set_title(col)
        ax.legend()
    fig.suptitle(f"Real vs synthetic distributions - {name}")
    fig.tight_layout()
    fig.savefig(figures_dir / f"{name}_distributions.png", dpi=120)
    plt.close(fig)


def _association_heatmaps(real_assoc: pd.DataFrame, synth_assoc: pd.DataFrame,
                          name: str, figures_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    sns.heatmap(real_assoc, ax=axes[0], vmin=0, vmax=1, cmap="viridis", square=True)
    axes[0].set_title("real associations")
    sns.heatmap(synth_assoc, ax=axes[1], vmin=0, vmax=1, cmap="viridis", square=True)
    axes[1].set_title("synthetic associations")
    sns.heatmap((synth_assoc - real_assoc).abs(), ax=axes[2], vmin=0, vmax=1, cmap="rocket", square=True)
    axes[2].set_title("absolute difference")
    fig.suptitle(f"Association structure - {name}")
    fig.tight_layout()
    fig.savefig(figures_dir / f"{name}_associations.png", dpi=120)
    plt.close(fig)

def evaluate_fidelity(real: pd.DataFrame,
                      synthetic: pd.DataFrame,
                      columns: Columns,
                      name: str,
                      figures_dir: Path,
                      seed: int = 0) -> dict:
    print(f"Evaluating fidelity for '{name}'...")

    # Structural enforcement check
    validate_column_config(real, columns)
    validate_column_config(synthetic, columns)

    univariate_numeric, univariate_categorical = univariate_fidelity(real, synthetic, columns)
    real_assoc, synth_assoc, named = multivariate_fidelity(real, synthetic, columns)
    missingness = missingness_comparison(real, synthetic, columns)
    plausibility = plausibility_checks(synthetic)
    detection = detection_auc(real, synthetic, columns, seed)

    print(univariate_numeric)
    print(univariate_categorical)
    print("named clinical checks:\n", named)

    figures_dir.mkdir(parents=True, exist_ok=True)
    _distribution_figures(real, synthetic, columns, name, figures_dir)
    _association_heatmaps(real_assoc, synth_assoc, name, figures_dir)

    print("Fidelity evaluation complete.")

    return {
        "generator": name,
        "mean_ks_distance": univariate_numeric["ks_distance"].mean(),
        "mean_tvd": univariate_categorical["tvd"].mean(),
        "mean_hellinger_distance": pd.concat([
            univariate_numeric["hellinger_distance"], univariate_categorical["hellinger_distance"]
        ]).mean(),
        "mean_association_abs_diff": _association_matrix_mae(real_assoc, synth_assoc),
        "named_max_abs_diff": float(named["abs_diff"].max()),
        "missingness_mae": float(missingness["abs_diff"].mean()),
        "detection_auc": detection,
        "n_implausible": int(sum(plausibility.values())),
    }

def evaluate_utility(train: pd.DataFrame,
                     test: pd.DataFrame,
                     synthetic: pd.DataFrame,
                     columns: Columns,
                     name: str,
                     target: str = "stroke",
                     seed: int = 0) -> dict:
    print(f"Evaluating utility for '{name}'...")
    result = utility_tstr_trtr(train, test, synthetic, columns, target=target, seed=seed)
    print(f"  ROC-AUC  TRTR={result['roc_auc_trtr']:.3f}"
          f"  TSTR={result['roc_auc_tstr']:.3f}  gap={result['roc_auc_gap']:.3f}")
    print(f"  PR-AUC   TRTR={result['pr_auc_trtr']:.3f}"
          f"  TSTR={result['pr_auc_tstr']:.3f}  gap={result['pr_auc_gap']:.3f}"
          f"  (baseline={result['pr_auc_baseline']:.3f})")
    print(f"  F1   TRTR={result['f1_trtr']:.3f}  TSTR={result['f1_tstr']:.3f}  gap={result['f1_gap']:.3f}")
    return result

def evaluate_privacy(train: pd.DataFrame,
                     test: pd.DataFrame,
                     synthetic: pd.DataFrame,
                     columns: Columns,
                     name: str,
                     n_attacks: int = 500,
                     seed: int = 0,
                     skip_attacks: bool = False) -> dict:
    print(f"Evaluating privacy for '{name}'...")
    result = dcr(train, test, synthetic, columns)
    print(f"  DCR overfitting_protection={result['dcr_overfitting_protection']:.3f}"
          f"  baseline_protection={result['dcr_baseline_protection']:.3f}")
    if skip_attacks:
        return result
    attacks = adversarial_attacks(train, test, synthetic, n_attacks=n_attacks, seed=seed)
    result.update(attacks)
    print(f"  singling-out(multi)={attacks.get('singling_out_multivariate_risk')}"
          f"  linkability={attacks.get('linkability_risk')}"
          f"  inference(mean)={attacks.get('inference_risk_mean')}")
    return result

def _append_results(results_path: Path, row: dict) -> None:
    """Accumulate one row per generator in outputs/results.csv."""
    results_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([row])
    if results_path.exists():
        existing = pd.read_csv(results_path)
        existing = existing[existing["generator"] != row["generator"]]  # replace on re-run
        frame = pd.concat([existing, frame], ignore_index=True)
    frame.to_csv(results_path, index=False)


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=Path, required=False)
    parser.add_argument("--synthetic-path", type=Path, required=True)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--n-attacks", type=int, default=200)
    parser.add_argument("--skip-privacy", action="store_true",
                        help="skip privacy evaluation entirely (DCR and Anonymeter attacks)")
    parser.add_argument("--skip-attacks", action="store_true",
                        help="run DCR only, skip the slower Anonymeter attacks")
                        
    return parser.parse_args()


def main() -> None:
    args = _parse_arguments()

    if args.config_path:
        config = load_config(args.config_path)
    else:
        config = load_config()

    seed = config.seed

    real_path = config.paths.processed_dir / "train.csv"
    real_data = load_csv(real_path)

    synthetic_path = args.synthetic_path
    synthetic_data = load_csv(synthetic_path)

    real_test_path = config.paths.processed_dir / "test.csv"
    real_test_data = load_csv(real_test_path)

    figures_dir = config.paths.outputs_dir / "figures"

    target = config.split.stratify_col

    fidelity = evaluate_fidelity(real_data, synthetic_data, config.columns, args.name, figures_dir, seed=seed)
    utility = evaluate_utility(real_data, real_test_data, synthetic_data, config.columns, args.name,
                               target=target, seed=seed)
    privacy = {}
    if not args.skip_privacy:
        privacy = evaluate_privacy(real_data, real_test_data, synthetic_data, config.columns,
                                   args.name, n_attacks=args.n_attacks,
                                   skip_attacks=args.skip_attacks, seed=seed)

    row = {**fidelity, **utility, **privacy}
    results_path = config.paths.outputs_dir / "results.csv"
    _append_results(results_path, row)
    print(f"\nResults written to {results_path}")
    print(pd.DataFrame([row]).T)


if __name__ == "__main__":
    main()
