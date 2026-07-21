import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from snsynth import Synthesizer

from src.column_utils import validate_column_config
from src.config import Columns, load_config
from src.io_utils import load_csv, write_csv

DEFAULT_EPSILONS = [0.5, 1.0, 3.0, 8.0]

# Public clinical ranges (NOT learned from the data) and bin count.
NUMERIC_BOUNDS = {"age": (0.0, 100.0), "avg_glucose_level": (40.0, 300.0), "bmi": (10.0, 100.0)}
BINS = 10
REPORTING_LENGTH = 3  # highest-order marginal PAC-Synth preserves

def _parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        argparse.Namespace: The parsed arguments. 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=Path, required=False)
    parser.add_argument("--epsilons", type=float, nargs="+", default=DEFAULT_EPSILONS)
    return parser.parse_args()


def generate_dp(data: pd.DataFrame,
                columns: Columns,
                epsilon: float,
                seed: int) -> pd.DataFrame:
    """Generate synthetic data using Differential Privacy (AIM mechanism).

    Args:
        data (pd.DataFrame):      Real dataset.
        columns (Columns):        Column-type groupings from configuration.
        epsilon (float):          Total privacy budget allocated for synthesis.
        seed (int):               Random seed for reproducibility across frameworks.

    Returns:
        pd.DataFrame: Synthetic dataset matching input schema.

    Raises:
        ValueError: If epsilon is non positive.
    """
    if epsilon <= 0:
        raise ValueError(f"epsilon must be strictly positive, got {epsilon}.")
    validate_column_config(data, columns)

    # Apply random seed for reproducibility
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    numeric = list(columns.numeric or [])

    # 1. Discretise continuous columns to bin indices. Missing -> "NA".
    binned = data.copy()
    for col in numeric:
        low, high = NUMERIC_BOUNDS[col]
        binned[col] = pd.cut(data[col], np.linspace(low, high, BINS + 1), labels=False, include_lowest=True)
    binned = binned.astype(object).where(binned.notna(), "NA").astype(str)

    # 2. Fit the DP marginal synthesizer and sample.
    synth = Synthesizer.create("pacsynth", epsilon=epsilon, reporting_length=REPORTING_LENGTH)
    synth.fit(binned, categorical_columns=list(binned.columns), preprocessor_eps=0.0)
    sample = synth.sample(len(data))

    # 3. Invert the binning: bin index -> midpoint. Any gap -> NaN.
    out = sample.replace({"NA": np.nan, "": np.nan})
    for col in numeric:
        low, high = NUMERIC_BOUNDS[col]
        edges = np.linspace(low, high, BINS + 1)
        midpoints = (edges[:-1] + edges[1:]) / 2
        codes = pd.to_numeric(sample[col], errors="coerce")
        out[col] = codes.map(lambda c: midpoints[int(c)] if pd.notna(c) else np.nan)

    # 4. Only missing columns from the real data keep NaN. Impute the rest so the
    #    target and categoricals stay complete, then restore original dtypes.
    nullable = {c for c in data.columns if data[c].isna().any()}
    for col in data.columns:
        if col not in nullable:
            fill = data[col].median() if col in numeric else data[col].mode().iloc[0]
            out[col] = out[col].fillna(fill)
        try:
            out[col] = out[col].astype(data[col].dtype)
        except (ValueError, TypeError):
            pass
    return out[data.columns]


def main() -> None:
    args = _parse_arguments()

    print("Generating Differential Privacy guided synthetic datasets...")

    if args.config_path:
        config = load_config(args.config_path)
    else:
        config = load_config()

    train_path = config.paths.processed_dir / "train.csv"
    data = load_csv(train_path)

    for epsilon in args.epsilons:
        print(f"\nGenerating DP synthetic data at epsilon={epsilon}...")

        sample = generate_dp(data, config.columns, epsilon, config.seed)

        output_path = config.paths.synthetic_dir / f"dp_eps{epsilon}.csv"
        write_csv(output_path, sample)
        print(f"    epsilon={epsilon} -> {output_path}")

    print("DP guided synthetic dataset generation completed successfully.")

if __name__ == "__main__":
    main()