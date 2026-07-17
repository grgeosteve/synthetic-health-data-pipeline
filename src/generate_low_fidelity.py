import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from src.column_utils import validate_column_config
from src.config import Columns, load_config
from src.io_utils import load_csv, write_csv


def synthesise_numeric_column(column: pd.Series, n: int, rng: np.random.Generator) -> pd.Series:
    """Synthesise one numeric column independently.

    Fits a kernel density estimate to the observed non-null values and draws new samples from it.
    Missingness is reproduced as an independent Bernoulli draw with the observed rate.
    If the original column was integer-typed, the synthetic values are rounded and cast 
    back to integer values to preserve type compatibility, since a KDE produces floating point
    values.

    Args:
        column (pd.Series):         The real column to fit.
        n (int):                    The number of samples to synthesise.
        rng (np.random.Generator):  Seeded random generator.

    Returns:
        pd.Series: The synthesised column. Same name and type as the input. 
    """
    missing_rate = column.isnull().mean()
    observed = column.dropna().to_numpy()
    is_integer = pd.api.types.is_integer_dtype(column)

    kde = gaussian_kde(observed)
    sampled_values = kde.resample(size=n, seed=rng)[0]

    if is_integer:
        sampled_values = np.round(sampled_values)

    is_missing = rng.random(n) < missing_rate
    result = np.where(is_missing, np.nan, sampled_values)

    result_series = pd.Series(result, name=column.name)
    if is_integer:
        result_series = result_series.astype("Int64" if missing_rate > 0 else "int64")

    return result_series


def synthesise_categorical_column(column: pd.Series, n: int, rng: np.random.Generator) -> pd.Series:
    """Synthesise a categorical or binary column independently.

    Samples from the column's observed frequency table. Missingness, if present,
    is treated as its own category and reproduced at its observed rate.

    Args:
        column (pd.Series):         The real column to fit.
        n (int):                    The number of samples to synthesise.
        rng (np.random.Generator):  Seeded random generator.

    Returns:
        pd.Series: The synthesised column. Same name and type as the input.
    """
    value_counts = column.value_counts(normalize=True, dropna=False)
    categories = value_counts.index.to_numpy()
    probabilities = value_counts.to_numpy()

    sampled_values = rng.choice(categories, size=n, p=probabilities)

    return pd.Series(sampled_values, name=column.name)

def generate(data: pd.DataFrame, columns: Columns, seed: int) -> pd.DataFrame:
    """Generate the low-fidelity (L2 - univariate) synthetic dataset.

    Every column is synthesised independently, so no inter-variable relationships
    are preserved. Row count matches the input data.

    Args:
        data (pd.DataFrame): Real data
        columns (Columns):   The column-type groupings from config.
        seed (int):          Random seed for reproducibility.

    Returns:
        pd.DataFrame: The generated synthetic dataset.
    """
    print("Generating low-fidelity (L2) synthetic data...")
    validate_column_config(data, columns)

    rng = np.random.default_rng(seed)
    n = len(data)

    synthetic_columns = {}

    for col in columns.numeric or []:
        synthetic_columns[col] = synthesise_numeric_column(data[col], n, rng)

    for col in (columns.binary or []) + (columns.categorical or []):
        synthetic_columns[col] = synthesise_categorical_column(data[col], n, rng)

    synthetic = pd.DataFrame(synthetic_columns)
    synthetic = synthetic[data.columns] # Preserve original column order

    print("Generation completed successfully.")
    return synthetic

def _parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        argparse.Namespace: The parsed arguments. 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=Path, required=False)
    return parser.parse_args()

def main() -> None:
    args = _parse_arguments()

    print("Generating low-fidelity synthetic dataset...")

    if args.config_path:
        config = load_config(args.config_path)
    else:
        config = load_config()

    train_path = config.paths.processed_dir / "train.csv"
    data = load_csv(train_path)

    synthetic = generate(data, config.columns, config.seed)

    output_path = config.paths.synthetic_dir / "low_fidelity.csv"
    write_csv(output_path, synthetic)

    print("Low-fidelity synthetic dataset generation completed successfully.")

if __name__ == "__main__":
    main()