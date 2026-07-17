"""prepare_dataset.py. Loads the raw dataset, applies the EDA decisions
(drop id, drop the single Other gender row), and writes stratified
train/test/val splits to data/processed/."""

import argparse
from pathlib import Path

import pandas as pd

from src.config import Split, load_config
from src.io_utils import load_csv, write_csv


def apply_decisions(data: pd.DataFrame) -> pd.DataFrame:
    """ Apply the data decisions surfaced from the EDA.

    1.) Drop the 'id' field from the data
    2.) Drop the single 'Other' sample entry from the 'gender' field.
    3.) Keep missingness in 'bmi' intact. No processing step is needed for this. 'N/A' is handled natively by pandas.

    Args:
        data (pd.DataFrame): The dataset in pandas DataFrame format 

    Returns:
        pd.DataFrame: The output dataset in pandas DataFrame format 
    """
    print("Applying data processing decisions...")

    # 1.) Drop the 'id' field from the data
    data = data.drop(['id'], axis=1)

    # 2.) Drop the single 'Other' sample entry from the 'gender' field.
    other_index = data[data['gender'] == 'Other'].index
    data = data.drop(other_index)
    data = data.reset_index(drop=True)

    print("Decisions applied.")

    return data

def split(data: pd.DataFrame,
          split_config: Split,
          seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """ Stratifies and split_config the dataset into Train, Test and optionally Validation sets.

    Args:
        data (pd.DataFrame):        The dataset in pandas DataFrame format 
        split_config (Split):       The split configuration 
        seed (int):                 The random seed to use for the stratified splits

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple containing the training, test and val splits.
            If the validation set size is zero, the third element will be None.
    """
    print("Creating stratified dataset splits...")
    print("Split configuration:")
    print(f" - Random seed: {seed}")
    print(f" - Training set ratio: {split_config.train_size:.2f}")
    print(f" - Test set ratio: {split_config.test_size:.2f}")
    print(f" - Validation set ratio: {split_config.val_size if split_config.val_size is not None else 0:.2f}")
    print(f" - Stratification on column: {split_config.stratify_col}")

    grouped = data.groupby(split_config.stratify_col, group_keys=False)

    train_slices, test_slices, val_slices = [], [], []

    for group_key, group in grouped:
        shuffled = group.sample(frac=1.0, random_state=seed)
        n = len(shuffled)

        train_end = int(n * split_config.train_size)
        train_slice = shuffled.iloc[:train_end]

        if split_config.val_size is None:
            test_slice = shuffled.iloc[train_end:]
            val_slice = None
        else:
            test_end = train_end + int(n * split_config.test_size)
            test_slice = shuffled.iloc[train_end:test_end]
            val_slice = shuffled.iloc[test_end:]

        if len(train_slice) == 0:
            raise ValueError(
                f"Stratified group '{group_key}' (n={n}) produced 0 training "
                f"rows at train_size={split_config.train_size:.2f}. Increase "
                f"the group's sample size or the train ratio."
            )
        if len(test_slice) == 0:
            raise ValueError(
                f"Stratified group '{group_key}' (n={n}) produced 0 test "
                f"rows at test_size={split_config.test_size:.2f}. Increase "
                f"the group's sample size or the test ratio."
            )
        if val_slice is not None and len(val_slice) == 0:
            raise ValueError(
                f"Stratified group '{group_key}' (n={n}) produced 0 "
                f"validation rows at val_size={split_config.val_size:.2f}. "
                f"Increase the group's sample size or the val ratio."
            )

        train_slices.append(train_slice)
        test_slices.append(test_slice)
        if val_slice is not None:
            val_slices.append(val_slice)

    train_df = pd.concat(train_slices).sample(frac=1.0, random_state=seed)
    test_df = pd.concat(test_slices).sample(frac=1.0, random_state=seed)
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    if split_config.val_size is None:
        val_df = None
    else:
        val_df = pd.concat(val_slices).sample(frac=1.0, random_state=seed)
        val_df = val_df.reset_index(drop=True)

    # Runtime check for stratification accuracy validity
    overall_rate = data[split_config.stratify_col].mean()
    train_rate = train_df[split_config.stratify_col].mean()
    test_rate = test_df[split_config.stratify_col].mean()

    tolerance = 0.005
    if abs(train_rate - overall_rate) >= tolerance:
        raise ValueError(
            f"Train stratification deviates from overall rate: "
            f"{train_rate:.4f} vs {overall_rate:.4f} (tolerance {tolerance})"
        )

    if abs(test_rate - overall_rate) >= tolerance:
        raise ValueError(
            f"Test stratification deviates from overall rate: "
            f"{test_rate:.4f} vs {overall_rate:.4f} (tolerance {tolerance})"
        )
        
    if val_df is not None:
        val_rate = val_df[split_config.stratify_col].mean()

        if abs(val_rate - overall_rate) >= tolerance:
            raise ValueError(
                f"Val stratification deviates from overall rate: "
                f"{val_rate:.4f} vs {overall_rate:.4f} (tolerance {tolerance})"
            )

    print("Training set size:", len(train_df))
    print(f"Train stratification rate: {train_rate:.4f} vs overall {overall_rate:.4f}")

    print("Test set size:", len(test_df))
    print(f"Test stratification rate: {test_rate:.4f} vs overall {overall_rate:.4f}")

    if val_df is not None:
        print("Validation set size:", len(val_df))
        print(f"Val stratification rate: {val_rate:.4f} vs overall {overall_rate:.4f}")
    print("Splitting complete.")

    return train_df, test_df, val_df


def write_outputs(dst_dir: Path, train: pd.DataFrame, test: pd.DataFrame, val: pd.DataFrame | None) -> None:
    """ Write splits to disk.

    Args:
        dst_dir (Path):       Destination directory path.
        train (pd.DataFrame):       training set pandas DataFrame 
        test (pd.DataFrame):        test set pandas DataFrame 
        val (pd.DataFrame | None):  (Optional) validation set pandas DataFrame

    Raises:
        ValueError:                  If a target file already exists.
        OSError:                     If the directory cannot be created or files cannot be written.
        PermissionError:             If the process lacks permissions to write to the destination.
        Exception:                   For any other unexpected errors during the write process.
    """
    print("Writing dataset splits to disk...")

    train_path = dst_dir / 'train.csv'
    test_path = dst_dir / 'test.csv'
    val_path = dst_dir / 'val.csv'

    # Prevent silent overwrites
    if train_path.is_file():
        raise ValueError(f"Training csv file {train_path} already exists. "
                         "Please delete or rename it before proceeding.")
    if test_path.is_file():
        raise ValueError(f"Test csv file {test_path} already exists. "
                         "Please delete or rename it before proceeding.")
    if val is not None and val_path.is_file():
        raise ValueError(f"Validation csv file {val_path} already exists. "
                         "Please delete or rename it before proceeding.")

    write_csv(train_path, train)
    write_csv(test_path, test)
    if val is not None:
        write_csv(val_path, val)

    print("Splits written to disk successfully.")

def _parse_arguments() -> argparse.Namespace:
    """ Parse command line arguments

    Returns:
        argparse.Namespace: The parsed arguments 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=Path, required=False)
    return parser.parse_args()

def main() -> None:
    args = _parse_arguments()

    print("Preparing dataset...")

    if args.config_path:
        config = load_config(args.config_path)
    else:
        config = load_config()

    data = load_csv(config.paths.raw)
    data = apply_decisions(data)
    train, test, val = split(data, config.split, config.seed)
    write_outputs(config.paths.processed_dir, train, test, val)

    print("Dataset preparation completed successfully.")

if __name__ == "__main__":
    main()