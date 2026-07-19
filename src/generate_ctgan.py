import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sdv.metadata import Metadata
from sdv.single_table import CTGANSynthesizer

from src.column_utils import validate_column_config
from src.config import Columns, load_config
from src.io_utils import load_csv, write_csv

DEFAULT_EPOCHS = 300


def build_metadata(data: pd.DataFrame, columns: Columns) -> Metadata:
    """Build SDV metadata explicitly from the config's column groupings.

    Column types are declared from config rather than SDV's own
    auto-detection, so the generator cannot diverge from the configuration
    set within this pipeline.

    Args:
        data (pd.DataFrame): The real data
        columns (Columns):   The column-type groupings from config.

    Returns:
        Metadata: SDV metadata with every column's sdtype
    """
    metadata = Metadata.detect_from_dataframe(data, table_name="table")

    for col in columns.numeric or []:
        metadata.update_column(col, sdtype="numerical", table_name="table")
    for col in (columns.binary or []) + (columns.categorical or []):
        metadata.update_column(col, sdtype="categorical", table_name="table")

    return metadata 


def generate(data: pd.DataFrame,
             columns: Columns,
             seed: int,
             epochs: int) -> tuple[pd.DataFrame, pd.DataFrame, CTGANSynthesizer]:
    """Generate the CTGAN (L3 - multivariate) synthetic dataset.

    CTGAN is used with SDV's library defaults, deliberately untuned, aside from the
    epoch count.

    Args:
        data (pd.DataFrame): Real data.
        columns (Columns):   The column-type groupings from config. 
        seed (int):          Random seed for reproducibility.
        epochs (int):        The number of epochs used for training.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, CTGANSynthesizer]: The synthetic dataset,
            the per-epoch loss history, and the fitted CTGAN generator.
    """

    print(f"Generating CTGAN (L3 - multivariate) synthetic data ({epochs} epochs)...")
    validate_column_config(data, columns)

    # Apply random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    metadata = build_metadata(data, columns)
    generator = CTGANSynthesizer(metadata, epochs=epochs)
    generator.fit(data)

    synthetic = generator.sample(num_rows=len(data))
    synthetic = synthetic[data.columns] # preserve original column order

    losses = generator.get_loss_values()

    return synthetic, losses, generator


def _parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        argparse.Namespace: The parsed arguments. 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=Path, required=False)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    return parser.parse_args()


def main() -> None:
    args = _parse_arguments()

    print("Generating high fidelity synthetic dataset with CTGAN...")

    if args.config_path:
        config = load_config(args.config_path)
    else:
        config = load_config()

    train_path = config.paths.processed_dir / "train.csv"
    data = load_csv(train_path)

    synthetic, losses, generator = generate(data, config.columns, config.seed, args.epochs)
    
    output_path = config.paths.synthetic_dir / "ctgan.csv"
    write_csv(output_path, synthetic)

    losses_path = config.paths.outputs_dir / "ctgan_loss.csv"
    write_csv(losses_path, losses)

    model_path = config.paths.outputs_dir / "ctgan_model.pkl"
    generator.save(str(model_path))

    metadata_path = config.paths.outputs_dir / "ctgan_metadata.json"
    generator.get_metadata().save_to_json(str(metadata_path), mode="overwrite")

    print("CTGAN synthetic data generation completed successfully.")


if __name__ == "__main__":
    main()
