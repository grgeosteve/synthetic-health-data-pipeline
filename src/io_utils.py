"""io_utils.py. Shared CSV loading and writing logic used across the pipeline (raw
data, processed splits, and synthetic outputs)."""

from pathlib import Path

import pandas as pd


def load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV file into a DataFrame.

    Args:
        path (Path): Path to the CSV file.

    Raises:
        FileNotFoundError:          If the file does not exist at the provided path.
        pd.errors.EmptyDataError:   If the CSV file is empty.
        pd.errors.ParserError:      If the CSV is malformed and cannot be parsed.
        Exception:                  For any other unexpected errors during loading.

    Returns:
        pd.DataFrame: The loaded dataset.
    """
    print(f"Loading data from {path}...")

    try:
        data = pd.read_csv(path)

        print("Loading completed successfully.")
        return data

    except FileNotFoundError as e:
        raise FileNotFoundError(f"The data file was not found at: {path}") from e
    except pd.errors.EmptyDataError as e:
        raise pd.errors.EmptyDataError(f"The CSV file at {path} is empty.") from e
    except pd.errors.ParserError as e:
        raise pd.errors.ParserError(f"Failed to parse the CSV file at {path}. Please check the file format.") from e
    except Exception as e:
        raise Exception(f"An unexpected error occurred while loading {path}") from e


def write_csv(path:Path, data: pd.DataFrame) -> None:
    """Write a pandas DataFrame to a CSV file.

    Creates the parent directory if needed. Guards againt silent overwrite of an existing file.

    Args:
        path (Path):         Destination file path.
        data (pd.DataFrame): The pandas DataFrame to write.

    Raises:
        ValueError:      If the target file already exists.
        PermissionError: If the process lacks permissions to write to the destination.
        OSError:         If the directory cannot be created or files cannot be written.
        Exception:       For any other unexpected errors during the write process.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.is_file():
            raise ValueError(
                f"File {path} already exists. Please delete or rename it before proceeding."
            )

        data.to_csv(path, index=False)

    except PermissionError as e:
        raise PermissionError(f"Permission denied: Unable to write data to {path}") from e
    except OSError as e:
        raise OSError(f"OS error occurred while writing data to {path}") from e
    except ValueError:
        raise
    except Exception as e:
        raise Exception(f"An unexpected error occurred while writing data to {path}") from e
