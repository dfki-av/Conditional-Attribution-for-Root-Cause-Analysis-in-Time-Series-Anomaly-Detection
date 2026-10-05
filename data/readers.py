"""CSV readers for the SWaT experiment."""
import os
from typing import Iterable, Optional, Union

import pandas as pd


def read_csv(
    csv_path: str,
    usecols: Optional[Union[Iterable[int], Iterable[str]]] = None,
    return_filenames: bool = False,
    **kwargs
) -> pd.DataFrame:
    """
    Read a CSV file into a pandas DataFrame.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file.
    usecols : iterable of int or str, optional
        Columns to read (e.g. range(83) or ['col1', 'col2']).
    return_filenames : bool, optional
        Also return the CSV filename stem.
    **kwargs
        Additional keyword arguments passed to pandas.read_csv
        (e.g. sep, dtype, nrows, chunksize).

    Returns
    -------
    pd.DataFrame or tuple[pd.DataFrame, str]
        Loaded DataFrame, optionally with its filename stem.
    """
    df = pd.read_csv(csv_path, usecols=usecols, **kwargs)
    return (df, os.path.splitext(os.path.basename(csv_path))[0]) if return_filenames else df


def load_csv_directory(
    folder_path: str,
    usecols: Optional[Union[Iterable[int], Iterable[str]]] = None,
    return_filenames: bool = False,
    **kwargs
) -> dict:
    """
    Load CSV files into a dictionary keyed by the numeric prefix before '_'.
    
    Parameters
    ----------
    folder_path : str
        Path to the folder containing CSV files.
    usecols : iterable of int or str, optional
        Columns to read.
    return_filenames : bool, optional
        Also return a mapping from attack IDs to CSV filename stems.
    **kwargs
        Additional keyword arguments passed to read_csv.
    
    Returns
    -------
    dict or tuple[dict, dict]
        DataFrames keyed by numeric prefix, optionally with filename stems.
    """
    data_dict = {}
    filenames = {}

    for filename in os.listdir(folder_path):
        if not filename.endswith(".csv"):
            continue

        # Extract key before first underscore
        key = filename.split("_", 1)[0]

        file_path = os.path.join(folder_path, filename)

        # Reuse your CSV reader
        try:
            result = read_csv(
                file_path, usecols=usecols, return_filenames=return_filenames, **kwargs
            )
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

        if return_filenames:
            data_dict[key], filenames[key] = result
        else:
            data_dict[key] = result

    return (data_dict, filenames) if return_filenames else data_dict
