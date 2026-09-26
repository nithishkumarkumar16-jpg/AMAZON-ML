"""Data loading module for Amazon ML Challenge 2026.

Supports efficient loading and chunked streaming of multi-million-row TSVs
using pandas with sep='\t', proper dtypes, and clean missing value handling.
"""

import os
from pathlib import Path
from typing import Generator, List, Optional, Union

import pandas as pd

try:
    from .config import (
        DEFAULT_CHUNK_SIZE,
        DELIMITER,
        ENCODING,
        GROUND_TRUTH_COLUMNS,
        SOURCE_COLUMNS,
    )
except ImportError:
    from config import (
        DEFAULT_CHUNK_SIZE,
        DELIMITER,
        ENCODING,
        GROUND_TRUTH_COLUMNS,
        SOURCE_COLUMNS,
    )


def load_source_tsv(
    file_path: Union[str, Path],
    columns: Optional[List[str]] = None,
    nrows: Optional[int] = None,
    dtype: Optional[dict] = None,
) -> pd.DataFrame:
    """Load an entire source TSV into memory with optimized string dtypes.

    Parameters
    ----------
    file_path : str or Path
        Path to the TSV file.
    columns : list of str, optional
        Subset of columns to read. Defaults to all SOURCE_COLUMNS.
    nrows : int, optional
        Number of rows to read (useful for testing and EDA).
    dtype : dict, optional
        Column type mapping. By default, forces string type for all fields.

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame with empty strings for missing values (no NaN objects).
    """
    if columns is None:
        columns = SOURCE_COLUMNS

    if dtype is None:
        dtype = {col: "string" for col in columns}

    df = pd.read_csv(
        file_path,
        sep=DELIMITER,
        usecols=columns,
        nrows=nrows,
        dtype=dtype,
        encoding=ENCODING,
        keep_default_na=False,  # Treat empty fields as empty strings, avoid NaN
    )
    return df


def stream_source_tsv(
    file_path: Union[str, Path],
    columns: Optional[List[str]] = None,
    chunksize: int = DEFAULT_CHUNK_SIZE,
    country_filter: Optional[str] = None,
) -> Generator[pd.DataFrame, None, None]:
    """Stream a multi-million-row source TSV in memory-safe chunks.

    Parameters
    ----------
    file_path : str or Path
        Path to the TSV file.
    columns : list of str, optional
        Subset of columns to read. Defaults to all SOURCE_COLUMNS.
    chunksize : int
        Number of rows per chunk (default 100,000).
    country_filter : str, optional
        If provided, yields only rows matching the specified country label (open set).

    Yields
    ------
    pd.DataFrame
        Consecutive chunks of the source file.
    """
    if columns is None:
        columns = SOURCE_COLUMNS

    dtype = {col: "string" for col in columns}

    reader = pd.read_csv(
        file_path,
        sep=DELIMITER,
        usecols=columns,
        chunksize=chunksize,
        dtype=dtype,
        encoding=ENCODING,
        keep_default_na=False,
    )

    for chunk in reader:
        if country_filter:
            chunk = chunk[chunk["country"] == country_filter]
            if chunk.empty:
                continue
        yield chunk


def load_ground_truth(
    file_path: Union[str, Path],
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """Load the ground truth match pairs into memory.

    Parameters
    ----------
    file_path : str or Path
        Path to train_ground_truth.tsv.
    nrows : int, optional
        Number of rows to read.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['source1_entity_id', 'matched_entity_ids'].
    """
    dtype = {col: "string" for col in GROUND_TRUTH_COLUMNS}
    df = pd.read_csv(
        file_path,
        sep=DELIMITER,
        usecols=GROUND_TRUTH_COLUMNS,
        nrows=nrows,
        dtype=dtype,
        encoding=ENCODING,
        keep_default_na=False,
    )
    return df
