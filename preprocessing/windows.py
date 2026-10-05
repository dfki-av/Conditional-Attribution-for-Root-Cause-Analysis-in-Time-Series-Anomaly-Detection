"""Windowing operations used by the experiment."""
import numpy as np
import pandas as pd
import torch


def make_windows(
    data: pd.DataFrame,
    window_size: int,
    stride: int = 50
) -> np.ndarray:
    """
    Slices data into sliding windows with a custom stride.

    If n_rows < window_size, pads the sequence by repeating the last row
    until it reaches window_size, and returns one window.

    Parameters
    ----------
    data : pd.DataFrame or torch.Tensor
        Equal sampled signal data; each column is a signal.
    window_size : int
        Number of time steps per sequence.
    stride : int, default=50
        Number of rows to move forward between sequences.

    Returns
    -------
    np.ndarray
        Array of shape (n_sequences, window_size, n_features)
    """
    if torch.is_tensor(data):
        data_np = data.detach().cpu().numpy().copy()
    else:
        data_np = data.values.copy()

    n_rows = len(data_np)

    if n_rows == 0:
        raise ValueError("Input data has 0 rows, cannot create sequences.")

    # Case 1: sequence shorter than window_size -> pad with last row
    if n_rows < window_size:
        pad_count = window_size - n_rows
        last_row = data_np[-1:, :]                 # shape (1, n_features)
        pad_block = np.repeat(last_row, pad_count, axis=0)
        padded = np.concatenate([data_np, pad_block], axis=0)
        return padded[None, :, :]                  # shape (1, window_size, n_features)

    # Case 2: normal sliding windows
    output = []
    for i in range(0, n_rows - window_size + 1, stride):
        output.append(data_np[i:i + window_size, :])

    return np.stack(output)
