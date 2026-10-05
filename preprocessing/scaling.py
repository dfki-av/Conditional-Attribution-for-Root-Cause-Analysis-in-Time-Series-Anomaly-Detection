"""Deterministic splitting and feature-wise normalization."""
import numpy as np
from sklearn.preprocessing import StandardScaler


class MinMaxScaler:
    """Scale each feature using the training minimum and range."""

    def fit(self, data):
        self.mini = np.min(data, axis=0)
        self.range = np.max(data, axis=0) - self.mini
        return self

    def transform(self, data):
        return (data - self.mini) / (self.range + 1e-7)


def split_windows(
    data: np.ndarray, valid_perc: float, shuffle: bool = True, seed: int = 123
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split the data into training and validation sets.

    Args:
        data (np.ndarray): The dataset to split.
        valid_perc (float): The percentage of data to use for validation.
        shuffle (bool, optional): Whether to shuffle the data before splitting.
                                  Defaults to True.
        seed (int, optional): The random seed to use for shuffling the data.
                              Defaults to 123.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the training data and
                                       validation data arrays.
    """
    N = data.shape[0]
    N_train = int(N * (1 - valid_perc))

    if shuffle:
        np.random.seed(seed)
        data = data.copy()
        np.random.shuffle(data)

    train_data = data[:N_train]
    valid_data = data[N_train:]
    return train_data, valid_data


def scale_windows(train_data, valid_data, method="min_max"):
    """
    Apply the selected feature-wise normalization to 3D time-series data.

    Input shapes:
        train_data: (N_train, T, D)
        valid_data: (N_valid, T, D)

    Output shapes remain unchanged.
    """

    if method == "min_max":
        scaler = MinMaxScaler().fit(train_data)
        return scaler.transform(train_data), scaler.transform(valid_data), scaler
    if method != "standard":
        raise ValueError(f"Unsupported normalization method: {method}")
    scaler = StandardScaler()

    # Store original shapes
    train_shape = train_data.shape
    valid_shape = valid_data.shape

    # Convert from (N, T, D) to (N * T, D)
    train_data_2d = train_data.reshape(-1, train_shape[-1])
    valid_data_2d = valid_data.reshape(-1, valid_shape[-1])

    # Fit only on training data
    scaler.fit(train_data_2d)
    scaled_train_data = scaler.transform(train_data_2d)

    # Use training statistics for validation data
    scaled_valid_data = scaler.transform(valid_data_2d)

    # Restore original 3D shapes
    scaled_train_data = scaled_train_data.reshape(train_shape)
    scaled_valid_data = scaled_valid_data.reshape(valid_shape)

    return scaled_train_data, scaled_valid_data, scaler


def transform_windows(windows, scaler):
    """Apply a fitted feature scaler to windows without changing their shape."""
    if isinstance(scaler, MinMaxScaler):
        return scaler.transform(windows)
    shape = windows.shape
    return scaler.transform(windows.reshape(-1, shape[-1])).reshape(shape)
