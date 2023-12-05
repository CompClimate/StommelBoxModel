import numpy as np
from sklearn.preprocessing import MinMaxScaler


def sliding_windows(data, seq_length):
    """Transforms a 1d time series into sliding windows of length `seq_length`."""
    x = []
    y = []

    for i in range(len(data) - seq_length - 1):
        _x = data[i : (i + seq_length)]
        _y = data[i + seq_length]
        x.append(_x)
        y.append(_y)

    return np.array(x), np.array(y)


def prepare_data(train, test, seq_len, scale=False):
    """From a train/test split, optionally scales the data and returns sliding windows."""
    if scale:
        sc = MinMaxScaler()
        train = sc.fit_transform(train)
        test = sc.fit_transform(test)

    X_train, y_train = sliding_windows(train, seq_len)
    X_test, y_test = sliding_windows(test, seq_len)

    X_train, X_test = X_train.astype(np.float32), X_test.astype(np.float32)
    y_train, y_test = y_train.astype(np.float32), y_test.astype(np.float32)

    return (X_train, y_train), (X_test, y_test)


def get_raw_data(
    series_dict,
    autoregressive=False,
    window_size=None,
):
    if autoregressive:
        y = series_dict["q"]
        X, y = sliding_windows(y, window_size)
    else:
        series_features = list(series_dict["features"].values())
        features = series_features[0].copy()
        for d in series_features[1:]:
            features.update(d)

        X = np.hstack([value.reshape(-1, 1) for value in features.values()])
        y = series_dict["q"]

    X, y = X.astype(np.float32), y.astype(np.float32)
    return X, y
