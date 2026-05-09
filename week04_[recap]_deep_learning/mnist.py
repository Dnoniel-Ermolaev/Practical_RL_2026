"""Load MNIST. Adapted: yann.lecun.com is offline since 2024, so we fetch from a
working CVDF mirror. Returns (X_train, y_train, X_val, y_val, X_test, y_test)
with shapes (N, 1, 28, 28) float32 in [0, 1] and label vectors of int64."""
import os
import gzip
import numpy as np
from urllib.request import urlretrieve

MIRRORS = [
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "http://yann.lecun.com/exdb/mnist/",
]


def _download(filename: str) -> None:
    last_err = None
    for src in MIRRORS:
        try:
            print(f"Downloading {filename} from {src}")
            urlretrieve(src + filename, filename)
            return
        except Exception as e:  # noqa: BLE001
            last_err = e
            continue
    raise RuntimeError(f"Could not download {filename} from any mirror; last error: {last_err}")


def _load_images(filename: str) -> np.ndarray:
    if not os.path.exists(filename):
        _download(filename)
    with gzip.open(filename, "rb") as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, 1, 28, 28)
    return data.astype(np.float32) / np.float32(256)


def _load_labels(filename: str) -> np.ndarray:
    if not os.path.exists(filename):
        _download(filename)
    with gzip.open(filename, "rb") as f:
        return np.frombuffer(f.read(), np.uint8, offset=8).astype(np.int64)


def load_dataset():
    X_train = _load_images("train-images-idx3-ubyte.gz")
    y_train = _load_labels("train-labels-idx1-ubyte.gz")
    X_test = _load_images("t10k-images-idx3-ubyte.gz")
    y_test = _load_labels("t10k-labels-idx1-ubyte.gz")

    # last 10k of training set used as validation
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]
    return X_train, y_train, X_val, y_val, X_test, y_test
