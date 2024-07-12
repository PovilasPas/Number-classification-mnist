import numpy as np


def read_mnist(image_file, label_file):
    with open(image_file, "rb") as fd:
        fd.read(16)
        buffer = fd.read()
        X = np.frombuffer(buffer, dtype=np.uint8)
        X = X.reshape((-1, 28 * 28))
        X = X

    with open(label_file, "rb") as fd:
        fd.read(8)
        buffer = fd.read()
        y = np.frombuffer(buffer, dtype=np.uint8)

    return X, y


def fixed_normalization(X, current_min, current_max, normalized_min, normalized_max):
    return (X - current_min)/(current_max - current_min) * (normalized_max - normalized_min) + normalized_min
