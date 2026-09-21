"""
Duomenų gavimo modulis (kolokviumo plano 2 punktas: "Duomenų gavimas").

Šaltinis: oficialus Fashion-MNIST idx-ubyte formatas iš
https://github.com/zalandoresearch/fashion-mnist (tas pats šaltinis,
kurio DOI/nuoroda naudota kolokviumo 3 punkte kaip Xiao, Rasul, Vollgraf
(2017) empirinių rezultatų šaltinis). Duomenys atsisiųsti vieną kartą per
git clone --depth 1 į data/fmnist_repo/, tada šis modulis juos nuskaito
tiesiai iš .gz idx failų (be papildomos priklausomybės nuo openml.org,
kuris šioje aplinkoje nepasiekiamas per tinklo politiką).

Grąžinamas fiksuotas train/test skaidymas: 60000 mokymo, 10000 testavimo
vaizdų, kaip reikalauja kolokviumo plano "Duomenų skaidymo reikalavimas".
"""

import gzip
import os
import struct

import numpy as np

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "fmnist_repo", "data", "fashion")


def _read_idx_images(path):
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, f"Netikėta magic reikšmė {magic} faile {path}"
        buf = f.read(n * rows * cols)
        images = np.frombuffer(buf, dtype=np.uint8).reshape(n, rows * cols)
    return images


def _read_idx_labels(path):
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        assert magic == 2049, f"Netikėta magic reikšmė {magic} faile {path}"
        buf = f.read(n)
        labels = np.frombuffer(buf, dtype=np.uint8)
    return labels


def load_fashion_mnist(data_dir=None):
    """Grąžina (X_train, y_train, X_test, y_test).

    X_* forma: (n, 784) uint8, reikšmės [0, 255].
    y_* forma: (n,) uint8, reikšmės [0, 9].
    Tai atitinka oficialų, fiksuotą Fashion-MNIST train/test skaidymą.
    """
    d = data_dir or _DATA_DIR
    X_train = _read_idx_images(os.path.join(d, "train-images-idx3-ubyte.gz"))
    y_train = _read_idx_labels(os.path.join(d, "train-labels-idx1-ubyte.gz"))
    X_test = _read_idx_images(os.path.join(d, "t10k-images-idx3-ubyte.gz"))
    y_test = _read_idx_labels(os.path.join(d, "t10k-labels-idx1-ubyte.gz"))
    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_fashion_mnist()
    print("X_train:", X_train.shape, X_train.dtype)
    print("y_train:", y_train.shape, y_train.dtype, "unique:", np.unique(y_train))
    print("X_test:", X_test.shape, X_test.dtype)
    print("y_test:", y_test.shape, y_test.dtype, "unique:", np.unique(y_test))
    print("Klasių balansas train:", np.bincount(y_train))
    print("Klasių balansas test:", np.bincount(y_test))
