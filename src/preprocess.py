"""
Duomenų paruošimo modulis (kolokviumo plano 2 punktas: "Duomenų paruošimas").

- Iš train dalies stratifikuotai atskiriama validacijos imtis (10%).
- Pikselių reikšmės normalizuojamos į [0, 1].
- Du požymių variantai: 'raw' (784-dim išlyginti pikseliai) ir 'hog'
  (Histogram of Oriented Gradients) -- naudojami abliacijai (6 punktas,
  "Požymių tipas: neapdoroti pikseliai vs. HOG požymiai").
- Normalizavimo/HOG parametrai fiksuojami TIK train dalyje, kad
  nebūtų duomenų nutekėjimo į val/test (žr. kolokviumo 6 punkto
  "Didžiausios grėsmės išvadų galiojimui").
"""

import numpy as np
from sklearn.model_selection import train_test_split
from skimage.feature import hog

RANDOM_SEED = 42  # fiksuota atsitiktinumo generatoriaus pradinė reikšmė


def split_train_val(X_train, y_train, val_fraction=0.1, seed=RANDOM_SEED):
    """Stratifikuotai atskiria validaciją iš mokymo dalies. Test imties tai neliečia."""
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train,
        test_size=val_fraction,
        stratify=y_train,
        random_state=seed,
    )
    return X_tr, y_tr, X_val, y_val


def normalize_raw(X, X_ref=None):
    """Min-max normalizavimas į [0,1]. Fashion-MNIST pikseliai visada [0,255],
    tad tai fiksuota transformacija (nereikia fituoti statistikų iš duomenų),
    bet X_ref parametras paliktas suderinamumui, jei vėliau norėtume pereiti
    prie z-score normalizavimo su train vidurkiu/dispersija."""
    return X.astype(np.float32) / 255.0


def extract_hog_features(X_uint8, pixels_per_cell=(4, 4), cells_per_block=(2, 2)):
    """Paverčia (n, 784) uint8 vaizdus į HOG požymių vektorius.
    Naudojama tik abliacijai -- pagrindiniame eksperimente naudojami raw pikseliai,
    kad rezultatas būtų tiesiogiai palyginamas su kNN baseline, kuris taip pat
    veikia raw pikselių erdvėje."""
    n = X_uint8.shape[0]
    imgs = X_uint8.reshape(n, 28, 28)
    feats = []
    for i in range(n):
        f = hog(
            imgs[i],
            orientations=9,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            block_norm="L2-Hys",
        )
        feats.append(f)
    return np.asarray(feats, dtype=np.float32)


if __name__ == "__main__":
    from data import load_fashion_mnist
    X_train, y_train, X_test, y_test = load_fashion_mnist()
    X_tr, y_tr, X_val, y_val = split_train_val(X_train, y_train)
    print("train:", X_tr.shape, "val:", X_val.shape, "test:", X_test.shape)
    print("train klasių balansas:", np.bincount(y_tr))
    print("val klasių balansas:", np.bincount(y_val))
    X_tr_norm = normalize_raw(X_tr)
    print("normalizuota train min/max:", X_tr_norm.min(), X_tr_norm.max())
    hog_sample = extract_hog_features(X_tr[:5])
    print("HOG požymių dydis vienam vaizdui:", hog_sample.shape[1])
