"""
Atsparumo bandymas (egzamino reikalavimas: "atlikti... bent vieną atsparumo
bandymą: triukšmui, trūkstamiems duomenims, laiko ar grupės poslinkiui").

Pasirinktas triukšmas + dalinis uždengimas (occlusion), nes tai tiesiogiai
operacionalizuoja kolokviumo 1 punkte aprašytus "Neaiškumus": realaus
sandėlio vaizdai turės šešėlių/foninio triukšmo ir gali būti dalinai
uždengti, o Fashion-MNIST vaizdai yra švarūs. Testuojami du atskiri
degradacijos tipai su keliais sunkumo lygiais, visiems trims modeliams,
ant to paties stratifikuoto test poaibio (config.SECONDARY_TEST_SIZE).
"""

import json
import os

import joblib
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from config import MODELS_DIR, TABLES_DIR, SECONDARY_TEST_SIZE, RANDOM_SEED
from data import load_fashion_mnist
from preprocess import normalize_raw


def add_gaussian_noise(X01, sigma, rng):
    """X01 -- normalizuoti [0,1] pikseliai. sigma -- triukšmo std [0,1] skalėje."""
    noisy = X01 + rng.normal(0, sigma, X01.shape).astype(np.float32)
    return np.clip(noisy, 0.0, 1.0)


def add_occlusion(X01, patch_frac, rng):
    """Uždengia atsitiktinį kvadratinį regioną (patch_frac -- dalis 28x28
    kraštinės), imituojant dalinai uždengtą rūbą (žr. 1 punkto neaiškumus)."""
    n = X01.shape[0]
    imgs = X01.reshape(n, 28, 28).copy()
    patch = max(1, int(round(28 * patch_frac)))
    for i in range(n):
        r0 = rng.integers(0, 28 - patch + 1)
        c0 = rng.integers(0, 28 - patch + 1)
        imgs[i, r0:r0 + patch, c0:c0 + patch] = 0.0
    return imgs.reshape(n, 784)


def main():
    rng = np.random.default_rng(RANDOM_SEED)
    X_train, y_train, X_test, y_test = load_fashion_mnist()
    X_sub, _, y_sub, _ = train_test_split(
        X_test, y_test, train_size=SECONDARY_TEST_SIZE,
        stratify=y_test, random_state=RANDOM_SEED)
    X_sub_n = normalize_raw(X_sub)

    models = {
        "kNN baseline": joblib.load(os.path.join(MODELS_DIR, "knn.joblib")),
        "SVM (RBF)": joblib.load(os.path.join(MODELS_DIR, "svm.joblib")),
        "MLP": joblib.load(os.path.join(MODELS_DIR, "mlp.joblib")),
    }

    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4]
    occlusion_levels = [0.0, 0.15, 0.25, 0.35, 0.45]  # dalis 28px kraštinės

    results = {"noise": {}, "occlusion": {}, "n_samples": SECONDARY_TEST_SIZE}

    print("== Atsparumas triukšmui (Gauso) ==")
    for sigma in noise_levels:
        X_noisy = add_gaussian_noise(X_sub_n, sigma, rng)
        row = {}
        for name, model in models.items():
            y_pred = model.predict(X_noisy)
            f1 = f1_score(y_sub, y_pred, average="macro")
            row[name] = round(f1, 4)
        results["noise"][str(sigma)] = row
        print(f"sigma={sigma}: " + ", ".join(f"{k}={v:.4f}" for k, v in row.items()))

    print("\n== Atsparumas daliniam uždengimui ==")
    for frac in occlusion_levels:
        X_occ = add_occlusion(X_sub_n, frac, np.random.default_rng(RANDOM_SEED))
        row = {}
        for name, model in models.items():
            y_pred = model.predict(X_occ)
            f1 = f1_score(y_sub, y_pred, average="macro")
            row[name] = round(f1, 4)
        results["occlusion"][str(frac)] = row
        print(f"patch_frac={frac}: " + ", ".join(f"{k}={v:.4f}" for k, v in row.items()))

    os.makedirs(TABLES_DIR, exist_ok=True)
    with open(os.path.join(TABLES_DIR, "robustness.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\nIšsaugota: results/tables/robustness.json")


if __name__ == "__main__":
    main()
