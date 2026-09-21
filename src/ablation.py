"""
Abliacija / požymių jautrumo bandymas (kolokviumo 6 punktas, egzamino
reikalavimas: "Atlikti bent vieną abliaciją arba požymių jautrumo bandymą").

Testuojama: neapdoroti (raw) pikseliai vs. HOG požymiai pagrindiniam
metodui (SVM). Tai tiesiogiai patikrina 3-ame kolokviumo punkte cituotą
teiginį (Xiao, Rasul, Vollgraf, 2017), kad HOG+SVM pasiekia ~92,6%
tikslumą -- mūsų eksperimente naudojami raw pikseliai (žr. evaluate.py),
tad šis bandymas parodo, kiek tiksliai HOG pagerina rezultatą tuo pačiu
mokymo poaibiu ir tais pačiais hiperparametrais.

Naudojamas tas pats MAIN_TRAIN_SIZE stratifikuotas poaibis kaip ir
pagrindiniame eksperimente, kad palyginimas liktų sąžiningas (vienintelis
kintamasis -- požymių tipas).
"""

import json
import os
import time

import numpy as np
from sklearn.metrics import f1_score
from sklearn.svm import SVC

from config import MAIN_TRAIN_SIZE, TABLES_DIR
from data import load_fashion_mnist
from preprocess import split_train_val, normalize_raw, extract_hog_features
from train import make_subsample, RANDOM_SEED


def main():
    X_train, y_train, X_test, y_test = load_fashion_mnist()
    X_tr, y_tr, X_val, y_val = split_train_val(X_train, y_train)
    X_tr_main, y_tr_main = make_subsample(X_tr, y_tr, MAIN_TRAIN_SIZE)

    print("HOG požymių skaičiavimas mokymo poaibiui ir test imčiai...")
    t0 = time.time()
    X_tr_hog = extract_hog_features(X_tr_main)
    X_test_hog = extract_hog_features(X_test)
    print(f"HOG skaičiavimas: {time.time() - t0:.1f}s "
          f"(train {X_tr_hog.shape}, test {X_test_hog.shape})")

    # HOG požymiai jau natūraliai apytiksliai [0,1] normalizuoti (L2-Hys
    # blokų normalizacija), tad papildomo min-max skalinimo netaikome --
    # tai skiriasi nuo raw pikselių, kur normalizavimas būtinas (žr.
    # preprocess.py "Duomenų paruošimas"), ir yra sąmoningas, dokumentuotas
    # sprendimas, o ne praleistas žingsnis.
    X_tr_raw_n = normalize_raw(X_tr_main)
    X_test_raw_n = normalize_raw(X_test)

    results = {}
    for label, (Xtr, Xte) in {
        "raw_pikseliai": (X_tr_raw_n, X_test_raw_n),
        "HOG": (X_tr_hog, X_test_hog),
    }.items():
        t0 = time.time()
        clf = SVC(kernel="rbf", C=10, gamma="scale", random_state=RANDOM_SEED)
        clf.fit(Xtr, y_tr_main)
        fit_s = time.time() - t0
        y_pred = clf.predict(Xte)
        macro_f1 = f1_score(y_test, y_pred, average="macro")
        acc = (y_pred == y_test).mean()
        results[label] = {"macro_f1": round(macro_f1, 4), "accuracy": round(acc, 4),
                           "fit_seconds": round(fit_s, 1),
                           "n_support": int(clf.support_.shape[0])}
        print(f"{label}: Macro-F1={macro_f1:.4f} tikslumas={acc:.4f} ({fit_s:.1f}s)")

    os.makedirs(TABLES_DIR, exist_ok=True)
    with open(os.path.join(TABLES_DIR, "ablation_raw_vs_hog.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\nIšsaugota: results/tables/ablation_raw_vs_hog.json")


if __name__ == "__main__":
    main()
