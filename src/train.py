"""
Modelio mokymo modulis (kolokviumo plano 2 punktas: "Modelio mokymas";
egzamino reikalavimas: "Palyginti metodus tuo pačiu duomenų skaidymu ir
tomis pačiomis metrikomis. Hiperparametrų parinkimui nenaudoti galutinio
testo.").

Trys modeliai, visi mokomi ant TO PATIES fiksuoto, stratifikuoto mokymo
poaibio (config.MAIN_TRAIN_SIZE), kad palyginimas būtų sąžiningas:
  - kNN baseline (su k, parinktu validacijoje)
  - SVM (RBF branduolys, pagrindinis metodas -- žr. kolokviumo 4 punktą),
    su (C, gamma) hiperparametrų paieška ant mažesnio poaibio
  - MLP (256-128-100, antrasis intelektualusis metodas)

Hiperparametrų paieška visada vertinama TIK validacijos imtimi (X_val,
y_val). Test imtis (X_test, y_test) šiame faile niekur nenaudojama --
tai patikrinama atskirai evaluate.py, kuris testą naudoja tik kartą,
galutiniam įvertinimui.
"""

import json
import os
import time

import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from config import (
    RANDOM_SEED, MAIN_TRAIN_SIZE, HPARAM_SEARCH_SIZE, MODELS_DIR, RESULTS_DIR,
)
from data import load_fashion_mnist
from preprocess import split_train_val, normalize_raw


def make_subsample(X, y, n, seed=RANDOM_SEED):
    if n >= len(y):
        return X, y
    X_sub, _, y_sub, _ = train_test_split(
        X, y, train_size=n, stratify=y, random_state=seed
    )
    return X_sub, y_sub


def search_knn_k(X_tr, y_tr, X_val, y_val, ks=(1, 3, 5, 7, 9)):
    results = {}
    for k in ks:
        t0 = time.time()
        clf = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
        clf.fit(X_tr, y_tr)
        acc = clf.score(X_val, y_val)
        results[k] = {"val_acc": acc, "seconds": time.time() - t0}
        print(f"  kNN k={k}: val_acc={acc:.4f} ({results[k]['seconds']:.1f}s)")
    best_k = max(results, key=lambda k: results[k]["val_acc"])
    return best_k, results


def search_svm_hparams(X_tr_small, y_tr_small, X_val, y_val,
                        C_values=(1, 10), gamma_values=("scale", 0.03)):
    results = []
    for C in C_values:
        for gamma in gamma_values:
            t0 = time.time()
            clf = SVC(kernel="rbf", C=C, gamma=gamma, random_state=RANDOM_SEED)
            clf.fit(X_tr_small, y_tr_small)
            acc = clf.score(X_val, y_val)
            dt = time.time() - t0
            print(f"  SVM C={C} gamma={gamma}: val_acc={acc:.4f} ({dt:.1f}s, "
                  f"n_support={clf.support_.shape[0]})")
            results.append({"C": C, "gamma": gamma, "val_acc": acc, "seconds": dt})
    best = max(results, key=lambda r: r["val_acc"])
    return best["C"], best["gamma"], results


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    log = {"random_seed": RANDOM_SEED, "main_train_size": MAIN_TRAIN_SIZE,
           "hparam_search_size": HPARAM_SEARCH_SIZE}

    print("== Duomenų gavimas ir paruošimas ==")
    X_train, y_train, X_test, y_test = load_fashion_mnist()
    X_tr, y_tr, X_val, y_val = split_train_val(X_train, y_train, seed=RANDOM_SEED)
    X_val_n = normalize_raw(X_val)

    X_tr_main, y_tr_main = make_subsample(X_tr, y_tr, MAIN_TRAIN_SIZE)
    X_tr_main_n = normalize_raw(X_tr_main)
    print(f"Pagrindinis mokymo poaibis: {X_tr_main.shape[0]} pavyzdžių "
          f"(klasių balansas: {np.bincount(y_tr_main).tolist()})")
    log["main_train_class_balance"] = np.bincount(y_tr_main).tolist()

    X_tr_small, y_tr_small = make_subsample(X_tr, y_tr, HPARAM_SEARCH_SIZE)
    X_tr_small_n = normalize_raw(X_tr_small)

    print("\n== kNN baseline: k paieška validacijoje ==")
    best_k, knn_search = search_knn_k(X_tr_main_n, y_tr_main, X_val_n, y_val)
    print(f"Geriausias k = {best_k}")
    t0 = time.time()
    knn = KNeighborsClassifier(n_neighbors=best_k, n_jobs=-1)
    knn.fit(X_tr_main_n, y_tr_main)
    log["knn"] = {"best_k": best_k, "search": knn_search,
                   "final_fit_seconds": time.time() - t0}
    joblib.dump(knn, os.path.join(MODELS_DIR, "knn.joblib"))

    print("\n== SVM: (C, gamma) paieška ant mažesnio poaibio ==")
    best_C, best_gamma, svm_search = search_svm_hparams(
        X_tr_small_n, y_tr_small, X_val_n, y_val)
    print(f"Geriausi hiperparametrai: C={best_C}, gamma={best_gamma}")
    print("Galutinis SVM mokymas ant pagrindinio poaibio...")
    t0 = time.time()
    svm = SVC(kernel="rbf", C=best_C, gamma=best_gamma,
              probability=True, random_state=RANDOM_SEED)
    # probability=True įjungia Platt skalinimą (kolokviumo 5 punktas) --
    # sklearn tai realizuoja kaip papildomą 5-fold kryžminį kalibravimą
    # ant mokymo duomenų, paverčiantį sprendimo funkcijos reikšmes į [0,1].
    svm.fit(X_tr_main_n, y_tr_main)
    fit_seconds = time.time() - t0
    print(f"SVM galutinis mokymas: {fit_seconds:.1f}s, "
          f"atraminių vektorių: {svm.support_.shape[0]}")
    log["svm"] = {"best_C": best_C, "best_gamma": best_gamma, "search": svm_search,
                   "final_fit_seconds": fit_seconds,
                   "n_support_vectors": int(svm.support_.shape[0])}
    joblib.dump(svm, os.path.join(MODELS_DIR, "svm.joblib"))

    print("\n== MLP (256-128-100) ==")
    t0 = time.time()
    mlp = MLPClassifier(hidden_layer_sizes=(256, 128, 100), max_iter=150,
                         random_state=RANDOM_SEED, early_stopping=True,
                         n_iter_no_change=10)
    mlp.fit(X_tr_main_n, y_tr_main)
    fit_seconds = time.time() - t0
    val_acc = mlp.score(X_val_n, y_val)
    print(f"MLP mokymas: {fit_seconds:.1f}s, n_iter={mlp.n_iter_}, val_acc={val_acc:.4f}")
    log["mlp"] = {"final_fit_seconds": fit_seconds, "n_iter": int(mlp.n_iter_),
                   "val_acc": val_acc}
    joblib.dump(mlp, os.path.join(MODELS_DIR, "mlp.joblib"))

    with open(os.path.join(RESULTS_DIR, "train_log.json"), "w") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    print(f"\nIšsaugota: {MODELS_DIR}{os.sep}(knn,svm,mlp).joblib, "
          f"{os.path.join(RESULTS_DIR, 'train_log.json')}")


if __name__ == "__main__":
    main()
