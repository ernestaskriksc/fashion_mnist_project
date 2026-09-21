"""
Vertinimo modulis (kolokviumo plano 2 punktas: "Vertinimas"; egzamino
reikalavimas: "Pateikti rezultatų lentelę... ir paaiškinti, kodėl
pasirinktas metodas laimėjo arba nelaimėjo").

SVARBU: test imtis (X_test, y_test) čia naudojama TIK VIENĄ KARTĄ, po to,
kai visi hiperparametrai jau pasirinkti train.py per validaciją. Tai
tiesiogiai įgyvendina egzamino reikalavimą "Hiperparametrų parinkimui
nenaudoti galutinio testo".
"""

import json
import os
import time

import joblib
import numpy as np
from sklearn.metrics import (
    f1_score, confusion_matrix, classification_report,
)

from config import MODELS_DIR, TABLES_DIR, CLASS_NAMES
from data import load_fashion_mnist
from preprocess import split_train_val, normalize_raw


def model_size_mb(path):
    return os.path.getsize(path) / (1024 * 1024)


def measure_inference_latency(model, X_one, n_repeats=50):
    """Vidutinė inferencijos trukmė VIENAM vaizdui (ms) -- imituoja 2-ame
    punkte aprašytą "Naudojimo" etapą (vienas naujas vaizdas iš karto)."""
    # apšilimas
    model.predict(X_one)
    times = []
    for _ in range(n_repeats):
        t0 = time.time()
        model.predict(X_one)
        times.append((time.time() - t0) * 1000)
    return float(np.mean(times)), float(np.std(times))


def main():
    os.makedirs(TABLES_DIR, exist_ok=True)
    X_train, y_train, X_test, y_test = load_fashion_mnist()
    _, _, _, _ = split_train_val(X_train, y_train)  # tik konsistencijai su train.py
    X_test_n = normalize_raw(X_test)

    models = {
        "kNN baseline": ("knn.joblib", "#8a8a8a"),
        "SVM (RBF)": ("svm.joblib", "#2563eb"),
        "MLP": ("mlp.joblib", "#059669"),
    }

    rows = []
    confusions = {}
    all_preds = {}
    one_sample = X_test_n[:1]

    for name, (fname, _color) in models.items():
        path = os.path.join(MODELS_DIR, fname)
        model = joblib.load(path)
        t0 = time.time()
        y_pred = model.predict(X_test_n)
        predict_all_seconds = time.time() - t0
        macro_f1 = f1_score(y_test, y_pred, average="macro")
        acc = (y_pred == y_test).mean()
        cm = confusion_matrix(y_test, y_pred)
        confusions[name] = cm
        all_preds[name] = y_pred

        size_mb = model_size_mb(path)
        lat_mean, lat_std = measure_inference_latency(model, one_sample)

        rows.append({
            "Metodas": name,
            "Test Macro-F1": round(macro_f1, 4),
            "Test tikslumas": round(acc, 4),
            "Modelio dydis (MB)": round(size_mb, 2),
            "Inferencija/vaizdui (ms, vid.)": round(lat_mean, 2),
            "Inferencija/vaizdui (ms, std)": round(lat_std, 2),
            "Viso test (10000) sek.": round(predict_all_seconds, 1),
        })
        print(f"{name}: Macro-F1={macro_f1:.4f} tikslumas={acc:.4f} "
              f"dydis={size_mb:.2f}MB inferencija={lat_mean:.2f}ms")

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(TABLES_DIR, "comparison_table.csv"), index=False)
    print("\n" + df.to_string(index=False))

    # Klasifikavimo ataskaita ir painiavos matricos pagrindiniam metodui (SVM)
    y_pred_svm = all_preds["SVM (RBF)"]
    report = classification_report(y_test, y_pred_svm, target_names=CLASS_NAMES,
                                    output_dict=True)
    with open(os.path.join(TABLES_DIR, "svm_classification_report.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    np.save(os.path.join(TABLES_DIR, "y_test.npy"), y_test)
    for name in models:
        np.save(os.path.join(TABLES_DIR, f"preds_{name.split()[0]}.npy"), all_preds[name])
        np.save(os.path.join(TABLES_DIR, f"cm_{name.split()[0]}.npy"), confusions[name])

    # Didžiausios painiavos poros SVM modeliui (klaidų analizei)
    cm = confusions["SVM (RBF)"].copy()
    np.fill_diagonal(cm, 0)
    pairs = []
    for i in range(10):
        for j in range(10):
            if i != j and cm[i, j] > 0:
                pairs.append((cm[i, j], CLASS_NAMES[i], CLASS_NAMES[j]))
    pairs.sort(reverse=True)
    print("\nTop 5 painiavos poros (SVM, tikra -> prognozuota):")
    for count, true_c, pred_c in pairs[:5]:
        print(f"  {true_c} -> {pred_c}: {count} klaidų")
    with open(os.path.join(TABLES_DIR, "top_confusions_svm.json"), "w") as f:
        json.dump([{"true": t, "pred": p, "count": int(c)} for c, t, p in pairs[:10]],
                   f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
