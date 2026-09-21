"""Rezultatų vizualizacijos: palyginimo grafikas, painiavos matrica,
atsparumo kreivės, abliacijos grafikas, klaidingai suklasifikuotų
pavyzdžių galerija. (Egzamino reikalavimas: "tinkamą grafiką, klaidų
pavyzdžius".)"""

import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import FIGURES_DIR, TABLES_DIR, MODELS_DIR, CLASS_NAMES
from data import load_fashion_mnist
from preprocess import normalize_raw

COLORS = {"kNN baseline": "#9ca3af", "SVM (RBF)": "#2563eb", "MLP": "#059669"}
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#374151", "axes.labelcolor": "#111827",
    "text.color": "#111827", "xtick.color": "#374151", "ytick.color": "#374151",
    "font.size": 11, "axes.grid": True, "grid.color": "#e5e7eb", "grid.linewidth": 0.7,
    "axes.axisbelow": True,
})


def fig_comparison():
    df = pd.read_csv(os.path.join(TABLES_DIR, "comparison_table.csv"))
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = [COLORS[m] for m in df["Metodas"]]
    bars = ax.bar(df["Metodas"], df["Test Macro-F1"], color=colors, width=0.55)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Test Macro-F1")
    ax.set_title("Metodų palyginimas fiksuotoje test imtyje (n=10000)")
    ax.axhline(df.loc[df["Metodas"] == "kNN baseline", "Test Macro-F1"].values[0],
               color="#9ca3af", linestyle="--", linewidth=1, label="baseline lygis")
    for b, v in zip(bars, df["Test Macro-F1"]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}",
                ha="center", fontsize=10)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "comparison_macro_f1.png"), dpi=160)
    plt.close(fig)


def fig_confusion_matrix():
    cm = np.load(os.path.join(TABLES_DIR, "cm_SVM.npy"))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(10)); ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticks(range(10)); ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Prognozuota klasė"); ax.set_ylabel("Tikra klasė")
    ax.set_title("SVM painiavos matrica (normalizuota pagal eilutę), test n=10000")
    for i in range(10):
        for j in range(10):
            v = cm_norm[i, j]
            if v > 0.02:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                         color="white" if v > 0.5 else "#111827", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8, label="dalis eilutėje")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "confusion_matrix_svm.png"), dpi=160)
    plt.close(fig)


def fig_robustness():
    with open(os.path.join(TABLES_DIR, "robustness.json")) as f:
        rob = json.load(f)

    for kind, xlabel, fname, title in [
        ("noise", "Gauso triukšmo std (σ)", "robustness_noise.png",
         "Atsparumas triukšmui"),
        ("occlusion", "Uždengto ploto dalis nuo kraštinės",
         "robustness_occlusion.png", "Atsparumas daliniam uždengimui"),
    ]:
        levels = sorted(rob[kind].keys(), key=float)
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        for model_name in ["kNN baseline", "SVM (RBF)", "MLP"]:
            ys = [rob[kind][lv][model_name] for lv in levels]
            ax.plot([float(lv) for lv in levels], ys, marker="o",
                     color=COLORS[model_name], label=model_name, linewidth=2)
        ax.set_xlabel(xlabel); ax.set_ylabel("Macro-F1 (n=%d)" % rob["n_samples"])
        ax.set_ylim(0, 1.0)
        ax.set_title(title)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(FIGURES_DIR, fname), dpi=160)
        plt.close(fig)


def fig_ablation():
    with open(os.path.join(TABLES_DIR, "ablation_raw_vs_hog.json")) as f:
        ab = json.load(f)
    labels = list(ab.keys())
    values = [ab[l]["macro_f1"] for l in labels]
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(labels, values, color=["#2563eb", "#f59e0b"], width=0.5)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Test Macro-F1")
    ax.set_title("Abliacija: požymių tipas (SVM, tas pats mokymo poaibis)")
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "ablation_raw_vs_hog.png"), dpi=160)
    plt.close(fig)


def fig_error_gallery():
    X_train, y_train, X_test, y_test = load_fashion_mnist()
    y_pred = np.load(os.path.join(TABLES_DIR, "preds_SVM.npy"))
    with open(os.path.join(TABLES_DIR, "top_confusions_svm.json")) as f:
        top = json.load(f)

    fig, axes = plt.subplots(2, 5, figsize=(13, 6.5))
    fig.subplots_adjust(hspace=0.55, wspace=0.25, top=0.87, bottom=0.03)
    shown = 0
    for pair in top:
        true_idx = CLASS_NAMES.index(pair["true"])
        pred_idx = CLASS_NAMES.index(pair["pred"])
        mask = (y_test == true_idx) & (y_pred == pred_idx)
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            continue
        img = X_test[idxs[0]].reshape(28, 28)
        ax = axes.flat[shown]
        ax.imshow(img, cmap="gray")
        ax.set_title(f"tikra: {pair['true']}\nprognozė: {pair['pred']}", fontsize=9, pad=6)
        ax.axis("off")
        shown += 1
        if shown >= 10:
            break
    fig.suptitle("SVM klaidų pavyzdžiai (dažniausios painiavos poros)", y=0.98)
    fig.savefig(os.path.join(FIGURES_DIR, "error_examples_svm.png"), dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig_comparison()
    fig_confusion_matrix()
    fig_robustness()
    fig_ablation()
    fig_error_gallery()
    print("Grafikai išsaugoti:", os.listdir(FIGURES_DIR))
