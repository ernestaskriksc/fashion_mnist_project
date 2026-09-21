#!/usr/bin/env python3
"""
Viena komanda pagrindiniam eksperimentui pakartoti (egzamino reikalavimas).

Naudojimas:
    python3 run_all.py

Paleidžia visą grandinę nuo duomenų atsisiuntimo (jei dar neatsiųsta) iki
galutinių grafikų: train -> evaluate -> ablation -> robustness -> visualize
-> verify_formula. Visos atsitiktinumo sėklos fiksuotos config.py faile
(RANDOM_SEED = 42), tad pakartotinis paleidimas turėtų grąžinti tuos
pačius (iki skaičiavimo tikslumo) rezultatus.

Reikalavimai: žr. requirements.txt (pip install -r requirements.txt).
Duomenų atsisiuntimui reikalingas interneto ryšys prie github.com (žr.
README.md "Duomenys" skyrių) -- šis skriptas atsisiuntimą atlieka
automatiškai, jei duomenų dar nėra.
"""

import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
DATA_DIR = os.path.join(ROOT, "data", "fmnist_repo", "data", "fashion")


def ensure_data():
    required = [
        "train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz",
    ]
    if all(os.path.exists(os.path.join(DATA_DIR, f)) for f in required):
        print("Duomenys jau atsisiųsti:", DATA_DIR)
        return
    print("Duomenys nerasti -- atsisiunčiama iš "
          "https://github.com/zalandoresearch/fashion-mnist ...")
    repo_dir = os.path.join(ROOT, "data", "fmnist_repo")
    os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1",
         "https://github.com/zalandoresearch/fashion-mnist.git", repo_dir],
        check=True,
    )


def run_step(name, script):
    print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")
    t0 = time.time()
    result = subprocess.run([sys.executable, script], cwd=ROOT)
    if result.returncode != 0:
        print(f"KLAIDA vykdant {script} (kodas {result.returncode}) -- stabdoma.")
        sys.exit(result.returncode)
    print(f"[{name} baigta per {time.time() - t0:.1f}s]")


def main():
    for d in [("results", "models"), ("results", "figures"), ("results", "tables")]:
        os.makedirs(os.path.join(ROOT, *d), exist_ok=True)

    ensure_data()
    run_step("1/6 Modelio mokymas (kNN, SVM, MLP)", os.path.join(SRC, "train.py"))
    run_step("2/6 Vertinimas test imtimi", os.path.join(SRC, "evaluate.py"))
    run_step("3/6 Abliacija (raw vs HOG)", os.path.join(SRC, "ablation.py"))
    run_step("4/6 Atsparumo bandymas (triukšmas, uždengimas)",
              os.path.join(SRC, "robustness.py"))
    run_step("5/6 Vizualizacijos", os.path.join(SRC, "visualize.py"))
    run_step("6/6 Formulės-kodo ryšio patikra", os.path.join(SRC, "verify_formula.py"))

    print("\nVisi rezultatai: results/tables/, results/figures/, results/models/")


if __name__ == "__main__":
    main()
