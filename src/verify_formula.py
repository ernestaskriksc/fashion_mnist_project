"""
Formulės-kodo ryšio patikra (egzamino reikalavimas: "Bent vienam modeliui
parodyti formulės ar taisyklės ryšį su konkrečia kodo vieta"; taip pat
kolokviumo 7 punkte pažadėta "Formulių tikrinimas": "apskaičiavus
sprendimo funkciją rankiniu būdu mažam pavyzdžiui ir palyginus su
biblioteka gautu rezultatu").

Kolokviumo 5 punkto formulė:
    f(x) = sign( Σ_i alpha_i * y_i * K(x_i, x) + b )
    K(x_i, x_j) = exp( -gamma * ||x_i - x_j||^2 )   (RBF branduolys)

sklearn.svm.SVC daugiaklasiam atvejui viduje naudoja glaustą one-vs-one
dual_coef_ koduotę, kurią rankiniu būdu atkartoti būtų klaidoms imli, tad
patikrai apmokamas ATSKIRAS, paprastas BINARINIS SVM dviem klasėms (Shirt
vs T-shirt/top -- kaip tik tos dvi klasės, kurias pagrindinis modelis
painioja dažniausiai, žr. results/tables/top_confusions_svm.json). Šiam
binariniam atvejui formulės kintamieji tiesiogiai atitinka scikit-learn
atributus:
    alpha_i * y_i  ->  clf.dual_coef_[0]      (jau su y_i ženklu)
    x_i            ->  clf.support_vectors_
    b              ->  clf.intercept_[0]
    gamma          ->  clf.gamma (arba clf._gamma po fit())
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

from data import load_fashion_mnist
from preprocess import normalize_raw
from config import RANDOM_SEED, CLASS_NAMES


def rbf_kernel_row(x, support_vectors, gamma):
    """K(x_i, x) visiems atraminiams vektoriams x_i vienam taškui x."""
    diff = support_vectors - x  # (n_sv, 784)
    sq_dist = np.sum(diff ** 2, axis=1)
    return np.exp(-gamma * sq_dist)


def manual_decision_function(x, clf):
    """f(x) BE sign() -- neapdorotas atstumas, tiesiogiai pagal formulę
    f(x) = sum_i (alpha_i * y_i) * K(x_i, x) + b."""
    K_row = rbf_kernel_row(x, clf.support_vectors_, clf._gamma)
    return float(np.dot(clf.dual_coef_[0], K_row) + clf.intercept_[0])


def main():
    X_train, y_train, X_test, y_test = load_fashion_mnist()
    shirt_idx = CLASS_NAMES.index("Shirt")
    tshirt_idx = CLASS_NAMES.index("T-shirt/top")

    mask_train = np.isin(y_train, [shirt_idx, tshirt_idx])
    X_bin, y_bin = X_train[mask_train], y_train[mask_train]
    X_bin_sub, _, y_bin_sub, _ = train_test_split(
        X_bin, y_bin, train_size=2000, stratify=y_bin, random_state=RANDOM_SEED)
    X_bin_n = normalize_raw(X_bin_sub)

    clf = SVC(kernel="rbf", C=10, gamma="scale", random_state=RANDOM_SEED)
    clf.fit(X_bin_n, y_bin_sub)
    print(f"Binarinis SVM (Shirt vs T-shirt/top) apmokytas ant {len(y_bin_sub)} "
          f"pavyzdžių, {clf.support_.shape[0]} atraminių vektorių, "
          f"gamma={clf._gamma:.6f}")

    mask_test = np.isin(y_test, [shirt_idx, tshirt_idx])
    X_test_bin = normalize_raw(X_test[mask_test][:10])

    print("\nx_i indeksas | biblioteka decision_function(x) | rankinė f(x) (be sign) | skirtumas")
    max_abs_diff = 0.0
    for i, x in enumerate(X_test_bin):
        lib_val = clf.decision_function(x.reshape(1, -1))[0]
        manual_val = manual_decision_function(x, clf)
        diff = abs(lib_val - manual_val)
        max_abs_diff = max(max_abs_diff, diff)
        print(f"{i:12d} | {lib_val:28.6f} | {manual_val:22.6f} | {diff:.2e}")

    print(f"\nDidžiausias absoliutus skirtumas tarp 10 pavyzdžių: {max_abs_diff:.2e}")
    assert max_abs_diff < 1e-6, "Formulė ir biblioteka NESUTAMPA -- patikrinti implementaciją!"
    print("PATVIRTINTA: rankinis f(x) skaičiavimas pagal 5-o kolokviumo punkto "
          "formulę sutampa su sklearn.svm.SVC.decision_function() (< 1e-6 skirtumas).")


if __name__ == "__main__":
    main()
