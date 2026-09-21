# Drabužių vaizdų klasifikavimas (Fashion-MNIST) — galutinio egzamino sprendimas

PEPfm-26, Ernestas Krikščiūnas. Šis katalogas įgyvendina kolokviumo plane
(žr. `PEPfm-26_Ernestas_Kriksciunas_kol.docx`) aprašytą sprendimą ir
atitinka „Galutinio egzamino dalies" reikalavimus. Kiekvienas
reikalavimo punktas su konkrečiu failu susietas `FINAL_REPORT.md` faile.

## Duomenys

Naudojamas oficialus Fashion-MNIST rinkinys (Xiao, Rasul, Vollgraf, 2017),
idx-ubyte formatu iš https://github.com/zalandoresearch/fashion-mnist —
tas pats šaltinis, kurio DOI/nuoroda cituota kolokviumo 3 punkte.
`sklearn.datasets.fetch_openml` (OpenML API) šioje aplinkoje tinklo
politika buvo užblokuota, tad duomenys atsisiunčiami tiesiai iš oficialaus
GitHub repo — tai tas pats duomenų rinkinys, tik kitas atsisiuntimo kelias.

Duomenų failų kopijų faile neteikiame — jos atsisiunčiamos automatiškai
paleidus `run_all.py` (arba rankiniu būdu: `git clone --depth 1
https://github.com/zalandoresearch/fashion-mnist.git data/fmnist_repo`).

## Priklausomybės

```
pip install -r requirements.txt
```

Tikslios versijos, su kuriomis kodas buvo faktiškai paleistas ir
patikrintas, nurodytos `requirements.txt`.

## Viena komanda pagrindiniam eksperimentui pakartoti

```
python3 run_all.py
```

Paleidžia visą grandinę: duomenų atsisiuntimą (jei reikia) → modelio
mokymą (kNN, SVM, MLP) → vertinimą test imtimi → abliaciją → atsparumo
bandymą → grafikus → formulės-kodo patikrą. Visos atsitiktinumo sėklos
fiksuotos `src/config.py` (`RANDOM_SEED = 42`).

**Trukmė ir resursai:** paleista CPU-only aplinkoje (be GPU — kaip ir
numatytas galutinis naudojimo scenarijus, žr. kolokviumo 1 punkto
apribojimus). Faktiškai išmatuotas viso `run_all.py` paleidimo laikas —
apie 13–15 minučių, iš kurių didžiausią dalį (~8,5 min) sudaro SVM
mokymas su `probability=True` (Platt skalinimui). Atskirų žingsnių
išmatuotos trukmės — `results/train_log.json` ir `results/*_run.log`
failuose.

## Katalogo struktūra

```
src/
  config.py          - fiksuoti nustatymai (RANDOM_SEED, imties dydžiai)
  data.py            - duomenų nuskaitymas iš idx-ubyte failų
  preprocess.py       - train/val skaidymas, normalizavimas, HOG požymiai
  train.py            - kNN/SVM/MLP mokymas su hiperparametrų paieška validacijoje
  evaluate.py          - vertinimas test imtimi (tik kartą), metrikos, modelio dydis/greitis
  ablation.py          - raw pikseliai vs HOG požymiai (SVM)
  robustness.py        - atsparumas triukšmui ir daliniam uždengimui
  visualize.py          - visi grafikai
  verify_formula.py     - formulės (kolokviumo 5 punktas) ir kodo sutapimo patikra
run_all.py             - viena komanda visai grandinei paleisti
requirements.txt
results/
  models/*.joblib      - apmokyti modeliai
  tables/*.csv, *.json - skaitiniai rezultatai
  figures/*.png        - grafikai
  train_log.json, *_run.log - vykdymo žurnalai (laikai, hiperparametrai)
FINAL_REPORT.md         - rezultatų suvestinė + kiekvieno egzamino punkto atitikimas
AI_USAGE_LOG.md          - AI naudojimo žurnalas (egzamino reikalavimas)
```

## Gyvo gynimo dalis

`run_all.py` neapima gyvo gynimo (nematyto dėstytojo bandymo paleidimo,
nedidelio pakeitimo demonstravimo) — tai reikalauja studento asmeninio
dalyvavimo ir negali būti atlikta iš anksto. Žr. `FINAL_REPORT.md`
paskutinį skyrių.
