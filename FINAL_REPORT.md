# Galutinio egzamino dalis — rezultatų suvestinė

PEPfm-26, Ernestas Krikščiūnas. Fashion-MNIST drabužių klasifikavimas.
Visi skaičiai šiame faile yra realaus kodo paleidimo rezultatai (žr.
`results/` katalogą ir `AI_USAGE_LOG.md` skyrių 5), ne įverčiai.

## Atitikimas egzamino reikalavimams

### 1. „Įgyvendinti duomenų paruošimo grandinę, baseline ir bent du intelektualiuosius metodus. Bent vienam modeliui parodyti formulės ar taisyklės ryšį su konkrečia kodo vieta."

Įvykdyta. Duomenų grandinė: `src/data.py` (atsisiuntimas/nuskaitymas) →
`src/preprocess.py` (stratifikuotas train/val skaidymas, normalizavimas,
HOG). Baseline — kNN (`src/train.py`, `search_knn_k`). Du intelektualieji
metodai — SVM (RBF) ir MLP (256-128-100), abu `src/train.py`.
Formulės-kodo ryšys parodytas **du kartus**: (a) `src/train.py`
komentaruose prie `SVC(..., probability=True)` paaiškinta, kaip Platt
skalinimas atitinka kolokviumo 5 punkto formulę; (b) `src/verify_formula.py`
**skaičiais** patikrina, kad rankinis f(x) = sign(Σ αᵢyᵢK(xᵢ,x) + b)
skaičiavimas sutampa su `sklearn` `decision_function()` iki 10⁻¹² tikslumo
(žr. `results/verify_formula_run.log`).

### 2. „Palyginti metodus tuo pačiu duomenų skaidymu ir tomis pačiomis metrikomis. Hiperparametrų parinkimui nenaudoti galutinio testo."

Įvykdyta. Visi trys modeliai mokomi ant to paties stratifikuoto 30 000
pavyzdžių poaibio (`config.MAIN_TRAIN_SIZE`), hiperparametrai (kNN k;
SVM C, γ) parinkti tik per validaciją (`src/train.py`), test imtis
(10 000, pilna) panaudota lygiai vieną kartą, `src/evaluate.py`.

**Rezultatų lentelė (test imtis, n=10 000):**

| Metodas | Test Macro-F1 | Test tikslumas | Modelio dydis | Inferencija/vaizdui |
|---|---|---|---|---|
| kNN baseline (k=5) | 0.8420 | 0.8427 | 89.95 MB | 17.25 ms |
| **SVM (RBF, C=10, γ=0.03)** | **0.8919** | **0.8920** | 81.98 MB | 5.75 ms |
| MLP (256-128-100) | 0.8795 | 0.8808 | 2.84 MB | 0.19 ms |

Grafikas: `results/figures/comparison_macro_f1.png`.

### 3. „Atlikti bent vieną abliaciją arba požymių jautrumo bandymą ir bent vieną atsparumo bandymą: triukšmui, trūkstamiems duomenims, laiko ar grupės poslinkiui."

Įvykdyta abu. **Abliacija** (`src/ablation.py`, kolokviumo 6 punkto planas):
požymių tipas raw pikseliai vs. HOG, SVM, tas pats mokymo poaibis ir
hiperparametrai (C=10, γ='scale'):

| Požymiai | Test Macro-F1 |
|---|---|
| raw pikseliai | 0.8867 |
| HOG | **0.9015** |

HOG duoda +1,48 p.p. — kryptis atitinka literatūrą (Xiao ir kt., 2017,
HOG+SVM 92,6%), nors absoliuti reikšmė žemesnė, nes naudotas 30 000, o
ne 60 000 pavyzdžių poaibis ir hiperparametrai nebuvo perrinkti specialiai
HOG požymiams (žr. `AI_USAGE_LOG.md`). Grafikas:
`results/figures/ablation_raw_vs_hog.png`.

**Atsparumo bandymas** (`src/robustness.py`): Gauso triukšmas (σ ∈
[0, 0.4]) ir dalinis uždengimas (patch ∈ [0%, 45% kraštinės) — abu
tiesiogiai operacionalizuoja kolokviumo 1 punkto „Neaiškumus" (šešėliai,
foninis triukšmas, dalinis uždengimas sandėlyje). **Svarbiausias
radinys:** esant stipriam triukšmui (σ ≥ 0,3), kNN baseline tampa
**atsparesnis** už SVM ir MLP — SVM Macro-F1 nukrenta nuo 0,895 (σ=0)
iki 0,271 (σ=0,4), o kNN nukrenta tik iki 0,748. Tai empyriškai
**patvirtina** kolokviumo 4 punkte iš anksto suformuluotą hipotezės
sąlygą: „SVM pranašumas prieš kNN gali sumažėti arba net apsiversti...
esant duomenų pasiskirstymo skirtumui". Grafikai:
`results/figures/robustness_noise.png`, `robustness_occlusion.png`.

### 4. „Pateikti rezultatų lentelę, tinkamą grafiką, klaidų pavyzdžius ir paaiškinti, kodėl pasirinktas metodas laimėjo arba nelaimėjo. Neigiamas rezultatas priimtinas, jei eksperimentas korektiškas."

Lentelė ir grafikai — žr. 2–3 skyrius aukščiau. Klaidų pavyzdžiai:
`results/figures/error_examples_svm.png` — 10 dažniausiai painiojamų
porų vaizdų. Painiavos matrica: `results/figures/confusion_matrix_svm.png`.

**Kodėl SVM laimėjo prieš baseline (bet ne visiškai pagal iš anksto
numatytą ribą):** SVM Macro-F1 pranašumas prieš kNN test imtyje —
**4,99 p.p.** (0,8919 vs 0,8420), t. y. praktiškai lygiai ties 4-ame
kolokviumo punkte pre-registruota 5 p.p. riba. Bootstrap pasikliautinasis
intervalas (2000 pakartojimų): **95% CI = [4,38; 5,64] p.p.**,
P(skirtumas ≥ 5 p.p.) = 0,486. Išvada: SVM patikimai (P(skirtumas>0)=1,0)
geresnis už baseline, bet duomenys **nepatvirtina vienareikšmiškai**, kad
pranašumas siekia būtent iš anksto numatytą 5 p.p. ribą — tai ribinis,
sąžiningai priimtinas rezultatas (nei aiškus H₁ patvirtinimas, nei
atmetimas), atitinkantis "Neigiamas rezultatas priimtinas, jei
eksperimentas korektiškas".

**Kodėl SVM laimėjo prieš MLP švariuose duomenyse, bet pralaimi po
uždengimu:** švariame test rinkinyje SVM (0,8919) > MLP (0,8795), bet
prie stipraus dalinio uždengimo (patch≥35%) MLP tampa tikslesnis už SVM
(0,7475 vs 0,7339 ties 35%) — abu jautrūs degradacijai panašiai, bet
kNN geriausiai išlieka atsparus abiem degradacijos tipams. Painiavos
matrica parodo, kad didžiausia SVM klaidų grupė — vizualiai panašios
viršutinės aprangos klasės (Shirt→T-shirt/top 119 klaidos,
T-shirt/top→Shirt 101, Shirt→Pullover 85, Coat↔Pullover ~80) — tiksliai
tai, kas buvo numatyta kolokviumo 1 ir 3 punktuose dar prieš
eksperimentą.

### 5. „Pateikti paleidimo instrukciją, priklausomybių sąrašą, fiksuotas atsitiktines sėklas ir vieną komandą pagrindiniam eksperimentui pakartoti. Duomenų failų kopijų teikti nereikia, jei yra atsisiuntimo instrukcija."

Įvykdyta — žr. `README.md`: `requirements.txt`, `python3 run_all.py`
(viena komanda), `RANDOM_SEED=42` visur (`src/config.py`), duomenų
atsisiuntimo instrukcija (automatinė per `run_all.py` arba rankinis
`git clone`).

### 6. „Pateikti AI naudojimo žurnalą: svarbiausios užklausos, priimti ir atmesti pasiūlymai, bent dvi aptiktos AI klaidos arba nepatikrintos prielaidos ir jų patikros būdas."

Įvykdyta — žr. `AI_USAGE_LOG.md` (trys dokumentuotos klaidos/prielaidos,
ne tik dvi: openml.org nepasiekiamumas, `probability=True` skaičiavimo
kaina, hipotezės ribos nevienareikšmiškumas).

### 7. „Gyvo gynimo metu paaiškinti pasirinktą formulę bei kodą, paleisti dėstytojo pateiktą nematytą bandymą ir atlikti nedidelį modelio, metrikos ar duomenų apdorojimo pakeitimą."

**Neįvykdyta iš anksto ir negali būti** — tai reikalauja studento
asmeninio, gyvo dalyvavimo gynimo metu (nematytas dėstytojo bandymas iš
principo negali būti žinomas ar paruoštas iš anksto). Paruošta, kas
priklauso nuo pasiruošimo: (a) formulės ir kodo ryšys jau parodytas ir
patikrintas (žr. 1 skyrių) — pasiruošti jį paaiškinti gyvai; (b) kodo
struktūra moduliška (`src/*.py`), tad nedidelis pakeitimas (pvz., kito
k kNN, kito C SVM, naujo atsparumo lygio) turėtų būti atliekamas
konkrečiame faile be viso pipeline'o perrašymo — verta iš anksto
pasitreniruoti tokį pakeitimą atlikti ir paleisti lokaliai prieš
gynimą.

## Sprendimo ribos ir praktinis tinkamumas (susieta su kolokviumo 1 punkto apribojimais)

Trys modeliai atskleidžia skirtingus kompromisus, aktualius sandėlio be
GPU scenarijui: **MLP** turi mažiausią modelį (2,84 MB) ir greičiausią
inferenciją (0,19 ms/vaizdui) — praktiškiausias, jei svarbiausia
resursų taupymas. **SVM** turi geriausią tikslumą švariomis sąlygomis,
bet didžiausią modelį ir lėčiausią mokymą (~6,5 min su Platt
skalinimu), ir jo pranašumas **išnyksta arba apsiverčia** esant stipriam
triukšmui — rizikinga pasirinkti vien pagal švarų test rezultatą, kaip
ir buvo įspėta kolokviumo 4 punkto hipotezės sąlygoje. **kNN**, nors ir
prasčiausias švariomis sąlygomis, yra stabiliausias esant triukšmui —
tai nebuvo akivaizdu iš anksto ir yra vienas svarbiausių šio eksperimento
radinių praktiniam sprendimui (žr. `AI_USAGE_LOG.md`).
