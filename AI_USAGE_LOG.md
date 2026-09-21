# AI naudojimo žurnalas

Šis žurnalas apima visą darbo su AI (Claude) eigą — nuo kolokviumo plano
rengimo iki galutinio kodo įgyvendinimo. Tai tiesioginis kolokviumo 7
punkte pažadėto plano įvykdymas.

## 1. Svarbiausios užklausos

**Kolokviumo plano rengimo etapas** (7 punktai, kiekvienas atskirai):
problemos/apribojimų/klaidų kainos formulavimas, sprendimo etapų
išskaidymas su įvestimi/išvestimi, 2–4 metodų ir baseline parinkimas su
literatūros pagrindimu (DOI patikra), pagrindinio metodo ir patikrinamos
hipotezės formulavimas, SVM formulių paaiškinimas ir susiejimas su
programos moduliais, duomenų skaidymo/metrikų/abliacijos/rizikų planas,
AI naudojimo planas. Kiekvienam punktui — atskira užklausa, po kurios
sekė keli tikslinimo klausimai (žr. skyrių 2–3 žemiau).

**Galutinio egzamino įgyvendinimo etapas**: „atlikti egzamino dalį pagal
reikalavimus ir pakomentuoti, kurį punktą kas įvykdo" — vienas bendras
prašymas, kurį AI išskaidė į duomenų gavimą, preprocessing, modelių
mokymą, vertinimą, abliaciją, atsparumo bandymą, vizualizacijas,
formulės-kodo patikrą ir dokumentaciją (žr. šio katalogo TaskList
struktūrą, atkartotą FINAL_REPORT.md).

## 2. Priimti pasiūlymai

- Formuluotės tikslinimas: „atsitiktinumo sėkla" → „atsitiktinė reikšmė"
  (priimta hipotezės sakinyje); „glotni bei bendra" → „lygi bei paprasta"
  (priimta γ paaiškinime, nes sudarė aiškesnę priešpriešą su „vingiuota").
- Pagrindinio metodo pasirinkimas: SVM vietoj MLP/CNN, pagrįstas ne vien
  tikslumu, bet ir GPU nebuvimo apribojimu — priimta ir panaudota kaip
  4-o kolokviumo punkto pagrindas.
- Vienos hipotezės (be formalios H₀/H₁ poros) forma — priimta po
  diskusijos, nes formali statistinė pora be konkretaus numatyto testo
  būtų buvusi papuošimas, o ne turinys.
- Bootstrap pasikliautinojo intervalo naudojimas hipotezei patikrinti
  (AI pasiūlyta kolokviumo 6 punkte kaip rizikos mažinimo priemonė,
  vėliau realiai įgyvendinta `evaluate.py` faile — žr. skyrių 4).

## 3. Atmesti arba nepilnai priimti pasiūlymai

- Siūlymas pakeisti „domeno poslinkis" į „duomenų pasiskirstymo
  skirtumas" 4-o punkto hipotezės sakinyje — NEPRIIMTAS toje konkrečioje
  vietoje (liko „domeno poslinkis"), nors ta pati AI siūlyta formuluotė
  vėliau savarankiškai panaudota 6-o punkto tekste. Sprendimas: studento
  teisė pasilikti trumpesnį variantą vienoje vietoje.
- Siūlymas pakeisti „ar rezultatai loginiai/logiškai teisingi" į „ar
  rezultatai atitinka pagrįstus lūkesčius" (7 punktas, kodo tikrinimas) —
  pasiūlymas pateiktas, bet dokumente liko originali formuluotė.
  Vertinant faktinį elgesį programoje (`evaluate.py`, `verify_formula.py`),
  realiai tikrinama būtent tai, ką siūlyta formuluotė apibūdino (reikšmių
  rėžiai, modelių santykis, sutapimas su rankiniu skaičiavimu), tad
  neatitikimas liko tik teksto, ne veiksmo, lygmenyje.
- „Platt skalinimas" vs „Platt scaling" — AI patvirtino, kad lietuviškas
  terminas „skalinimas" yra tinkamas ir įprastas, bet studentas dokumente
  pasiliko anglišką „Platt scaling" formą. Kodo komentaruose (`train.py`)
  naudojamas lietuviškas „Platt skalinimas".

## 4. Aptiktos AI klaidos ir nepatikrintos prielaidos

**(1) Neteisinga prielaida apie duomenų šaltinio pasiekiamumą.**
AI pirmiausia bandė duomenis atsisiųsti per
`sklearn.datasets.fetch_openml(data_id=40996)` — tiesiogiai tą patį
šaltinį, kurį nurodo užduoties nuoroda (openml.org/d/40996). Tai
nepavyko: `api.openml.org` šioje aplinkoje užblokuotas tinklo politikos
(403 Forbidden per proxy). **Patikros būdas:** paleista keletas `curl`
testų kitiems potencialiems šaltiniams (huggingface.co,
storage.googleapis.com, raw.githubusercontent.com — visi taip pat
užblokuoti), kol nustatyta, kad `git clone` per `github.com` veikia.
Duomenys atsisiųsti iš oficialaus `zalandoresearch/fashion-mnist`
repo, o po atsisiuntimo patikrinta, kad gautų masyvų forma (60000×784 /
10000×784) ir klasių balansas (6000/1000 kiekvienai klasei) tiksliai
atitinka tai, kas buvo aprašyta kolokviumo plane — t. y. tas pats
duomenų rinkinys, tik kitas atsisiuntimo kelias.

**(2) Klaidinga prielaida apie `probability=True` skaičiavimo kainą.**
Kolokviumo 6 punkto „Skaičiavimo biudžeto" skyriuje buvo numatyta, kad
SVM mokymas ant 30000 pavyzdžių užtruks apie tiek pat, kiek išmatuota be
Platt skalinimo (~45 s pagal ankstesnį bandymą su `probability=False`).
Realiai paleidus su `probability=True` (būtina pasitikėjimo balui gauti,
kaip aprašyta 5 punkte), mokymas užtruko **389 s — apie 8,6 karto
ilgiau**, nes sklearn viduje atlieka papildomą 5-fold kryžminį
kalibravimą. **Patikros būdas:** laikas išmatuotas tiesiogiai (`time.time()`
prieš/po `fit()`) ir užfiksuotas `results/train_log.json`; tai realus,
patikrintas (ne prielaida) skaičius, kuris pakoreguoja anksčiau plane
įrašytą laiko įvertį — pataisyta informacija įtraukta į šio README
„Trukmė ir resursai" skyrių.

**(3, papildomai) Iš anksto neišbandyta hipotezės riba.** Kolokviumo 4
punkte pre-registruota hipotezė teigė, kad SVM pranoks kNN baseline
bent 5 procentiniais punktais pagal Macro-F1. Faktinis išmatuotas
skirtumas test imtyje — 4,99 p.p. (SVM 0,8919 vs kNN 0,8420), t. y.
tiesiogiai ties riba. **Patikros būdas:** atliktas bootstrap
pasikliautinasis intervalas (2000 pakartojimų) — 95% CI = [0,0438;
0,0564], P(skirtumas ≥ 0,05) = 0,486. Išvada nėra nei aiškus
patvirtinimas, nei atmetimas — tai sąžiningai užfiksuota
`FINAL_REPORT.md` kaip ribinis, statistiškai nevienareikšmis rezultatas,
o ne pakoreguota post-hoc, kad "patvirtintų" hipotezę.

## 5. Bendra pastaba

Visas šio katalogo kodas buvo realiai paleistas šioje aplinkoje (ne tik
sugeneruotas) — kiekvieno modulio išvestis (laikai, tikslumo skaičiai,
grafikai) yra faktiniai vykdymo rezultatai, saugomi `results/` kataloge
ir `*_run.log` failuose, o ne AI numatyti ar apytiksliai skaičiai.
