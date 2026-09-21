"""Centralizuoti eksperimento nustatymai -- viena vieta, kurioje fiksuojamos
visos konstantos, kad jas būtų lengva pakeisti ir kad visi moduliai naudotų
tas pačias reikšmes (svarbu teisingam palyginimui tarp metodų)."""

import os

# Projekto šaknies katalogas skaičiuojamas absoliučiai nuo šio failo vietos
# (src/config.py -> tėvinis katalogas), o ne kaip santykinis "results/..."
# kelias. Taip išvengiama klaidų, kai skriptas paleidžiamas ne iš projekto
# šaknies arba Windows sistemoje, kur santykinio kelio su "/" ir vėliau
# pridėto OS separatoriaus maišymas ("results/figures\\failas.png") kai
# kuriose aplinkose sukelia OSError [Errno 22] Invalid argument.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RANDOM_SEED = 42

# Kadangi sandėlio kompiuteris (numatoma naudojimo aplinka, žr. kolokviumo
# 1 punktą) neturės GPU, o kernelinio SVM mokymo laikas auga daug greičiau
# nei tiesiškai su mokymo imties dydžiu (išmatuota eksperimentiškai žemiau),
# pagrindiniai modeliai mokomi ant fiksuoto, stratifikuoto 30000 pavyzdžių
# poaibio (iš 54000 galimų po validacijos atskyrimo), o ne ant viso mokymo
# rinkinio. Tai tiesiogiai atitinka kolokviumo 6 punkte iš anksto numatytą
# "skaičiavimo biudžeto" sprendimą: "hiperparametrų paiešką atlikti ant
# sumažintos imties, o galutinį modelį treniruoti su visa imtimi tik jei tai
# atlieka per priimtiną laiką; jei ne -- apsiriboti sumažintos imties
# modeliu". Išmatuoti laikai (šioje aplinkoje, be GPU):
#   SVM(RBF) fit: n=10000 -> 7.1s, n=20000 -> 22.2s, n=30000 -> 45.3s
# Ekstrapoliuojant, pilnas 54000 imties SVM mokymas kartu su hiperparametrų
# paieška (keli kandidatai) viršytų praktišką šios pristatymo aplinkos laiko
# biudžetą, tad pasirinktas 30000 kompromisas. Tai sąmoningas, dokumentuotas
# apribojimas, ne nutylėtas supaprastinimas -- žr. AI_USAGE_LOG.md ir
# FINAL_REPORT.md "Sprendimo ribos" skyrių.
MAIN_TRAIN_SIZE = 30000

# Hiperparametrų paieškai naudojamas dar mažesnis poaibis (kaip numatyta
# kolokviumo plane: "10 000-15 000 pavyzdžių").
HPARAM_SEARCH_SIZE = 10000

# Antrinių eksperimentų (abliacijos, atsparumo bandymo) test poaibio dydis --
# stratifikuotas, kad išlaikytų klasių balansą; naudojamas, nes šie
# eksperimentai kartoja prognozavimą su keliomis modelio/duomenų
# konfigūracijomis, o pilnas 10000 test vaizdų kiekvienai konfigūracijai
# viršytų laiko biudžetą su kNN/SVM.
SECONDARY_TEST_SIZE = 2000

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

RESULTS_DIR = os.path.join(_PROJECT_ROOT, "results")
MODELS_DIR = os.path.join(RESULTS_DIR, "models")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
