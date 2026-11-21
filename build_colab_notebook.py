"""Build the Colab notebooks: the confirmatory experiment and the post hoc exploratory analyses."""

from __future__ import annotations

import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "colab_experiment.ipynb"
OUT_EXPLORATORY = ROOT / "colab_exploratory.ipynb"

RELATIVE = ["config.yaml", "requirements.txt", "preregistration.md", "exploratory_addendum.md", "pyproject.toml"]
RELATIVE += [path.relative_to(ROOT).as_posix() for path in sorted((ROOT / "commissioning").glob("*.py"))]

PAYLOAD = {
    rel: base64.b64encode((ROOT / rel).read_bytes()).decode("ascii")
    for rel in RELATIVE
}


def cell_markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(source)}


def cell_code(source: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": _lines(source),
    }


def _lines(source: str) -> list[str]:
    rows = source.split("\n")
    return [row + "\n" for row in rows[:-1]] + ([rows[-1]] if rows[-1] else [])


UNPACK = '''# KOMÓRKA 2/10 — odtworzenie plików projektu na maszynie Colaba
# Ta komórka zapisuje pakiet commissioning, config.yaml i preregistration.md.
import base64
from pathlib import Path

PROJECT = Path("/content/commissioning_project")
FILES = ''' + json.dumps(PAYLOAD) + '''

for relative, encoded in FILES.items():
    destination = PROJECT / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(base64.b64decode(encoded))
print(f"Zapisano {len(FILES)} plików do {PROJECT}")
'''

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
        "colab": {"name": "colab_experiment.ipynb", "provenance": []},
    },
    "cells": [
        cell_markdown(
            """# Eksperyment: ile dobrych sztuk przed włączeniem odrzutu

Otwórz **ten jeden plik** w Colabie: *File → Upload notebook*. Komórki są już podzielone. Każda zaczyna się od komentarza `# KOMÓRKA n/10`. Nie łącz ich z powrotem w jedną.

Przed uruchomieniem:

1. *Runtime → Change runtime type → GPU*.
2. Ustaw zmienną `DATA_ROOT` na katalog MVTec AD (w środku foldery `bottle`, `cable`, … z `train/good` i `test`).
3. Edytuj tylko komórkę 4. Resztę uruchamiaj po kolei.

Wyniki trafiają do `COMMISSIONING_WORK` (domyślnie `commissioning_runs`). Po przerwaniu sesji uruchom komórki 1–4 ponownie, a potem tę, na której stanęło: gotowe pliki są pomijane.

`PILOT = True` liczy tylko klasę transistor i oznacza przebieg jako exploratory. Właściwy eksperyment do artykułu to `PILOT = False`."""
        ),
        cell_code(
            """# KOMÓRKA 1/10 — montowanie Dysku Google
from google.colab import drive
drive.mount("/content/drive")"""
        ),
        cell_code(UNPACK),
        cell_code(
            """# KOMÓRKA 3/10 — biblioteki
# Kolejne komórki uruchamiają eksperyment w osobnym procesie, więc widzą wersje zainstalowane tutaj.
%pip install -q "scipy>=1.11" "pandas>=2.1" "pyarrow>=14" "scikit-learn>=1.3" "matplotlib>=3.8" "PyYAML>=6" "Pillow>=10" "tqdm>=4.66" "timm>=0.9" "anomalib>=2.0"
import subprocess, sys
print(subprocess.check_output(
    [sys.executable, "-c", "import torch, anomalib; print('torch', torch.__version__, '| anomalib', anomalib.__version__, '| GPU', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'BRAK')"],
    text=True,
))
# Jeśli wypisało GPU BRAK: Runtime -> Change runtime type -> GPU, potem od komórki 1."""
        ),
        cell_code(
            """# KOMÓRKA 4/10 — JEDYNA EDYCJA: ścieżki i tryb
import os

# False = 15 klas, bramka E0, przebieg do artykułu.
# True  = sam transistor, bez E0, wynik exploratory.
PILOT = False

DATA = os.environ["DATA_ROOT"]
WORK = os.environ.get("COMMISSIONING_WORK", "commissioning_runs")

os.environ["COMMISSIONING_DATA"] = DATA
os.environ["COMMISSIONING_WORK"] = WORK
os.chdir("/content/commissioning_project")
import sys
if "/content/commissioning_project" not in sys.path:
    sys.path.insert(0, "/content/commissioning_project")
if not os.path.isdir(os.path.join(DATA, "transistor", "train", "good")):
    raise FileNotFoundError(f"Nie widzę {DATA}/transistor/train/good. Popraw DATA.")

from commissioning.config import load_config
cfg = load_config()
print("hash", cfg.hash)
print("dane", cfg.data_root)
print("wyniki", cfg.work_root)
print("pilot", PILOT)"""
        ),
        cell_code(
            """# KOMÓRKA 5/10 — podziały C/P i kontrola liczności
import subprocess, sys
command = [sys.executable, "-m", "commissioning", "prepare"]
if PILOT:
    command += ["--classes", "transistor"]
subprocess.check_call(command)"""
        ),
        cell_code(
            """# KOMÓRKA 6/10 — cechy Wide-ResNet-50-2, raz na klasę
import subprocess, sys
command = [sys.executable, "-m", "commissioning", "features"]
if PILOT:
    command += ["--classes", "transistor"]
subprocess.check_call(command)"""
        ),
        cell_code(
            """# KOMÓRKA 7/10 — tensor odległości, raz na klasę
import subprocess, sys
command = [sys.executable, "-m", "commissioning", "distances"]
if PILOT:
    command += ["--classes", "transistor"]
subprocess.check_call(command)"""
        ),
        cell_code(
            """# KOMÓRKA 8/10 — bramka E0
# Przy PILOT = True ta komórka nic nie liczy. Pełny przebieg bez PASS stąd nie przejdzie do komórki 9.
import subprocess, sys
if PILOT:
    print("Pilot: pomijam E0. Komórka 9 oznaczy trajektorie jako exploratory.")
else:
    subprocess.check_call([sys.executable, "-m", "commissioning", "e0"])"""
        ),
        cell_code(
            """# KOMÓRKA 9/10 — trajektorie RQ1–RQ4
import subprocess, sys
command = [sys.executable, "-m", "commissioning", "run"]
if PILOT:
    command += ["--classes", "transistor", "--exploratory"]
subprocess.check_call(command)"""
        ),
        cell_code(
            """# KOMÓRKA 10/10 — statystyka i raport HTML
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "commissioning", "analyze"])
subprocess.check_call([sys.executable, "-m", "commissioning", "report"])
print("Raport:", WORK + "/report/index.html")"""
        ),
    ],
}

CLASSES = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut", "pill", "screw",
           "tile", "toothbrush", "transistor", "wood", "zipper"]

exploratory_notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"name": "colab_exploratory.ipynb", "provenance": []},
    },
    "cells": [
        cell_markdown(
            """# Analizy eksploracyjne (post hoc) do artykułu

Ten notatnik **nie liczy cech i nie potrzebuje GPU**. Korzysta z plików, które pełny przebieg zostawił w `COMMISSIONING_WORK`: `cache/dist`, `splits`, `runs/scores` i `analysis`. Wyników potwierdzających nie zmienia. Zapisuje katalog `exploratory` w tym samym miejscu i odświeża rysunki w `report`.

Uruchamiaj komórki po kolei. Edytujesz tylko komórkę 4. Opis analiz jest w `exploratory_addendum.md`. Po przerwaniu sesji uruchom komórki 1–4 i potem 5: klasy już policzone są pomijane."""
        ),
        cell_code(
            """# KOMÓRKA 1/6 — montowanie Dysku Google
from google.colab import drive
drive.mount("/content/drive")"""
        ),
        cell_code(UNPACK.replace("KOMÓRKA 2/10", "KOMÓRKA 2/6")),
        cell_code(
            """# KOMÓRKA 3/6 — biblioteki i (opcjonalnie) obrazy MVTec do rysunków
# Obrazy są potrzebne tylko do rysunku odrzucanych dobrych sztuk i do miniatur w raporcie.
%pip install -q "scipy>=1.11" "pandas>=2.1" "pyarrow>=14" "scikit-learn>=1.3" "matplotlib>=3.8" "PyYAML>=6" "Pillow>=10"
import os
import zipfile
from pathlib import Path

drive_dir = Path(os.environ["DATA_ROOT"])
local_data_dir = Path("/content/mvtec_anomaly_detection")
local_data_dir.mkdir(parents=True, exist_ok=True)
classes = """ + json.dumps(CLASSES) + """
for cls in classes:
    zip_path = drive_dir / f"{cls}.zip"
    extract_path = local_data_dir / cls
    if zip_path.exists() and not extract_path.exists():
        print(f"Rozpakowuje {zip_path.name}...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_path)
    elif not zip_path.exists():
        print(f"Brak pliku w DATA_ROOT: {zip_path.name} (rysunki z obrazami tej klasy zostaną pominięte)")
print("Gotowe!")"""
        ),
        cell_code(
            """# KOMÓRKA 4/6 — JEDYNA EDYCJA: ścieżki
import os, sys

DATA = os.environ.get("DATA_ROOT", "/content/mvtec_anomaly_detection")
WORK = os.environ.get("COMMISSIONING_WORK", "commissioning_runs")

os.environ["COMMISSIONING_DATA"] = DATA
os.environ["COMMISSIONING_WORK"] = WORK
os.chdir("/content/commissioning_project")
if "/content/commissioning_project" not in sys.path:
    sys.path.insert(0, "/content/commissioning_project")
for needed in ("cache/dist", "splits", "runs/scores", "analysis/metrics.parquet"):
    if not os.path.exists(os.path.join(WORK, needed)):
        raise FileNotFoundError(f"Nie widzę {WORK}/{needed}. To musi być katalog z pełnego przebiegu.")

from commissioning.config import load_config
cfg = load_config()
print("hash", cfg.hash, "(musi być ec70992336b4c4a4)")
assert cfg.hash == "ec70992336b4c4a4", "Inny hash konfiguracji niż w przebiegu potwierdzającym."
print("dane", cfg.data_root, "| obrazy:", os.path.isdir(os.path.join(DATA, "transistor", "train", "good")))
print("wyniki", cfg.work_root)"""
        ),
        cell_code(
            """# KOMÓRKA 5/6 — analizy E1–E5 (ok. 15–30 min, głównie odczyt cache/dist z COMMISSIONING_WORK)
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "commissioning", "explore"])
print("Raport eksploracyjny:", WORK + "/exploratory/index.html")"""
        ),
        cell_code(
            """# KOMÓRKA 6/6 — raport główny z poprawionymi rysunkami (liczby bez zmian)
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "commissioning", "report"])
print("Raport główny:", WORK + "/report/index.html")"""
        ),
    ],
}

OUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"wrote {OUT} ({OUT.stat().st_size} bytes, {len(RELATIVE)} embedded files)")
OUT_EXPLORATORY.write_text(json.dumps(exploratory_notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"wrote {OUT_EXPLORATORY} ({OUT_EXPLORATORY.stat().st_size} bytes)")
