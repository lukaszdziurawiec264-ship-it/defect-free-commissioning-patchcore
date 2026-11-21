"""Build colab_visa.ipynb: the confirmatory VisA replication of preregistration_visa.md."""

from __future__ import annotations

import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "colab_visa.ipynb"

RELATIVE = ["config.yaml", "config_visa.yaml", "requirements.txt", "preregistration.md", "preregistration_visa.md",
            "pyproject.toml"]
RELATIVE += [path.relative_to(ROOT).as_posix() for path in sorted((ROOT / "commissioning").glob("*.py"))]
RELATIVE += [path.relative_to(ROOT).as_posix() for path in sorted((ROOT / "commissioning" / "visa").glob("*.py"))]

PAYLOAD = {rel: base64.b64encode((ROOT / rel).read_bytes()).decode("ascii") for rel in RELATIVE}
TOTAL = 12


def cell_markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(source)}


def cell_code(source: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": _lines(source)}


def _lines(source: str) -> list[str]:
    rows = source.split("\n")
    return [row + "\n" for row in rows[:-1]] + ([rows[-1]] if rows[-1] else [])


def stage(number: int, title: str, command: str, note: str = "") -> dict:
    prefix = f"# KOMÓRKA {number}/{TOTAL} — {title}\n" + (f"# {note}\n" if note else "")
    return cell_code(
        prefix
        + "import subprocess, sys\n"
        + f"subprocess.check_call([sys.executable, '-m', 'commissioning.visa', {command}])"
    )


UNPACK = f'''# KOMÓRKA 2/{TOTAL} — odtworzenie plików projektu na maszynie Colaba
import base64
from pathlib import Path

PROJECT = Path("/content/commissioning_project")
FILES = ''' + json.dumps(PAYLOAD) + '''

for relative, encoded in FILES.items():
    destination = PROJECT / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(base64.b64decode(encoded))
status = next(line for line in (PROJECT / "preregistration_visa.md").read_text(encoding="utf-8").splitlines() if line.startswith("Status:"))
print(f"Zapisano {len(FILES)} plików do {PROJECT}")
print("Preregistracja:", status)
'''

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
        "colab": {"name": "colab_visa.ipynb", "provenance": []},
    },
    "cells": [
        cell_markdown(
            f"""# Replikacja potwierdzająca na VisA (preregistration_visa.md)

Otwórz ten plik w Colabie: *File → Upload notebook*. Każda komórka zaczyna się od `# KOMÓRKA n/{TOTAL}`.

Przed uruchomieniem:

1. *Runtime → Change runtime type → GPU* (T4 wystarczy, L4/A100 szybciej).
2. Pobierz `VisA_20220922.tar` (ok. 1,8 GB) ze strony zbioru i ustaw `VISA_TAR` na ścieżkę do tego archiwum. Oficjalna strona: https://github.com/amazon-science/spot-diff.
3. **Zamroź preregistrację przed komórką 9.** Komórki 1–8 (konwersja, podziały, cechy) wolno uruchomić wcześniej. Odległości, E0 i trajektorie nie wystartują, dopóki w `preregistration_visa.md` nie ma linii `Status: **FROZEN** on <data>`. Plik jest wbudowany w ten notatnik, więc po zamrożeniu zbuduj notatnik ponownie (`python build_colab_visa_notebook.py`). Przy pierwszym liczeniu wyników jego SHA-256 jest zapisywany w VISA_WORK; późniejsza zmiana pliku zatrzyma przebieg.
4. Edytuj tylko komórkę 4.

Wyniki (podziały, E0, trajektorie, analiza, raport) trafiają do `VISA_WORK` (domyślnie `visa_runs`). Duże pliki (obrazy, cechy ok. 32 GB, odległości ok. 12 GB) są na lokalnym dysku maszyny i znikają po zakończeniu sesji. Po przerwaniu uruchom komórki 1–7 i potem tę, na której stanęło: gotowe pliki są pomijane, a brakujące cechy i odległości liczą się od nowa (kilka minut na klasę)."""
        ),
        cell_code(f"""# KOMÓRKA 1/{TOTAL} — montowanie Dysku Google
from google.colab import drive
drive.mount("/content/drive")"""),
        cell_code(UNPACK),
        cell_code(
            f"""# KOMÓRKA 3/{TOTAL} — biblioteki
%pip install -q "scipy>=1.11" "pandas>=2.1" "pyarrow>=14" "scikit-learn>=1.3" "matplotlib>=3.8" "PyYAML>=6" "Pillow>=10" "tqdm>=4.66" "timm>=0.9" "anomalib>=2.0"
import subprocess, sys
print(subprocess.check_output(
    [sys.executable, "-c", "import torch, anomalib; print('torch', torch.__version__, '| anomalib', anomalib.__version__, '| GPU', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'BRAK')"],
    text=True,
))
# Jeśli wypisało GPU BRAK: Runtime -> Change runtime type -> GPU, potem od komórki 1."""
        ),
        cell_code(
            f"""# KOMÓRKA 4/{TOTAL} — JEDYNA EDYCJA: ścieżki
import os, shutil, sys

VISA_TAR = os.environ["VISA_TAR"]
WORK = os.environ.get("VISA_WORK", "visa_runs")
RAW = "/content/VisA_raw"                    # rozpakowane archiwum, lokalnie
DATA = "/content/visa_1cls"                  # układ MVTec, lokalnie
CACHE = "/content/visa_cache"                # cechy i odległości, lokalnie

os.environ["VISA_DATA"] = DATA
os.environ["VISA_WORK"] = WORK
os.environ["VISA_CACHE"] = CACHE
os.chdir("/content/commissioning_project")
if "/content/commissioning_project" not in sys.path:
    sys.path.insert(0, "/content/commissioning_project")
if not os.path.isfile(VISA_TAR):
    raise FileNotFoundError(f"Nie widzę {{VISA_TAR}}. Popraw VISA_TAR.")

from commissioning.visa.config import load_visa_config
cfg = load_visa_config()
print("hash VisA", cfg.hash)
print("wyniki", cfg.work_root, "| cache", cfg.cache_root)
print("wolne miejsce na dysku lokalnym: %.0f GB (potrzeba ok. 50 GB)" % (shutil.disk_usage("/content").free / 1024**3))"""
        ),
        cell_code(
            f"""# KOMÓRKA 5/{TOTAL} — rozpakowanie archiwum VisA na dysk lokalny (ok. 2–4 min)
import os, tarfile
if not os.path.isdir(os.path.join(RAW, "candle")):
    os.makedirs(RAW, exist_ok=True)
    with tarfile.open(VISA_TAR) as archive:
        archive.extractall(RAW)
# Archiwum może mieć katalog nadrzędny VisA/; konwersja potrzebuje folderu z candle/ i split_csv/.
root = RAW if os.path.isdir(os.path.join(RAW, "candle")) else os.path.join(RAW, os.listdir(RAW)[0])
print("katalog VisA:", root, "|", sorted(os.listdir(root))[:6])
RAW_ROOT = root"""
        ),
        cell_code(
            f"""# KOMÓRKA 6/{TOTAL} — konwersja do układu MVTec według split_csv/1cls.csv (kopiowanie, bez zmiany plików)
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "commissioning.visa", "convert", "--raw-root", RAW_ROOT])"""
        ),
        stage(7, "kontrola liczności i duplikatów, pula U, zbiór H, podziały C/P", "'prepare'",
              "Nie liczy żadnych wyników. Zapisuje data_check.json i splits/ w VISA_WORK."),
        stage(8, "cechy Wide-ResNet-50-2, raz na klasę (GPU, ok. 1–2 min na klasę)", "'features'",
              "Wolno uruchomić przed zamrożeniem preregistracji."),
        stage(9, "tensor odległości, raz na klasę (GPU, kilka minut na klasę)", "'distances'",
              "Od tej komórki preregistracja musi być zamrożona."),
        stage(10, "bramka E0 (Anomalib na U)", "'e0'", "Bez PASS komórka 11 nie wystartuje."),
        cell_code(
            f"""# KOMÓRKA 11/{TOTAL} — trajektorie, wymienialność, podział budżetu
import subprocess, sys
for stage_name in ("run", "exchange", "budget"):
    subprocess.check_call([sys.executable, "-m", "commissioning.visa", stage_name])
# Opcjonalnie, żeby zwolnić miejsce po E0: shutil.rmtree(CACHE + "/features")"""
        ),
        cell_code(
            f"""# KOMÓRKA 12/{TOTAL} — statystyka i raport HTML
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "commissioning.visa", "analyze"])
subprocess.check_call([sys.executable, "-m", "commissioning.visa", "report"])
print("Raport:", WORK + "/report/index.html")"""
        ),
    ],
}

OUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"wrote {OUT} ({OUT.stat().st_size} bytes, {len(RELATIVE)} embedded files)")
