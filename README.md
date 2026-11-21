# Defect-free calibration of a PatchCore-based inspection decision

Code, frozen protocol notes, and the small result tables behind a manuscript on commissioning a memory-bank anomaly detector when no defective images are available. The detector is the PatchCore memory bank (WideResNet-50, layers 2 and 3). The decision threshold is set from defect-free calibration images only. MVTec AD is the main study. VisA is the second dataset.

This repository does not contain the manuscript, the image datasets, feature caches, or distance tensors.

The files named preregistration*.md are internal analysis protocols fixed before the corresponding main runs; they were not registered with a third party.

## Requirements

- Python 3.10 or newer.
- `requirements.txt` pins the local test environment used to prepare this repository (Python 3.14). That environment did not produce the MVTec AD or VisA results.
- Versions copied from the stored `environment.json` files, without paths, are in `environment_used.txt`. `torchvision` is not recorded in either file. The MVTec file records `torch` `2.14.1+cpu`, CUDA off, and `anomalib` null. The VisA file records the GPU runtime: Python 3.13.16, `torch` `2.14.1+cu130`, `timm` `1.0.31`, `anomalib` `2.7.1`.
- `torch`, `torchvision`, `timm`, and `anomalib` in `requirements.txt` are lower bounds only. They were not installed in the local test environment. Feature extraction and the reference PatchCore check need a CUDA GPU.
- Tables and figures can be rebuilt from `results/` without a GPU.

## Installation

```bash
python -m venv .venv
```

Windows:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS or Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

`pyproject.toml` still declares lower bounds for a package install (`pip install -e .`). Use `requirements.txt` when you want the pins above.

## Data

Download the datasets from the publishers. This repository does not redistribute them.

- MVTec AD: https://www.mvtec.com/company/research/datasets/mvtec-ad
- VisA: https://github.com/amazon-science/spot-diff

The MVTec layout expected by `python -m commissioning prepare` is the official class folders (`train/good`, `test/good`, `test/<defect>`). The VisA layout expected after `python -m commissioning.visa convert` is one folder per class with `train/good`, `test/good`, and `test/bad`.

## Paths

Paths are not part of the configuration hash. Scientific settings stay in `config.yaml` and `config_visa.yaml`.

After the main runs, the only change in the calculation package is path handling. `DATA_ROOT` selects the MVTec AD directory, and `COMMISSIONING_DATA` is still accepted. This does not affect the calculations. The configuration hashes remain `ec70992336b4c4a4` (MVTec AD) and `6177f9b75e4f7e32` (VisA). The package copy embedded in the notebooks is older than that change and sets paths through `COMMISSIONING_DATA`.

MVTec AD:

| Variable | Role | Default in `config.yaml` |
| --- | --- | --- |
| `DATA_ROOT` | extracted MVTec AD root | `data/mvtec` |
| `COMMISSIONING_DATA` | older name for the same path; used only if `DATA_ROOT` is unset | |
| `COMMISSIONING_WORK` | run directory | `.` |

`DATA_ROOT` does not change the VisA paths.

VisA:

| Variable | Role | Default in `config_visa.yaml` |
| --- | --- | --- |
| `VISA_DATA` | converted one-class VisA root | `data/visa_1cls` |
| `VISA_WORK` | run directory | `visa_runs` |
| `VISA_CACHE` | features and distances | `visa_cache` |
| `VISA_TAR` | path to `VisA_20220922.tar`, read by the Colab notebook only | none |

Command-line `--data-root`, `--work-root`, and `--cache-root` override the variables.

PowerShell:

```powershell
$env:DATA_ROOT = "D:\data\mvtec"
$env:COMMISSIONING_WORK = ".\work\mvtec"
$env:VISA_DATA = "D:\data\visa_1cls"
$env:VISA_WORK = ".\work\visa"
$env:VISA_CACHE = ".\work\visa_cache"
$env:VISA_TAR = "D:\data\VisA_20220922.tar"
```

bash:

```bash
export DATA_ROOT=/data/mvtec
export COMMISSIONING_WORK=./work/mvtec
export VISA_DATA=/data/visa_1cls
export VISA_WORK=./work/visa
export VISA_CACHE=./work/visa_cache
export VISA_TAR=/data/VisA_20220922.tar
```

## Run order

From the repository root.

MVTec AD, in this order (`all` runs every stage and stops if the E0 gate fails):

```bash
python -m commissioning prepare
python -m commissioning features
python -m commissioning distances
python -m commissioning e0
python -m commissioning run
python -m commissioning analyze
python -m commissioning report
```

`explore` is separate. It is the post hoc analysis and is not part of `all`:

```bash
python -m commissioning explore
```

VisA:

```bash
python -m commissioning.visa convert --raw-root /path/to/extracted/VisA
python -m commissioning.visa prepare
python -m commissioning.visa features
python -m commissioning.visa distances
python -m commissioning.visa e0
python -m commissioning.visa run
python -m commissioning.visa exchange
python -m commissioning.visa budget
python -m commissioning.visa analyze
python -m commissioning.visa report
```

Scoring stages refuse to start unless `preregistration_visa.md` contains the status line that the code checks. The Colab notebooks `colab_experiment.ipynb`, `colab_exploratory.ipynb`, `colab_visa.ipynb`, and `notebooks/colab_driver.ipynb` call the same stages. Set the variables above before the path cell. `drive.mount("/content/drive")` only mounts Colab storage; it is not a data path. The package copy embedded in those notebooks is older than the `DATA_ROOT` change and sets paths through `COMMISSIONING_DATA`.

Tests that do not need the datasets:

```bash
pytest
```

## Tables and figures

`python -m commissioning analyze` writes the MVTec AD tables. The copies used for the manuscript are in `results/mvtec/analysis/`:

| File | Contents |
| --- | --- |
| `summary.csv` | per-class area under the TPR and FPR curves |
| `fpr_aggregate.csv` | false-reject rate by memory-bank size |
| `stopping.csv` | sequential stopping sizes |
| `false_reject_budget.csv` | per-class false-reject budgets |
| `metrics.parquet` | one row per class, repeat, strategy, and bank size |
| `statistics.json` | paired strategy comparisons |

`python -m commissioning explore` writes the post hoc tables now in `results/mvtec/exploratory/`: `budget_best.csv`, `budget_summary.csv`, `calibration_size.csv`, `ceiling.csv`, `ceiling_base.csv`, `censoring.csv`, `exchangeability.csv`, `rejected_good.csv`.

`python -m commissioning.visa analyze` writes the VisA tables now in `results/visa/analysis/`: `summary.csv`, `metrics.parquet`, `statistics.json`, `n_star.csv`, `stopping.csv`, `stopping_summary.csv`, `ceiling.csv`, `p2_calibration.csv`, `p3_shift.csv`, `budget_best.csv`, `budget_summary.csv`. The VisA splits are in `results/visa/splits/`, the exchangeability records in `results/visa/exchange/`, and the data check in `results/visa/data_check.json`.

`python manuscript/make_figures.py` reads `results/mvtec/` and `results/visa/` and writes `manuscript/figures/`. It does not refit the detector.

| Figure file | Source |
| --- | --- |
| `fig1_protocol` | diagram, no result file |
| `fig2_mvtec_curves` | `results/mvtec/analysis/metrics.parquet` |
| `fig3_mvtec_shift` | `results/mvtec/exploratory/exchangeability.csv` |
| `fig4_visa_fpr` | `results/visa/analysis/metrics.parquet` |
| `fig5_visa_cost` | `results/visa/analysis/statistics.json` |
| `fig6_kcenter` | both `analysis/statistics.json` files |
| `figA1_ceiling` | closed-form calculation, no result file |

## Layout

```text
commissioning/            MVTec AD package
commissioning/visa/       VisA package
tests/                    local checks
config.yaml               MVTec settings
config_visa.yaml          VisA settings
preregistration.md        MVTec protocol text
preregistration_visa.md   VisA protocol text
exploratory_addendum.md   post hoc analyses
environment_used.txt      versions recorded for the runs
manuscript/make_figures.py
colab_*.ipynb             notebooks that call the stages
results/mvtec/            MVTec tables and the E0 gate
results/visa/             VisA tables, splits, exchange records, and the E0 gate
```

Feature caches, distance tensors, image archives, reports, and the manuscript are ignored.

## Citation

[DO UZUPEŁNIENIA po publikacji]
