# Conformal Calibration for Multi-Modal Regression with Missing Modalities

[![Project page](https://img.shields.io/badge/Project_page-online-7C3AED?logo=data:image/svg%2Bxml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjZmZmZmZmIiBzdHJva2Utd2lkdGg9IjIuMiIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIj48cGF0aCBkPSJNMy41IDYuNWgxN00zLjUgMTJoMTdNMy41IDE3LjVoMTciLz48Y2lyY2xlIGN4PSI4IiBjeT0iNi41IiByPSIyLjUiIGZpbGw9IiNmZmZmZmYiIHN0cm9rZT0ibm9uZSIvPjxjaXJjbGUgY3g9IjE2IiBjeT0iMTIiIHI9IjIuNSIgZmlsbD0iI2ZmZmZmZiIgc3Ryb2tlPSJub25lIi8+PGNpcmNsZSBjeD0iMTAuNSIgY3k9IjE3LjUiIHI9IjIuNSIgZmlsbD0iI2ZmZmZmZiIgc3Ryb2tlPSJub25lIi8+PC9zdmc+)](https://unco3892.github.io/modality-aware-conformal/)
[![Paper](https://img.shields.io/badge/Paper-PDF-0b7285?logo=data:image/svg%2Bxml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjZmZmZmZmIiBzdHJva2Utd2lkdGg9IjIuMSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIj48cGF0aCBkPSJNMTQgMi44SDYuNEEyLjQgMi40IDAgMCAwIDQgNS4ydjEzLjZhMi40IDIuNCAwIDAgMCAyLjQgMi40aDExLjJhMi40IDIuNCAwIDAgMCAyLjQtMi40VjguOHoiLz48cGF0aCBkPSJNMTQgMi44VjguOGg2Ii8+PHBhdGggZD0iTTggMTMuNGg4TTggMTcuMmg1LjUiLz48L3N2Zz4=)](http://iliaazizi.com/publications/modality-aware-conformal/copa_2026.pdf)
[![Python](https://img.shields.io/badge/python-3.11-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-2ea44f?logo=opensourceinitiative&logoColor=white)](LICENSE)
<!-- [![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/XXXX.XXXXX) -->
<!-- [![COPA 2026](https://img.shields.io/badge/COPA_2026-PMLR_v329-0b7285?logo=data:image/svg%2Bxml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjZmZmZmZmIiBzdHJva2Utd2lkdGg9IjIuMSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIj48cGF0aCBkPSJNMTQgMi44SDYuNEEyLjQgMi40IDAgMCAwIDQgNS4ydjEzLjZhMi40IDIuNCAwIDAgMCAyLjQgMi40aDExLjJhMi40IDIuNCAwIDAgMCAyLjQtMi40VjguOHoiLz48cGF0aCBkPSJNMTQgMi44VjguOGg2Ii8+PHBhdGggZD0iTTggMTMuNGg4TTggMTcuMmg1LjUiLz48L3N2Zz4=)](https://proceedings.mlr.press/v329/) -->

**[Ilia Azizi](https://iliaazizi.com)**<sup>1,2</sup>

<sup>1</sup> Department of Operations, HEC Lausanne, University of Lausanne, Switzerland
<sup>2</sup> BegooAI, Switzerland

## Abstract

Prediction intervals for multi-modal regression with tabular variables, text,
images, or other input sources are difficult to calibrate when those sources
disagree or one is missing. A single global quantile averages these regimes
together instead of calibrating to the modality pattern observed at test time.
We address this through a modality-aware conformal calibration layer. The
layer trains or reuses one predictor per modality, computes a disagreement
score from their predictions, and uses that score in split conformal
calibration under a strict split protocol. We use the score in two
complementary ways. First, a continuous disagreement-scaled method reallocates
interval width across examples while preserving the usual marginal
split-conformal guarantee. Second, a Mondrian (stratified) method calibrates
within groups defined by disagreement or modality availability fixed before
calibration, giving group guarantees under joint exchangeability of the
calibration and test examples. Across four multi-modal datasets, the
disagreement-scaled layer matches or improves the marginal conformal baseline
in 59 of 60 paired runs for interval continuous ranked probability score
(CRPS) and in 52 of 60 for interval width, while keeping empirical coverage
near the 95% target. In stress tests with missing modalities, mask-matched
recalibration recovers up to 19.5 percentage points of coverage in the hardest
fixed-mask regime. The result is a simple, model-agnostic reliability layer
for multi-modal regression systems.

## Installation

Python 3.11 is supported. Clone the repository and install the tested
dependency versions with Conda:

```bash
git clone https://github.com/Unco3892/modality-aware-conformal.git
cd modality-aware-conformal
conda create -n modality_aware_conformal python=3.11 -y
conda activate modality_aware_conformal
pip install -r requirements.txt
```

Alternatively, use a virtual environment:

```bash
python3.11 -m venv .venv
. .venv/bin/activate          # .\.venv\Scripts\Activate.ps1 on Windows
pip install --upgrade pip
pip install -r requirements.txt
```

Exact versions are recorded in `requirements.txt` to reduce numerical drift.
For GPU neural-baseline runs, install the PyTorch build appropriate for the
local CUDA environment before installing the remaining requirements.

## Repository layout

```text
.
├── src/
│   ├── multimodal_calibration/
│   │   ├── calibration.py                  # conformal methods and metrics
│   │   ├── experiment_config.py            # shared scientific configuration
│   │   ├── run_predagn_ablation.py         # cross-dataset experiments across base predictors
│   │   ├── run_missing_regime_ablation.py  # stress tests with missing modalities
│   │   ├── run_hetero_mixture.py           # source-wise mixture experiments
│   │   ├── run_nn_baselines.py             # neural baselines
│   │   ├── aggregate_with_nn.py             # auxiliary-result aggregation
│   │   ├── exp_worked_examples.py           # worked-example generation
│   │   ├── exp_gamma_sim.py                 # simulation of scale tuning at small sample sizes
│   │   ├── run_baseline.py                 # frozen-feature construction
│   │   └── loaders/                        # dataset-specific loaders
│   ├── sred/
│   │   ├── run_full_experiment.py          # SRED/SEMF experiment driver
│   │   ├── augmented_theta.py              # derived SEMF interval analysis
│   │   ├── testsplit_semf_calibration.py   # held-out SEMF calibration
│   │   ├── predictor_agnostic_mondrian.py  # SEMF predictor comparison
│   │   └── baseline.py                     # SRED feature construction
│   └── semf/                               # adapted SEMF implementation
├── results/                                # curated per-seed and summary CSVs
├── docs/                                   # project page
├── data/
│   └── cache_manifest.json                 # reference cache inventory
├── requirements.txt
└── LICENSE
```

## Data

Raw datasets and frozen-encoder features are not committed. Experiment drivers
expect the reviewed feature arrays under:

```text
data/
├── sred/embeddings/
├── mercari/embeddings/
├── pawpularity/embeddings/
└── imdb_wiki/embeddings/
```

### Reference cache workflow

The reported experiments use a locally installed, verified reference cache.
For the present private workflow, place the reviewed arrays in the
directories above and verify them against the tracked input inventory in
`src/multimodal_calibration/experiment_config.py`.

The installer uses the tracked `data/cache_manifest.json`, which fixes every
archive and extracted array by name, size, and checksum (SHA-256). This
prevents a partial download or a different feature build from being used
silently. Install the reviewed local bundle with:

```bash
python src/multimodal_calibration/download_caches.py --archive-dir /path/to/cache_release
```

The directory must contain one manifest-named ZIP for each requested dataset.
The installer validates the archive and every extracted array before
installation. Verify an existing installation without changing it with:

```bash
python src/multimodal_calibration/download_caches.py --verify-only
```

Subject to the underlying dataset licences, a future GitHub release can use
the same installer:

```bash
python src/multimodal_calibration/download_caches.py --base-url https://github.com/Unco3892/modality-aware-conformal/releases/download/CACHE_TAG
```

The release path must use an immutable tag rather than `latest`. No public cache release is configured or claimed at present.

### Raw inputs

Mercari and Pawpularity are Kaggle competitions:

```bash
python src/multimodal_calibration/download_kaggle.py --datasets mercari-price-suggestion-challenge petfinder-pawpularity-score
```

The downloader reads `KAGGLE_API_TOKEN` from `.env` or the environment and
writes the raw files below `data/mercari/raw/` and
`data/pawpularity/raw/`.

For IMDB-WIKI, download `wiki_crop.tar` from the
[original release](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/) and
extract it under `data/imdb_wiki/` so that
`data/imdb_wiki/wiki_crop/wiki.mat` exists.

For SRED, download `SRED_data.zip` from the
[SRED_2022 release](https://github.com/Unco3892/SRED_2022) and unpack it so
that `data/sred/metadata/train_data_with_text.csv` and
`data/sred/processed_images/{train,test}/` exist.

### Rebuilding features

The documented encoder recipes can be run from the raw inputs:

```bash
python src/sred/baseline.py
python src/sred/baseline.py --approximate-fast
python src/multimodal_calibration/run_baseline.py --dataset mercari
python src/multimodal_calibration/run_baseline.py --dataset pawpularity
python src/multimodal_calibration/run_baseline.py --dataset imdb_wiki
```

The two SRED commands produce the cross-dataset E5/ViT-B features and the
smaller MiniLM/ViT-S features used by the SEMF diagnostic. Feature rebuilding
is a separate reproduction path: newly encoded arrays are exact reference
inputs only if their hashes match the reviewed inventory. The loaders do not
silently fall back to another encoder or an incomplete cache.

The remaining reference recipes are dataset-specific: Mercari uses
multilingual MiniLM text features, Pawpularity uses DINOv2 ViT-S/14 image
features, and IMDB-WIKI uses both.

## Running the experiments

The public experiment entry points live under `src/`. Their scientific
defaults are centralized in
`src/multimodal_calibration/experiment_config.py`.

```bash
python src/multimodal_calibration/run_predagn_ablation.py --help
python src/multimodal_calibration/run_missing_regime_ablation.py --help
python src/multimodal_calibration/run_hetero_mixture.py --help
python src/multimodal_calibration/run_nn_baselines.py --help
python src/multimodal_calibration/aggregate_with_nn.py --help
python src/multimodal_calibration/exp_worked_examples.py --help
python src/multimodal_calibration/exp_gamma_sim.py --help
python src/sred/run_full_experiment.py --help
python src/sred/augmented_theta.py --help
python src/sred/testsplit_semf_calibration.py --help
python src/sred/predictor_agnostic_mondrian.py --help
```

Each driver accepts explicit datasets, seeds, and output directories. For
example, a small isolated predictor-agnostic run is:

```bash
python src/multimodal_calibration/run_predagn_ablation.py --datasets sred --seeds 0 --campaign-tag local_check --allow-incomplete --out-dir results/local_check/predictor_agnostic
```

Use the complete recorded dataset and seed grid for a full replication; do
not treat a one-seed check as evidence for the reported aggregate results.

Run `aggregate_with_nn.py` after the heterogeneous-model and neural
experiments. For SRED, run `run_full_experiment.py`, then
`augmented_theta.py`, before either calibration diagnostic.

SEMF training uses the corrected replication-weight alignment by default. The
historical ordering remains available through `--weight-alignment legacy` for
artifact comparison and is not used by the reported results.

## Included results

The tracked `results/` directory contains 16 curated CSV files and one JSON
configuration record. It retains per-seed rows alongside their derived
summaries for the cross-dataset benchmark across base predictors, the
comparisons by disagreement bin, the stress tests with missing modalities, and
the SRED SEMF diagnostics, plus the compact auxiliary-baseline aggregate and
the summary of the scale-tuning simulation with its configuration record. All
files match what was obtained in the paper.

## Reproducibility note

The verified feature caches are the reference inputs for downstream
experiments. Rebuilding encoders can introduce small
hardware-dependent floating-point differences, which tree models may amplify
into different but scientifically comparable fits. The dependency versions,
input hashes, run configuration, and output manifests record the conditions
needed to distinguish exact artifact reproduction from a fresh scientific
replication.

The reported downstream experiments were run on a Slurm-managed Linux x86-64
cluster using CPython 3.11.15. The main predictor-agnostic and
missing-modality jobs were allocated 12 CPUs and 72 GB of RAM per task. The
auxiliary neural and heterogeneous-model jobs used one NVIDIA A100 PCIe 40 GB
GPU, eight CPUs, and 48 GB of RAM per task, while the SEMF jobs used CPU
allocations of up to 48 CPUs and 48 GB of RAM. The exact dependency versions
are listed in `requirements.txt`.

## Citation

```bibtex
@inproceedings{azizi2026modality,
  title     = {Conformal Calibration for Multi-Modal Regression with Missing Modalities},
  author    = {Azizi, Ilia},
  booktitle = {Proceedings of the Fifteenth Symposium on Conformal and Probabilistic Prediction with Applications},
  series    = {Proceedings of Machine Learning Research},
  volume    = {329},
  year      = {2026},
  publisher = {PMLR}
}
```

## License

MIT, see [LICENSE](LICENSE).
