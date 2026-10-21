# EnMAP-SS-MSAE

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23157865.svg)](https://doi.org/10.5281/zenodo.23157865)

Code and results for the manuscript:

> **Self-Supervised Masked Spectral Autoencoders for Label-Free Spectral-Unit Mapping and Anomaly Prioritization in EnMAP Imagery, with Multi-Benchmark Validation**
> Fatma M. Talaat, Asmaa A. Hekal (manuscript under review)

## Purpose

This repository provides open-source Python code for label-free analysis of hyperspectral imagery with masked spectral autoencoders. It covers two tasks:

1. **Spectral-unit mapping** of a full EnMAP Level-2A scene. Pixel spectra are encoded with a masked spectral autoencoder, the latent vectors are clustered with K selection and multi-seed stability checks, and the unit map is refined with a 3 × 3 spatial consensus.
2. **Hyperspectral anomaly detection (SS-MSAE).** A derivative-aware masked spectral autoencoder is trained on each scene without labels. Three cues (reconstruction error, latent RX distance and local latent contrast) are then combined by fixed rank fusion.

## Repository structure

```
src/enmap_ssmsae/              # Python package (importable source files)
  preprocessing.py             # robust band scaling, loading of ABU/HYDICE .mat scenes
  models.py                    # 1-D CNN masked spectral autoencoder and training loop
  scoring.py                   # RX, PCA-RX, latent RX, local latent contrast, rank fusion
  metrics.py                   # ROC-AUC, PR-AUC, F1@P99, FPR@95TPR
examples/
  quick_test.py                # quick test (about 20 s on a CPU)
  make_synthetic_scene.py      # script that generated the test scene
  data/synthetic_scene.npz     # small synthetic test scene (100 x 100 px, 60 bands)
scripts/
  run_benchmark_scene.py       # run RX, PCA-RX and SS-MSAE on one ABU/HYDICE .mat scene
notebooks/
  enmap_ss_msae_pipeline.ipynb # full pipeline used for the paper (EnMAP + 14 benchmarks)
  make_study_area_figure.py    # Figure 2 of the paper
results/                       # tables, GeoTIFFs and figures reported in the paper
requirements.txt, LICENSE (MIT), CITATION.cff
```

## Installation

Python ≥ 3.10 is required. A GPU is optional.

```bash
git clone https://github.com/asmahekal/EnMAP-SS-MSAE.git
cd EnMAP-SS-MSAE
pip install -r requirements.txt
```

## Quick test

From the repository root, run:

```bash
python examples/quick_test.py
```

The script loads `examples/data/synthetic_scene.npz`, which contains a 100 × 100-pixel, 60-band scene with 73 implanted sub-pixel anomaly pixels. It then:

1. trains the masked spectral autoencoder without labels (15 epochs, CPU);
2. computes the SS-MSAE cues and the fused score;
3. compares them with RX and PCA-RX.

The ground-truth mask is used only at the end, to compute the metrics. On a standard laptop CPU the test takes about 20 s and ends with `QUICK TEST PASSED`. The expected output (exact values may vary slightly across PyTorch versions) is:

```
Method            ROC-AUC   PR-AUC   F1@P99
RX                  0.997    0.923    0.775
PCA-RX              1.000    1.000    0.844
Reconstruction      0.619    0.187    0.197
Latent RX           1.000    0.999    0.844
Local contrast      1.000    0.988    0.844
SS-MSAE (fused)     0.962    0.432    0.358
QUICK TEST PASSED
```

The test checks that the full pipeline runs end to end. It is not a performance benchmark: on this simple synthetic scene the classical detectors are already near-perfect.

## Running on a benchmark scene

Download an ABU scene (for example `abu-urban-4.mat`) from https://github.com/sxt1996/Airport-Beach-Urban-ABU, then run:

```bash
python scripts/run_benchmark_scene.py path/to/abu-urban-4.mat --epochs 15 --seed 42
```

## Reproducing the paper

`notebooks/enmap_ss_msae_pipeline.ipynb` contains the complete pipeline used for the manuscript (EnMAP spectral-unit mapping plus validation on 13 ABU scenes and HYDICE Urban). It was run on Kaggle with an NVIDIA Tesla P100 GPU:

1. Add the EnMAP L2A product as a Kaggle dataset.
2. Enable GPU and Internet; the benchmark scenes are downloaded automatically.
3. Run all cells. The outputs are written to `/kaggle/working/enmap_aamsae_paper/results`.

The results reported in the paper are included in `results/`.

## Data

| Data | Source |
|---|---|
| EnMAP L2A scene `ENMAP01-____L2A-DT0000180014_20260216T085731Z_004` | EnMAP data portal (DLR), https://www.enmap.org/data_access/. It is not redistributed here because of the EnMAP data licence. |
| ABU (13 scenes) | https://github.com/sxt1996/Airport-Beach-Urban-ABU |
| HYDICE Urban | https://github.com/sxt1996/HYDICE |

## Licence and citation

The code is released under the MIT licence (see `LICENSE`); the licence covers the code only, not the EnMAP data. If you use the code, please cite the paper and this repository (see `CITATION.cff`, DOI: [10.5281/zenodo.23157865](https://doi.org/10.5281/zenodo.23157865)).

## Contact

Asmaa A. Hekal, Egyptian Chinese University, Cairo, Egypt (asmaa.aboulfadl@ecu.edu.eg)
