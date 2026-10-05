# EnMAP-SS-MSAE

Code and results for the manuscript:

> **Self-Supervised Masked Spectral Autoencoders for Label-Free Spectral-Unit Mapping and Anomaly Prioritization in EnMAP Imagery, with Multi-Benchmark Validation**
> Fatma M. Talaat, Asmaa A. Hekal (manuscript under review)

The pipeline learns per-pixel spectral representations with masked spectral autoencoders and uses them for two label-free tasks:

1. **Spectral-unit mapping** of a full EnMAP Level-2A scene (south-central Eastern Desert of Egypt), using latent clustering with K selection, multi-seed stability and a 3 × 3 spatial consensus.
2. **Hyperspectral anomaly detection**: SS-MSAE rank-fuses the reconstruction error, the latent RX distance and the local latent contrast. It is validated on 14 public scenes (13 ABU + HYDICE Urban) using ground truth **only after** score generation.

## Repository structure

```
notebooks/
  enmap_ss_msae_pipeline.ipynb   # complete pipeline (EnMAP + benchmark validation)
  make_study_area_figure.py      # Figure 2 of the paper
results/
  enmap/
    00_scene_summary.csv ... 06_top_anomaly_candidates.csv
    tables/                      # K selection, unit statistics, thresholds, seed robustness, bootstrap
    geotiff/                     # spectral units (raw, consensus), anomaly score, P99 anomaly mask (UTM 36N)
    figures/                     # paper-ready figures
    ENMAP_PAPER_RESULTS_TABLES.xlsx, run_config_PRO.json, V3_EXPERIMENT_MANIFEST.json
  benchmark/
    10_benchmark_all_metrics.csv ... 17_bootstrap_scene_level_ci.csv
    figures/, per_scene_maps/, BENCHMARK_PAPER_RESULTS.xlsx
```

## Requirements

Python ≥ 3.10 and a CUDA GPU are recommended (the pipeline also runs on CPU, but much more slowly).

```bash
pip install -r requirements.txt
```

`libarchive` is used only to extract a zipped or RAR-compressed EnMAP product inside the notebook.

## Data

| Data | Source | Notes |
|---|---|---|
| EnMAP L2A scene `ENMAP01-____L2A-DT0000180014_20260216T085731Z_004` | [EnMAP data portal (DLR)](https://www.enmap.org/data_access/) | Not redistributed here because of the EnMAP data licence. Register with DLR and download it yourself. |
| ABU (13 scenes) | public repositories, e.g. [sxt1996/Airport-Beach-Urban-ABU](https://github.com/sxt1996/Airport-Beach-Urban-ABU) | Downloaded automatically by the notebook |
| HYDICE Urban | public repository, e.g. [sxt1996/HYDICE](https://github.com/sxt1996/HYDICE) | Downloaded automatically by the notebook |

## How to run

The notebook was developed on Kaggle.

1. Create a Kaggle notebook and upload `notebooks/enmap_ss_msae_pipeline.ipynb`.
2. Add the EnMAP L2A product (ZIP or RAR) as a Kaggle dataset. It appears under `/kaggle/input`.
3. Enable **GPU** and **Internet**, which the benchmark download needs.
4. Run all cells. Outputs are written to `/kaggle/working/enmap_aamsae_paper/results`.

When run outside Kaggle, the notebook reads its inputs from `/mnt/data` and writes there as well. Edit `DATA_ROOT` / `WORK_ROOT` in the configuration cell to change this.

Key settings: seed 42; 35% band masking; latent dimension 32 (EnMAP) / 24 (benchmark); AdamW, lr 2e-3, cosine decay to 2e-4; batch size 512; 25 epochs (EnMAP AAMSAE) / 15 epochs (benchmark); benchmark seeds 42, 123, 2026; fusion weights 0.30 / 0.45 / 0.25. Small numerical differences can occur across GPU and library versions.

## Main results

| | Value |
|---|---|
| EnMAP scene | 1194 × 1165 px, 219 bands (418–2445 nm), 1,034,606 valid pixels (931 km²) |
| Best clustering representation | MSAE, K = 4 (Silhouette 0.336, DB 0.979, ARI 0.818) |
| Lowest reconstruction error | AAMSAE (MSE 0.0075) |
| Spatial consensus | neighbour agreement 0.830 → 0.900 |
| SS-MSAE on 14 benchmarks | ROC-AUC 0.953 (95% CI 0.932–0.973), PR-AUC 0.390 |
| Statistical comparison | significantly better than the latent-RX detectors; not significantly different from RX, PCA-RX or MSAE-Fusion |

**Interpretation note:** the EnMAP spectral units and anomaly scores are label-free screening products. They are **not** lithological classes or confirmed mineral occurrences, and they require independent field or laboratory validation.

## Citation

If you use this code, please cite the paper (details will be added after publication) and this repository (see `CITATION.cff`).

## Acknowledgements

EnMAP data © DLR. We thank the providers of the ABU and HYDICE benchmark datasets.

## Licence

MIT. See `LICENSE`. The licence covers the code only, not the EnMAP data.
