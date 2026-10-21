"""Run RX, PCA-RX and SS-MSAE on one ABU or HYDICE benchmark scene (.mat file).

Usage:
    python scripts/run_benchmark_scene.py path/to/abu-urban-4.mat [--epochs 15] [--seed 42]

The ABU scenes can be downloaded from
https://github.com/sxt1996/Airport-Beach-Urban-ABU (one .mat file per scene).
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from enmap_ssmsae.preprocessing import load_mat_scene, robust_band_scale  # noqa: E402
from enmap_ssmsae.models import train_masked_ae, encode_reconstruct  # noqa: E402
from enmap_ssmsae.scoring import classical_rx, pca_rx, ss_msae_scores  # noqa: E402
from enmap_ssmsae.metrics import metrics_from_score  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('mat'); ap.add_argument('--epochs', type=int, default=15); ap.add_argument('--seed', type=int, default=42)
a = ap.parse_args()
cube, gt = load_mat_scene(a.mat)
H, W, B = cube.shape
X, _, _ = robust_band_scale(cube)
model, _, _ = train_masked_ae(X, mode='derivative', seed=a.seed, epochs=a.epochs)
s = ss_msae_scores(encode_reconstruct(model, X), X, H, W)
s['RX'] = classical_rx(X); s['PCA-RX'] = pca_rx(X, 20)
print(f'{Path(a.mat).name}: {H}x{W}x{B}, anomalies={int(gt.sum())}')
for k in ['RX', 'PCA-RX', 'rec', 'latent_rx', 'local', 'ss_msae']:
    m = metrics_from_score(gt.ravel(), s[k])
    print(f'{k:<10} ROC-AUC={m["roc_auc"]:.4f}  PR-AUC={m["pr_auc"]:.4f}  F1@P99={m["f1_p99"]:.4f}')
