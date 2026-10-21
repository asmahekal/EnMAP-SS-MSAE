"""Quick test of the SS-MSAE anomaly detector on a small synthetic scene.

Runs on a CPU in about one minute. It trains the derivative-aware masked
spectral autoencoder on the unlabelled scene, computes the three anomaly cues and
their rank fusion, and compares the result with classical RX and PCA-RX. The
ground-truth mask is used only at the end, to compute the metrics.

Usage (from the repository root):
    python examples/quick_test.py
"""
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from enmap_ssmsae.preprocessing import robust_band_scale  # noqa: E402
from enmap_ssmsae.models import train_masked_ae, encode_reconstruct  # noqa: E402
from enmap_ssmsae.scoring import classical_rx, pca_rx, ss_msae_scores  # noqa: E402
from enmap_ssmsae.metrics import metrics_from_score  # noqa: E402


def main(epochs=15):
    t0 = time.perf_counter()
    d = np.load(ROOT / 'examples' / 'data' / 'synthetic_scene.npz')
    cube, gt = d['cube'], d['gt']
    H, W, B = cube.shape
    print(f'Scene: {H} x {W} pixels, {B} bands, {int(gt.sum())} anomaly pixels')

    X, _, _ = robust_band_scale(cube)                      # label-free preprocessing
    model, losses, train_s = train_masked_ae(X, mode='derivative', seed=42, epochs=epochs, device='cpu')
    print(f'Trained masked spectral AE for {epochs} epochs in {train_s:.1f} s (final loss {losses[-1]:.4f})')
    scores = ss_msae_scores(encode_reconstruct(model, X), X, H, W)
    scores['RX'] = classical_rx(X)
    scores['PCA-RX'] = pca_rx(X, 20)

    y = gt.ravel()                                         # ground truth used only from here on
    print(f'\n{"Method":<16}{"ROC-AUC":>9}{"PR-AUC":>9}{"F1@P99":>9}')
    order = [('RX', 'RX'), ('PCA-RX', 'PCA-RX'), ('rec', 'Reconstruction'), ('latent_rx', 'Latent RX'),
             ('local', 'Local contrast'), ('ss_msae', 'SS-MSAE (fused)')]
    res = {}
    for k, name in order:
        m = metrics_from_score(y, scores[k]); res[k] = m
        print(f'{name:<16}{m["roc_auc"]:>9.3f}{m["pr_auc"]:>9.3f}{m["f1_p99"]:>9.3f}')
    ok = res['ss_msae']['roc_auc'] > 0.90
    print(f'\nTotal time: {time.perf_counter() - t0:.1f} s')
    print('QUICK TEST PASSED' if ok else 'QUICK TEST FAILED (SS-MSAE ROC-AUC <= 0.90)')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
