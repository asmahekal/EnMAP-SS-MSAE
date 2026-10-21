"""Generate the small synthetic hyperspectral scene used by quick_test.py.

100 x 100 pixels, 60 bands. The background is a spatially smooth mixture of three
smooth endmember spectra plus noise; twelve small targets with a distinct spectrum
are implanted at sub-pixel abundance (30-60%). The output file
examples/data/synthetic_scene.npz is already included in the repository, so this
script only needs to be run if you want to regenerate it.
"""
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter

H, W, B, SEED = 100, 100, 60, 7
rng = np.random.default_rng(SEED)
wl = np.linspace(0, 1, B)
E = np.stack([0.30 + 0.20 * wl - 0.05 * np.exp(-((wl - 0.7) / 0.05) ** 2),
              0.20 + 0.25 * np.sin(2.5 * wl) ** 2,
              0.45 - 0.15 * wl + 0.04 * np.cos(9 * wl)])
target = 0.25 + 0.10 * wl + 0.12 * np.exp(-((wl - 0.35) / 0.04) ** 2) - 0.10 * np.exp(-((wl - 0.85) / 0.03) ** 2)
A = np.stack([gaussian_filter(rng.random((H, W)), 6) for _ in range(3)], -1)
A = (A - A.mean()) / A.std()
A = np.exp(1.5 * A); A /= A.sum(-1, keepdims=True)
cube = (A @ E) + rng.normal(0, 0.006, (H, W, B))
gt = np.zeros((H, W), np.uint8)
for (r, c, s) in [(10, 15, 2), (18, 80, 3), (38, 45, 2), (52, 12, 3), (62, 70, 2), (78, 30, 3), (86, 86, 2), (30, 28, 2), (70, 50, 3), (90, 10, 2), (8, 60, 3), (45, 92, 2)]:
    a = rng.uniform(0.3, 0.6)
    cube[r:r + s, c:c + s] = (1 - a) * cube[r:r + s, c:c + s] + a * target
    gt[r:r + s, c:c + s] = 1
out = Path(__file__).parent / 'data' / 'synthetic_scene.npz'
out.parent.mkdir(exist_ok=True)
np.savez_compressed(out, cube=cube.astype(np.float32), gt=gt)
print(f'saved {out}  cube={cube.shape}  anomaly pixels={gt.sum()}')
