"""Preprocessing utilities (identical to the benchmark pipeline in the notebook)."""
import numpy as np
from scipy.io import loadmat


def robust_band_scale(cube):
    """Robust per-band standardization.

    Each band is centred on its median and divided by IQR/1.349 (falls back to
    the standard deviation when the robust scale is ~0). Values are clipped to
    [-8, 8].

    Parameters
    ----------
    cube : ndarray, shape (H, W, B)

    Returns
    -------
    X : ndarray, shape (H*W, B), float32
    med, scale : ndarray, shape (B,)
    """
    H, W, B = cube.shape
    X = cube.reshape(-1, B).astype(np.float32)
    med = np.median(X, axis=0)
    q25 = np.percentile(X, 25, axis=0)
    q75 = np.percentile(X, 75, axis=0)
    scale = (q75 - q25) / 1.349
    std = X.std(axis=0)
    scale = np.where(scale > 1e-8, scale, np.where(std > 1e-8, std, 1.0))
    X = np.clip((X - med) / scale, -8, 8).astype(np.float32)
    return X, med.astype(np.float32), scale.astype(np.float32)


def load_mat_scene(path):
    """Load an ABU / HYDICE benchmark scene stored as a MATLAB file.

    The file must contain a 3-D cube (key ``data``) and a 2-D ground-truth map
    (key ``map``); if the keys differ, the largest 3-D and smallest 2-D numeric
    arrays are used. Returns ``cube`` (H, W, B) float32 and binary ``gt`` (H, W).
    """
    m = loadmat(path)
    arrs = {k: np.asarray(v) for k, v in m.items()
            if not k.startswith('__') and isinstance(v, np.ndarray) and np.issubdtype(np.asarray(v).dtype, np.number)}
    cube, gt = arrs.get('data'), arrs.get('map')
    if cube is None or gt is None:
        cubes = [a for a in arrs.values() if a.ndim == 3]
        maps = [a for a in arrs.values() if a.ndim == 2]
        if not cubes or not maps:
            raise ValueError(f'Could not identify cube/map in {path}; keys={list(arrs)}')
        cube, gt = max(cubes, key=lambda a: a.size), min(maps, key=lambda a: a.size)
    cube = np.asarray(cube, dtype=np.float32)
    gt = np.asarray(gt).squeeze()
    if cube.shape[:2] != gt.shape:
        if cube.shape[1:] == gt.shape:
            cube = np.moveaxis(cube, 0, -1)
        elif (cube.shape[0], cube.shape[2]) == gt.shape:
            cube = np.moveaxis(cube, 1, -1)
        else:
            raise ValueError(f'Spatial mismatch cube={cube.shape}, gt={gt.shape}')
    gt = (gt > 0).astype(np.uint8)
    for b in range(cube.shape[2]):
        band = cube[:, :, b]
        bad = ~np.isfinite(band)
        if bad.any():
            med = np.nanmedian(band)
            band[bad] = med if np.isfinite(med) else 0.0
    return cube, gt
