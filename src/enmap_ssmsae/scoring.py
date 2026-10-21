"""Anomaly scores: classical RX, PCA-RX and the three SS-MSAE cues with rank fusion."""
import numpy as np
from scipy.ndimage import uniform_filter
from scipy.stats import rankdata
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA


def ecdf_rank_score(x):
    """Empirical rank in [0, 1] (average ranks for ties)."""
    x = np.asarray(x, dtype=float).ravel()
    if len(x) <= 1:
        return np.zeros_like(x)
    return (rankdata(x, method='average') - 1) / (len(x) - 1)


def classical_rx(X):
    """Global RX with a Ledoit-Wolf shrinkage covariance."""
    cov = LedoitWolf().fit(X)
    d = X - cov.location_
    return np.einsum('ij,jk,ik->i', d, cov.precision_, d)


def pca_rx(X, ncomp=20, seed=42):
    n = min(ncomp, X.shape[1], max(2, X.shape[0] - 1))
    return classical_rx(PCA(n_components=n, random_state=seed).fit_transform(X))


def latent_rx(Z):
    """RX distance on standardized latent vectors."""
    mu = Z.mean(0); sd = Z.std(0); sd = np.where(sd > 1e-8, sd, 1.0)
    return classical_rx(((Z - mu) / sd).astype(np.float32))


def local_latent_contrast(Z, H, W, size=5):
    """RMS distance of each latent vector to its size x size neighbourhood mean."""
    z = Z.reshape(H, W, -1).astype(np.float32)
    loc = np.empty_like(z)
    for j in range(z.shape[2]):
        loc[:, :, j] = uniform_filter(z[:, :, j], size=size, mode='reflect')
    d = z - loc
    return np.sqrt(np.mean(d * d, axis=2)).ravel()


def spectral_recon_score(X, R):
    return np.mean((R - X) ** 2, axis=1)


def fixed_fusion(rec, lrx, local, weights=(0.30, 0.45, 0.25)):
    """Scene-independent rank fusion used by SS-MSAE (weights fixed a priori)."""
    w1, w2, w3 = weights
    return w1 * ecdf_rank_score(rec) + w2 * ecdf_rank_score(lrx) + w3 * ecdf_rank_score(local)


def ss_msae_scores(model_outputs, X, H, W, size=5):
    """Convenience wrapper: (Z, R) from the autoencoder -> dict of cue and fused scores."""
    Z, R = model_outputs
    rec = spectral_recon_score(X, R)
    lrx = latent_rx(Z)
    loc = local_latent_contrast(Z, H, W, size)
    return {'rec': rec, 'latent_rx': lrx, 'local': loc, 'ss_msae': fixed_fusion(rec, lrx, loc)}
