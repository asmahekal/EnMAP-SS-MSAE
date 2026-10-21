"""Masked spectral autoencoder used by SS-MSAE (benchmark configuration).

The encoder is a four-block 1-D CNN over the spectral axis with a three-channel
input (masked spectrum, binary mask, normalized band position). Training uses
random band masking and the loss  L = L_mask + lambda_d * L_der + 0.03 * L_global.
"""
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class BenchSpectralMAE(nn.Module):
    def __init__(self, n_bands, latent_dim=24):
        super().__init__()
        self.n_bands = n_bands
        self.encoder = nn.Sequential(
            nn.Conv1d(3, 32, 7, padding=3), nn.BatchNorm1d(32), nn.GELU(),
            nn.Conv1d(32, 64, 5, stride=2, padding=2), nn.BatchNorm1d(64), nn.GELU(),
            nn.Conv1d(64, 96, 5, stride=2, padding=2), nn.BatchNorm1d(96), nn.GELU(),
            nn.Conv1d(96, 128, 3, stride=2, padding=1), nn.BatchNorm1d(128), nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.to_latent = nn.Linear(128, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.GELU(),
            nn.Linear(128, 256), nn.GELU(),
            nn.Linear(256, n_bands),
        )
        pos = np.linspace(0, 1, n_bands, dtype=np.float32)
        self.register_buffer('pos', torch.tensor(pos).view(1, 1, -1))

    def make_input(self, x, mask=None):
        if mask is None:
            mask = torch.zeros_like(x, dtype=torch.bool)
        xm = x.masked_fill(mask, 0.0)
        p = self.pos.expand(x.shape[0], -1, -1)
        return torch.cat([xm[:, None, :], mask.float()[:, None, :], p], 1)

    def forward(self, x, mask=None):
        h = self.encoder(self.make_input(x, mask)).squeeze(-1)
        z = self.to_latent(h)
        return self.decoder(z), z


def train_masked_ae(X, mode='derivative', seed=42, epochs=15, batch_size=512, mask_ratio=0.35,
                    latent_dim=24, max_train=30000, device=None, verbose=False):
    """Train a spectral autoencoder on unlabelled pixels.

    mode: 'plain' (full MSE, no masking), 'masked' (masked MSE + 0.03 global) or
    'derivative' (masked MSE + 0.12 derivative + 0.03 global; the SS-MSAE encoder).
    Returns (model, losses, training_seconds).
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = BenchSpectralMAE(X.shape[1], latent_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs), eta_min=2e-4)
    rng = np.random.default_rng(seed)
    Xt = X[rng.choice(len(X), max_train, replace=False)] if len(X) > max_train else X
    dl = DataLoader(TensorDataset(torch.from_numpy(Xt)), batch_size=batch_size, shuffle=True)
    losses, best, state = [], np.inf, None
    t0 = time.perf_counter()
    for ep in range(epochs):
        model.train(); s = 0.0; n = 0
        for (xb,) in dl:
            xb = xb.to(device)
            if mode == 'plain':
                mask = torch.zeros_like(xb, dtype=torch.bool)
            else:
                mask = torch.rand_like(xb) < mask_ratio
                empty = ~mask.any(1)
                if empty.any():
                    mask[empty, 0] = True
            rec, _ = model(xb, mask)
            sq = (rec - xb) ** 2
            if mode == 'plain':
                loss = sq.mean()
            else:
                recon = (sq * mask.float()).sum() / mask.float().sum().clamp_min(1.0)
                global_loss = sq.mean()
                if mode == 'derivative':
                    dm = (mask[:, 1:] | mask[:, :-1]).float()
                    dr = (rec[:, 1:] - rec[:, :-1]) - (xb[:, 1:] - xb[:, :-1])
                    deriv = ((dr ** 2) * dm).sum() / dm.sum().clamp_min(1.0)
                    loss = recon + 0.12 * deriv + 0.03 * global_loss
                else:
                    loss = recon + 0.03 * global_loss
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); s += float(loss.item()) * len(xb); n += len(xb)
        sch.step(); el = s / max(n, 1); losses.append(el)
        if verbose:
            print(f'  epoch {ep + 1:02d}/{epochs} loss={el:.5f}')
        if el < best:
            best = el; state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if state is not None:
        model.load_state_dict(state)
    model.to(device).eval()
    return model, losses, time.perf_counter() - t0


@torch.no_grad()
def encode_reconstruct(model, X, batch=4096):
    """Return latent vectors Z (N, d) and reconstructions R (N, B)."""
    device = next(model.parameters()).device
    Z, R = [], []
    for s in range(0, len(X), batch):
        xb = torch.from_numpy(X[s:s + batch]).to(device)
        rec, z = model(xb, None)
        Z.append(z.cpu().numpy()); R.append(rec.cpu().numpy())
    return np.vstack(Z), np.vstack(R)
