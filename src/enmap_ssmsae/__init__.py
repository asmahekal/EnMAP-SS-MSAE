"""EnMAP-SS-MSAE: masked spectral autoencoders for label-free hyperspectral
spectral-unit mapping and anomaly detection.

Modules
-------
preprocessing : robust band scaling and benchmark (.mat) loading
models        : 1-D convolutional masked spectral autoencoder and training loop
scoring       : RX, PCA-RX, latent RX, local latent contrast and rank fusion
metrics       : ROC-AUC, PR-AUC, F1@P99, FPR@95TPR
"""
__version__ = "1.1.0"
