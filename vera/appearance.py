"""Coarse evidence: background-referenced appearance LLR (Eqs. whitening, gaussian, appearance).

RoI features are whitened by a fitted PCA (d_a = 32). Each known class is a Gaussian
with a shared, Ledoit-Wolf shrunk covariance fitted on class-centered known residuals;
the background is one Gaussian over known and generic aircraft. The evidence for a
claim c is log p_c(e) - log p_bg(e).
"""
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA

KNOWN = ['A2', 'A3', 'A4', 'A6', 'A7', 'A8', 'A9', 'A10', 'A14', 'A20']
NEGATIVE = ['A1', 'A5', 'A13', 'A16', 'A19']
PCA_DIM = 32


def gaussian_logpdf(x, mean, covariance):
    delta = np.asarray(x, dtype=np.float64) - np.asarray(mean, dtype=np.float64)
    sign, logdet = np.linalg.slogdet(covariance)
    assert sign > 0 and np.isfinite(logdet)
    solution = np.linalg.solve(covariance, delta.T).T
    quad = np.einsum("nd,nd->n", delta, solution)
    return -.5 * (delta.shape[1] * np.log(2 * np.pi) + logdet + quad)


def fit_appearance(features, labels, known=KNOWN, dimension=PCA_DIM):
    """Fit the whitening transform, class references and background reference.

    ``features`` are RoI embeddings of known and generic (fighter) training aircraft;
    ``labels`` are their class codes. Evaluated unknown types must not be included.
    """
    labels = np.asarray(labels)
    counts = {code: int(np.sum(labels == code)) for code in known}
    assert all(count >= 2 for count in counts.values()), counts
    assert len(labels) > dimension
    pca = PCA(n_components=dimension, whiten=True, svd_solver="full")
    fit_z = pca.fit_transform(np.asarray(features).astype(np.float64))
    means = {code: fit_z[labels == code].mean(axis=0) for code in known}
    known_mask = np.isin(labels, known)
    residuals = np.asarray([z - means[code] for z, code in zip(fit_z[known_mask], labels[known_mask])])
    covariance = LedoitWolf(assume_centered=False).fit(residuals).covariance_
    background_mean = fit_z.mean(axis=0)
    background_covariance = LedoitWolf(assume_centered=False).fit(fit_z).covariance_
    return dict(pca=pca, known=list(known), means=means, covariance=covariance,
                background_mean=background_mean, background_covariance=background_covariance)


def appearance_llr(model, features):
    """Return the (n, |K|) matrix of ell_app(x, c) for every known class c."""
    z = model["pca"].transform(np.asarray(features).astype(np.float64))
    background = gaussian_logpdf(z, model["background_mean"], model["background_covariance"])
    scores = np.column_stack([gaussian_logpdf(z, model["means"][code], model["covariance"]) - background
                              for code in model["known"]])
    assert np.isfinite(scores).all()
    return scores
