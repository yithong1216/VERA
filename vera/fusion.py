"""Evidence accumulation and decision (Eqs. claim, verification, threshold, veto band).

V(x, c) = log(a(x) q_c(x)) + ell_app(x, c) + ell_str(x, c) + m_tail(x, c) + m_nose(x, c)

All terms have unit weight. A claim is accepted when V >= tau_rho, where tau_rho is the
empirical (1 - rho) quantile of localized-known scores; otherwise the box keeps its
location and is relabelled unknown.
"""
import numpy as np

RETENTION = 0.905
UNKNOWN = -1


def claim(known_confidence):
    """Eq. claim: c = argmax_k q_k(x)."""
    return np.asarray(known_confidence).argmax(1)


def verification_score(log_detection, appearance, structure, tail, nose):
    """Eq. verification; every argument is a length-n array for the claimed class."""
    terms = [np.asarray(v, np.float64) for v in (log_detection, appearance, structure, tail, nose)]
    if not all(np.isfinite(v).all() for v in terms):
        raise ValueError('nonfinite evidence component')
    return sum(terms)


def threshold(known_scores, retention=RETENTION):
    """Eq. threshold: tau_rho = Q_{1 - rho} over localized known aircraft."""
    return float(np.quantile(np.asarray(known_scores, np.float64), 1 - retention))


def decide(scores, claims, tau):
    """Accepted claims keep their class index; refused claims become UNKNOWN."""
    return np.where(np.asarray(scores) >= tau, np.asarray(claims), UNKNOWN)
