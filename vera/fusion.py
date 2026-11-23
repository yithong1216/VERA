"""Evidence accumulation and decision.

VERA scores a known-class claim c for a proposal x by adding five evidence terms:

    t = (log(a(x) q_c(x)),  ell_app(x, c),  ell_str(x, c),  m_tail(x, c),  m_nose(x, c))

The tail and nose margins are clipped to [-3, 3] before fusion.

VERA-z (the revised method) divides each term by its standard deviation over the
localized known proposals of the development split and adds the standardized terms
with equal weights:

    V(x, c) = sum_j t_j(x, c) / s_j,      s_j = SD_dev(t_j)

The scales s_j and the threshold tau are both fitted on the development split and are
then frozen; nothing is refitted on the test split. A claim is accepted when V >= tau;
otherwise the box keeps its location and is relabelled unknown.

Development-split scales fitted on the released MAR20 terms (results/scores):
    log-detection 0.7255, appearance 9.3303, structure 2.8002, tail 1.6304, nose 1.1726
"""
import numpy as np

RETENTION = 0.905
MARGIN_CLIP = 3.0
UNKNOWN = -1
TERMS = ('log_detection', 'appearance', 'structure', 'tail', 'nose')
PAPER_SCALES = dict(log_detection=0.7255, appearance=9.3303, structure=2.8002, tail=1.6304, nose=1.1726)


def claim(known_confidence):
    """c = argmax_k q_k(x)."""
    return np.asarray(known_confidence).argmax(1)


def _stack(log_detection, appearance, structure, tail, nose, clip=MARGIN_CLIP):
    terms = [np.asarray(v, np.float64) for v in (log_detection, appearance, structure, tail, nose)]
    if not all(np.isfinite(v).all() for v in terms):
        raise ValueError('nonfinite evidence component')
    terms[3] = np.clip(terms[3], -clip, clip)  # an unavailable region margin is passed as 0
    terms[4] = np.clip(terms[4], -clip, clip)
    return terms


def fit_scales(log_detection, appearance, structure, tail, nose):
    """Per-term standard deviations over development-split localized known proposals."""
    terms = _stack(log_detection, appearance, structure, tail, nose)
    scales = {k: float(np.std(v)) for k, v in zip(TERMS, terms)}
    if not all(s > 0 for s in scales.values()):
        raise ValueError('degenerate evidence term on the development split')
    return scales


def verification_score(log_detection, appearance, structure, tail, nose, scales=None):
    """VERA-z score for the claimed class (pass scales=None for the unscaled VERA sum)."""
    terms = _stack(log_detection, appearance, structure, tail, nose)
    if scales is None:
        return sum(terms)
    return sum(v / scales[k] for k, v in zip(TERMS, terms))


def threshold(development_known_scores, retention=RETENTION):
    """tau = Q_{1 - rho} over development-split localized known scores; frozen for testing."""
    return float(np.quantile(np.asarray(development_known_scores, np.float64), 1 - retention))


def decide(scores, claims, tau):
    """Accepted claims keep their class index; refused claims become UNKNOWN."""
    return np.where(np.asarray(scores) >= tau, np.asarray(claims), UNKNOWN)
