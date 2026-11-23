"""Evaluation: known end-to-end accuracy, unknown rejection per group, achieved known retention and AUROC.

Protocol used in the paper: tau is calibrated once on the development split
(summary(dev_scene, dev_score) with tau=None) and then passed unchanged to
summary(test_scene, test_score, tau=tau). The achieved test retention is reported
next to every rejection rate. The bootstrap resamples test scene families with tau held fixed.
"""
import numpy as np

KNOWN = ['A2','A3','A4','A6','A7','A8','A9','A10','A14','A20']
UNKNOWN = ['A11','A12','A15','A17','A18']
GROUPS = {'all5':UNKNOWN,'G3':['A11','A12','A18'], **{c:[c] for c in UNKNOWN}}


def summary(scene, score, weights=None, tau=None, retention=0.905):
    lab, code, prop, pred = [scene[k] for k in ('label','code','prop','pred')]
    w = np.ones(len(lab), int) if weights is None else np.asarray(weights)
    assert w.shape == lab.shape and w.dtype.kind in 'iu' and np.all(w >= 0)
    loc = prop >= 0
    known = (lab >= 0) & (lab < 10)
    sp = np.where(loc, score[np.maximum(prop,0)], -np.inf)
    pp = np.where(loc, pred[np.maximum(prop,0)], -9)
    localized_known = known & loc
    assert w[known].sum()
    if tau is None:  # calibrate on this split (use only on the development split)
        calibration = np.repeat(sp[localized_known], w[localized_known])
        assert len(calibration)
        tau = float(np.quantile(calibration, 1-retention))
    accept = sp >= tau
    correct = known & loc & accept & (pp == lab)
    out = dict(tau=tau, known_e2e=float((w*correct).sum()/w[known].sum()),
        known_correct_count=int((w*correct).sum()), known_GT=int(w[known].sum()),
        localized_known_count=int(w[localized_known].sum()),
        localized_known_retention=float((w*(localized_known&accept)).sum()/w[localized_known].sum()))
    for group, cs in GROUPS.items():
        u = np.isin(code, cs)
        rejected = u & loc & ~accept
        n = int(w[u].sum())
        assert n > 0
        out[group] = dict(GT_count=n, localized_count=int((w*(u&loc)).sum()),
            rejected_count=int((w*rejected).sum()), rejection=float((w*rejected).sum()/n))
    out['per_known'] = {}
    for c, name in enumerate(KNOWN):
        m = known & (lab == c)
        n, nl, kept = int(w[m].sum()), int((w*(m&loc)).sum()), int((w*(m&loc&accept)).sum())
        out['per_known'][name] = dict(GT_count=n, localized_count=nl, retained_count=kept,
            retention_all_GT=kept/n if n else None,
            retention_localized=kept/nl if nl else None)
    return out


def family_bootstrap(scene, score, family, tau, n=2000, seed=20260930):
    """Scene-family bootstrap with a frozen tau; returns n flat summaries.

    family: one scene-family id per GT object in scene (objects of one family are drawn together).
    """
    family = np.asarray(family)
    fams, inverse = np.unique(family, return_inverse=True)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n):
        counts = np.bincount(rng.integers(0, len(fams), len(fams)), minlength=len(fams))
        draws.append(flat_summary(summary(scene, score, weights=counts[inverse], tau=tau)))
    return draws


def flat_summary(r):
    out = {'known_e2e':r['known_e2e'], 'localized_known_retention':r['localized_known_retention'], 'tau':r['tau']}
    out.update({g+'_rejection':r[g]['rejection'] for g in GROUPS})
    for c,v in r['per_known'].items():
        for k in ('retention_all_GT','retention_localized'):
            out[c+'_'+k] = v[k]
    return out


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if not len(pos) or not len(neg): return None
    ordered = np.sort(neg)
    return float((np.searchsorted(ordered,pos,'left') + np.searchsorted(ordered,pos,'right')).sum()/(2*len(pos)*len(neg)))


def saturation(par):
    mk,sk,mo,so = map(float,par)
    a = .5*(1/so**2 - 1/sk**2)
    b = mk/sk**2 - mo/so**2
    c = -.5*mk**2/sk**2 + .5*mo**2/so**2 + np.log(so/sk) - 3
    roots = np.roots([a,b,c]) if a != 0 else np.roots([b,c])
    roots = sorted(float(r.real) for r in roots if abs(r.imag) < 1e-10)
    increasing = [r for r in roots if 2*a*r+b > 0]
    return dict(mu_c=mk,std_c=sk,mu_other=mo,std_other=so,
        roots_LLR_equals_plus3=roots,
        increasing_crossing=increasing[0] if increasing else None,
        saturation_region='outside the two roots' if a > 0 else 'between the two roots' if a < 0 else 'above the root' if b > 0 else 'below the root')
