"""Fine evidence I: part structure (Eqs. coordinates, count, polar, fisher, shape, structure).

Input is one proposal crop with typed part polygons from the frozen part detector and
the detector box in crop coordinates. Each proposal becomes a record holding clipped
physical part counts, polar position bins and 16 geometric descriptors under up to four
heading hypotheses. Known-class templates and one background template are fitted on
training proposals; ell_str is the standardized class-background contrast of the
count, polar and Fisher-weighted geometry scores.
"""
from collections import Counter
import math

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import f as fdist

KNOWN = ['A2', 'A3', 'A4', 'A6', 'A7', 'A8', 'A9', 'A10', 'A14', 'A20']
FIGHTERS = ['A1', 'A5', 'A13', 'A16', 'A19']
TYPES = ['main_wing|straight', 'main_wing|swept_fixed', 'main_wing|variable_sweep',
         'propeller_disk|under_wing', 'propeller_disk|on_wing',
         'exhaust|fuselage_integrated', 'exhaust|tail_mounted',
         'intake|fuselage_integrated', 'intake|under_wing',
         'vertical_tail|center', 'vertical_tail|twin',
         'nacelle', 'horizontal_tail', 'nose_cockpit', 'canard', 'rotodome']
NT = len(TYPES)
INDEX = {t: i for i, t in enumerate(TYPES)}

PART_THRESHOLD = .30
MAXC, ALPHA = 12, .1
RB, RMAX, TB = 40, 1.6, 36
RW, TW = RMAX / RB, np.pi / TB
PEPS, SIGMA_FLOOR, NU = 1e-4, .5, 4.
VAR_FLOOR, LAMBDA = 1e-4, .1
DS = 16
ABSTAIN_MIN_SUPPORT = 50

ORIGIN = np.array([224., 224.])
NOSES = np.array([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]])
WING = {INDEX[t] for t in TYPES if t.startswith('main_wing|')}
TAILS = [INDEX[t] for t in ('horizontal_tail', 'vertical_tail|center', 'vertical_tail|twin',
                            'exhaust|fuselage_integrated', 'exhaust|tail_mounted')]
HTAIL = INDEX['horizontal_tail']
ENGINE = {INDEX['nacelle']} | {INDEX[t] for t in TYPES if t.startswith('propeller_disk|')}


# ---------------------------------------------------------------- measurement

def centroid(vertices):
    p = np.asarray(vertices, dtype=np.float64)
    x, y = p[:, 0], p[:, 1]
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    area = cross.sum() / 2
    if abs(area) < 1e-6:
        return p.mean(0)
    return np.array([((x + np.roll(x, -1)) * cross).sum(), ((y + np.roll(y, -1)) * cross).sum()]) / (6 * area)


def area(P):
    return abs(float((P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1]).sum())) / 2


def _log(v):
    return float(np.log(max(float(v), 1e-3)))


def descriptors(t, X, ob):
    """16 shape descriptors in one hypothesis frame; X are polygons as (x toward nose, y lateral)."""
    d = np.full(DS, np.nan)
    length, width = np.ptp(ob[:, 0]), np.ptp(ob[:, 1])
    nose, tail = ob[:, 0].max(), ob[:, 0].min()
    d[0] = _log(length / max(width, 1e-3))
    wings = [P for u, P in zip(t, X) if u in WING]
    if not wings:
        return d
    W = np.concatenate(wings)
    span = max(2 * np.abs(W[:, 1]).max(), 1e-3)
    wing_area = max(sum(area(P) for P in wings), 1e-3)
    xw = np.mean([P.mean(0)[0] for P in wings])
    d[1], d[2] = _log(span / length), _log(span ** 2 / wing_area)
    d[3] = 2 * np.abs(W[:, 1]).min() / span
    d[4] = (nose - xw) / max(nose - tail, 1e-3)
    sweeps = []
    for P in wings:
        y = np.abs(P[:, 1])
        lo, hi = np.quantile(y, [.33, .67])
        inner, outer = y <= lo, y >= hi
        if inner.any() and outer.any() and y[outer].mean() > y[inner].mean():
            sweeps.append(np.degrees(np.arctan2(P[inner, 0].max() - P[outer, 0].max(),
                                                y[outer].mean() - y[inner].mean())))
    if sweeps:
        d[5] = np.mean(sweeps) / 10
    tails = [P for u, P in zip(t, X) if u == HTAIL]
    if tails:
        H = np.concatenate(tails)
        hs = max(2 * np.abs(H[:, 1]).max(), 1e-3)
        d[6], d[7] = _log(hs / span), _log(sum(area(P) for P in tails) / wing_area)
        d[8] = (xw - np.mean([P.mean(0)[0] for P in tails])) / span
        d[9] = (H[:, 0].min() - tail) / span
        if len(tails) >= 2:
            d[10] = 2 * np.abs(H[:, 1]).min() / hs
    engines = [P for u, P in zip(t, X) if u in ENGINE]
    if len(engines) >= 2:
        eta = np.sort([abs(P.mean(0)[1]) / (span / 2) for P in engines])
        d[11], d[12] = eta[:len(eta) // 2].mean(), eta[len(eta) // 2:].mean()
        d[13] = (np.mean([P.mean(0)[0] for P in engines]) - xw) / span
        d[14] = _log(np.mean([np.ptp(P[:, 0]) for P in engines]) / span)
        d[15] = _log(np.mean([np.ptp(P[:, 0]) / max(np.ptp(P[:, 1]), 1e-3) for P in engines]))
    return d


def four_heading_record(t, polys, obb):
    """Counts, polar bins (Eq. coordinates) and descriptors under all four canonical headings."""
    n = len(t)
    centers = np.array([centroid(P) for P in polys]).reshape(-1, 2)
    rb, tb = np.zeros((4, n), dtype=np.int64), np.zeros((4, n), dtype=np.int64)
    shape = np.full((4, DS), np.nan)
    for h in range(4):
        a = NOSES[h]
        p = np.array([-a[1], a[0]])
        X = [np.stack([(P - ORIGIN) @ a, (P - ORIGIN) @ p], -1) for P in polys]
        ob = np.stack([(obb - ORIGIN) @ a, (obb - ORIGIN) @ p], -1)
        wings = [P for u, P in zip(t, X) if u in WING]
        span = 2 * np.abs(np.concatenate(wings)[:, 1]).max() if wings else np.ptp(ob[:, 1])
        if n:
            u = ((centers - ORIGIN) @ a) / (np.ptp(ob[:, 0]) / 2)
            v = np.abs((centers - ORIGIN) @ p) / (max(span, 1e-3) / 2)
            rb[h] = np.minimum(np.floor(np.hypot(u, v) / RW).astype(np.int64), RB - 1)
            tb[h] = np.minimum(np.floor(np.arctan2(v, u) / TW).astype(np.int64), TB - 1)
        shape[h] = descriptors(t, X, ob)
    return dict(counts=np.minimum(np.bincount(t, minlength=NT), MAXC), t=t, rb=rb, tb=tb, shape=shape)


def part_frame(t, centers):
    """Heading set H(x): [(nose index, weight)] from wings and tail parts, or None with fewer than 2 wings."""
    wing = centers[np.isin(t, list(WING))]
    if len(wing) < 2:
        return None
    d = np.linalg.norm(wing[:, None] - wing[None], axis=-1)
    i, j = np.unravel_index(d.argmax(), d.shape)
    span = wing[i] - wing[j]
    axis = 0 if abs(span[1]) >= abs(span[0]) else 1
    tails = centers[np.isin(t, TAILS)]
    if len(tails):
        back = tails.mean(0)[axis] - (wing[i][axis] + wing[j][axis]) / 2
        return [(2 * axis + (1 if back > 0 else 0), 1.)]
    return [(2 * axis, .5), (2 * axis + 1, .5)]


def restrict_record(record, frame):
    """Represent a None/single/equal-pair heading set using four rows."""
    if frame is None:
        return record
    frame = list(frame)
    if len(frame) not in (1, 2) or any(
            int(h) != h or not 0 <= h < 4 or not np.isclose(w, 1/len(frame))
            for h, w in frame):
        raise ValueError('Only single or equally weighted pair frames are supported')
    indices = [int(h) for h, _ in frame] * (4 // len(frame))
    return {key: np.asarray(value)[indices].copy() if key in ('rb', 'tb', 'shape') else value
            for key, value in record.items()}


def physical_counts(parts, types, nt=NT, maximum=MAXC):
    """Per-type counts with engine nacelles counted in physical engine units."""
    units = []
    for part in parts:
        value = int(part['attributes']['unit_count']) if part['category_id'] == 1 else 1
        assert value in (1, 2), 'engine unit_count must be 1 or 2'
        units.append(value)
    return np.minimum(np.rint(np.bincount(types, weights=units, minlength=nt)).astype(np.int64), maximum)


def make_record(partrow):
    """Structure record of one proposal.

    ``partrow['predictions']``: part detections with ``score``, ``refined_type`` (one of
    TYPES), crop ``polygon``, ``category_id`` and ``attributes['unit_count']``.
    ``partrow['teacher_obb_in_crop']``: the aircraft box polygon in crop coordinates.
    """
    parts = [p for p in partrow['predictions'] if p['score'] >= PART_THRESHOLD]
    types = np.array([INDEX[p['refined_type']] for p in parts], dtype=int)
    polys = [np.asarray(p['polygon'], dtype=float) for p in parts]
    obb = np.asarray(partrow['teacher_obb_in_crop'], dtype=float)
    centers = np.array([centroid(p) for p in polys]).reshape(-1, 2)
    record = restrict_record(four_heading_record(types, polys, obb), part_frame(types, centers))
    return dict(record, counts=physical_counts(parts, types))


# ---------------------------------------------------------------- templates

def contribution(replicas, frame):
    w = 1. / len(replicas)
    frames = frame if frame is not None else [(h, .25) for h in range(4)]
    C, P = np.zeros((NT, MAXC + 1)), np.zeros((NT, RB, TB))
    X, W = [], []
    for r in replicas:
        C[np.arange(NT), r['counts']] += w
        for h, wh in frames:
            np.add.at(P, (r['t'], r['rb'][h], r['tb'][h]), w * wh)
            X.append(r['shape'][h])
            W.append(w * wh)
    return C, P, np.array(X), np.array(W)


def count_polar_model(C, P, n):
    """Additively smoothed count and Gaussian-smoothed polar log probabilities."""
    count = np.log((np.maximum(C, 0.) + ALPHA) / (n + ALPHA * (MAXC + 1)))
    P = gaussian_filter(np.maximum(P, 0.), sigma=(0, 1.5, 1.5), mode='reflect')
    mass = P.sum((1, 2), keepdims=True)
    return count, np.log(np.divide(P, mass, out=np.zeros_like(P), where=mass > 1e-12) + PEPS)


def count_polar_score(model, r):
    """(S_cnt, S_pol) of one record (Eqs. count and polar)."""
    count, polar = model
    c = count[np.arange(NT), r['counts']].sum()
    p = polar[r['t'][None, :], r['rb'], r['tb']].mean(1).max() if len(r['t']) else np.log(PEPS)
    return np.array([c, p])


def wmean(X, w):
    obs = ~np.isnan(X)
    ws = w @ obs
    return np.divide(w @ np.where(obs, X, 0.), ws, out=np.full(DS, np.nan), where=ws > 1e-12)


def pooled_cov(R, w):
    obs = ~np.isnan(R)
    Rz = np.where(obs, R, 0.)
    pw = (obs * w[:, None]).T @ obs
    S = np.divide((Rz * w[:, None]).T @ Rz, pw, out=np.zeros((DS, DS)), where=pw > 1e-12)
    var = np.maximum(np.diag(S), VAR_FLOOR)
    S = (1 - LAMBDA) * S
    np.fill_diagonal(S, var)
    vals, vecs = np.linalg.eigh(S)
    return (vecs * np.maximum(vals, VAR_FLOOR)) @ vecs.T


def fisher_weights(mu, bg, var):
    """Eq. fisher: per-coordinate class-background separation, normalized to mean one."""
    mu, bg, var = map(lambda x: np.asarray(x, float), (mu, bg, var))
    valid = np.isfinite(mu) & np.isfinite(bg) & np.isfinite(var)
    w = np.zeros_like(mu)
    w[valid] = (mu[valid] - bg[valid]) ** 2 / np.maximum(var[valid], 1e-12)
    assert w.sum() > 0, 'degenerate Fisher weights'
    return w / w.mean()


def shape(x, mu, sd, w, nu):
    """Eq. shape: Fisher-weighted F-tail geometry score, maximized over the heading set."""
    obs = np.isfinite(x) & np.isfinite(mu)
    u = np.where(obs, (x - mu) / np.maximum(sd, 1e-6), 0.)
    df = np.maximum(obs.sum(1), 1)
    return fdist.logsf((w * u * u).sum(1) / df, df, nu).reshape(-1, 4).max(1)


def fit_class_templates(records, class_codes):
    """Known-class count/polar templates with leave-one-object-out standardization, and shape means."""
    members = {code: [i for i, c in enumerate(class_codes) if c == code] for code in KNOWN}
    if any(not indices for indices in members.values()):
        raise ValueError('Every known class requires fitting samples')
    contributions = [contribution([record], None) for record in records]
    models, means, residuals = {}, {}, {}
    for code in KNOWN:
        mem = members[code]
        C = sum(contributions[i][0] for i in mem)
        P = sum(contributions[i][1] for i in mem)
        count, polar = count_polar_model(C, P, len(mem))
        loo = [count_polar_score(count_polar_model(C - contributions[i][0], P - contributions[i][1], len(mem) - 1),
                                 records[i]) for i in mem]
        models[code] = dict(count=count, polar=polar, cp_mean=np.mean(loo, 0),
                            cp_sd=np.maximum(np.std(loo, 0), SIGMA_FLOOR))
        X = np.concatenate([contributions[i][2] for i in mem])
        W = np.concatenate([contributions[i][3] for i in mem])
        means[code] = wmean(X, W)
        residuals[code] = (X - means[code], W)
        models[code]['shape_mean'] = means[code]
    S = pooled_cov(np.concatenate([residuals[c][0] for c in KNOWN]),
                   np.concatenate([residuals[c][1] for c in KNOWN]))
    return dict(known=list(KNOWN), covariance=S, models=models)


def fit_background(bank, records, labels):
    """Shared structural background template and per-class Fisher geometry models."""
    assert set(labels) <= set(KNOWN + FIGHTERS)
    contrib = [contribution([r], None) for r in records]
    cb = sum(c[0] for c in contrib)
    pb = sum(c[1] for c in contrib)
    x = np.concatenate([c[2] for c in contrib])
    weight = np.concatenate([c[3] for c in contrib])
    mu = wmean(x, weight)
    var = np.maximum(np.diag(pooled_cov(x - mu, weight)), VAR_FLOOR)
    cp = count_polar_model(cb, pb, len(records))
    loo = np.array([count_polar_score(count_polar_model(cb - c[0], pb - c[1], len(records) - 1), r)
                    for c, r in zip(contrib, records)])
    out = dict(background=dict(mean=mu, sd=np.sqrt(var), count=cp[0], polar=cp[1],
                               cp_mean=loo.mean(0), cp_sd=np.maximum(loo.std(0), SIGMA_FLOOR)),
               models={}, nu=NU)
    sd = np.sqrt(np.maximum(np.diag(bank['covariance']), VAR_FLOOR))
    for code in KNOWN:
        m = np.array(bank['models'][code]['shape_mean'], float)
        w = fisher_weights(m, mu, var)
        members = [i for i, c in enumerate(labels) if c == code]
        # Mean-only leave-one-object-out standardization of the class geometry score.
        lt = []
        for i in members:
            other = [j for j in members if j != i]
            mi = wmean(np.concatenate([contrib[j][2] for j in other]), np.concatenate([contrib[j][3] for j in other]))
            lt.append(shape(records[i]['shape'], mi, sd, w, NU)[0])
        bs = shape(x, mu, np.sqrt(var), w, NU)
        out['models'][code] = dict(weights=w, mean=m, sd=sd, t_mean=float(np.mean(lt)),
                                   t_sd=float(max(np.std(lt), SIGMA_FLOOR)),
                                   bg_mean=float(bs.mean()), bg_sd=float(max(bs.std(), SIGMA_FLOOR)))
    return out


def structure_evidence(bank, fisher, records):
    """Eq. structure: (n, |K|) matrix of ell_str(x, c)."""
    if not records:
        return np.empty((0, len(KNOWN)))
    x = np.concatenate([r['shape'] for r in records])
    bg = fisher['background']
    bgcp = np.array([count_polar_score((bg['count'], bg['polar']), r) for r in records])
    bgz = (bgcp - bg['cp_mean']) / bg['cp_sd']
    values = []
    for code in KNOWN:
        model = fisher['models'][code]
        known = bank['models'][code]
        cp = np.array([count_polar_score((np.asarray(known['count']), np.asarray(known['polar'])), r)
                       for r in records])
        zcp = (cp - known['cp_mean']) / known['cp_sd']
        zshape = (shape(x, model['mean'], model['sd'], model['weights'], fisher['nu'])
                  - model['t_mean']) / model['t_sd']
        zbgshape = (shape(x, bg['mean'], bg['sd'], model['weights'], fisher['nu'])
                    - model['bg_mean']) / model['bg_sd']
        values.append(zcp.sum(1) + zshape - zbgshape - bgz.sum(1))
    matrix = np.array(values).T
    assert matrix.shape == (len(records), len(KNOWN)) and np.isfinite(matrix).all()
    return matrix


def fit_and_score(training, targets):
    """Fit templates on ``training`` rows ({'matched_code', '_record'}) and score ``targets``."""
    known = [p for p in training if p['matched_code'] in KNOWN]
    background = [p for p in training if p['matched_code'] in KNOWN + FIGHTERS]
    bank = fit_class_templates([p['_record'] for p in known], [p['matched_code'] for p in known])
    fisher = fit_background(bank, [p['_record'] for p in background], [p['matched_code'] for p in background])
    return bank, fisher, structure_evidence(bank, fisher, [p['_record'] for p in targets])


def abstained_classes(fit_codes, known=KNOWN, minimum=ABSTAIN_MIN_SUPPORT):
    """Classes whose structural reference abstains for lack of fitting support."""
    counts = Counter(fit_codes)
    return [code for code in known if counts[code] < minimum]


def zero_abstained(values, known, abstained):
    if len(values) != len(known) or not all(math.isfinite(value) for value in values):
        raise ValueError('nonfinite or wrong-width structure score vector')
    blocked = set(abstained)
    return [0.0 if code in blocked else float(value) for code, value in zip(known, values)]
