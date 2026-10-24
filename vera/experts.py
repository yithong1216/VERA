"""Fine evidence II: tail and nose region experts (Eq. margin).

Part geometry places an aligned 224x224 canvas on the tail (from the mean horizontal
stabilizer centroid, scaled by the half-wingspan) or on the nose (about the estimated
nose tip, scaled by aircraft length). A ConvNeXt-Tiny classifies each canvas over the
known classes; logits average two seeds and a horizontal flip. The evidence for claim c
is the clipped log-odds m_r = clip(z_c - logsumexp_{k != c} z_k, -3, 3), and 0 when the
region geometry is unavailable.
"""
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision

from .geometry import invert_part_polygon, polygon_centroid

K = ['A2', 'A3', 'A4', 'A6', 'A7', 'A8', 'A9', 'A10', 'A14', 'A20']
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
CLIP = 3.
SEEDS = (1000, 1001)
CONVNEXT_TINY = Path.home() / '.cache/torch/hub/checkpoints/convnext_tiny-983f1562.pth'
dev = 'cuda'

# Nose canvas: everything is a fraction of aircraft length L along the alignment axis.
# The box over-extends, so the nose tip is moved back by DELTA_F * L.
DELTA_F, BACK_F, FRONT_F = 0.15, 0.40, 0.22


# ---------------------------------------------------------------- canvases

def tail_centroid(part):
    """Mean horizontal-stabilizer centroid in source-image coordinates, or None."""
    cs = []
    for pr in part['predictions']:
        if float(pr['score']) < .30 or str(pr['refined_type']).split('|', 1)[0] != 'horizontal_tail':
            continue
        try:
            cs.append(polygon_centroid(invert_part_polygon(pr['polygon'], part['source_to_crop'])))
        except ValueError:
            continue
    return np.mean(cs, axis=0) if cs else None


def tail_matrix(tail, axis, hs):
    """Affine from the source image to the tail canvas; ``hs`` is the half-wingspan."""
    axis = np.asarray(axis, float)
    tr = np.array([-axis[1], axis[0]])
    M = np.eye(3)
    M[0, :2] = axis * (224 / hs)
    M[1, :2] = tr * (224 / (.70 * hs))
    M[0, 2] = -float(M[0, :2] @ tail)
    M[1, 2] = 112 - float(M[1, :2] @ tail)
    return M


def nose_matrix(origin, u, hs, source_to_crop, obb_crop):
    """Affine from the source image to the nose canvas; ``u`` points toward the tail."""
    u = np.asarray(u, float)
    origin = np.asarray(origin, float)
    S = np.array(source_to_crop, float)
    if S.shape == (2, 3):
        S = np.vstack([S, [0, 0, 1]])
    c = np.c_[np.asarray(obb_crop, float), np.ones(len(obb_crop))] @ np.linalg.inv(S).T
    pr = c[:, :2] @ u
    pmin, pmax = float(pr.min()), float(pr.max())
    L = pmax - pmin
    nose = origin + u * (pmin - float(u @ origin) + DELTA_F * L)
    a = -u
    tr = np.array([-a[1], a[0]])
    P0 = nose + u * BACK_F * L
    span = (BACK_F + FRONT_F) * L
    M = np.eye(3)
    M[0, :2] = a * (224 / span)
    M[1, :2] = tr * (224 / span)
    M[0, 2] = -float(M[0, :2] @ P0)
    M[1, 2] = 112 - float(M[1, :2] @ origin)
    return M, nose


def render(image_bgr, M):
    """Warp a source image (BGR) onto a 224x224 RGB canvas."""
    canvas = cv2.warpAffine(image_bgr, M[:2], (224, 224), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------- classifier

def degrade(im, rng):
    if rng.random() < 0.7:
        f = rng.uniform(1.0, 3.0)
        h, w = im.shape[:2]
        im = cv2.resize(cv2.resize(im, (max(int(w / f), 8), max(int(h / f), 8)), interpolation=cv2.INTER_AREA),
                        (w, h), interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.7:
        im = cv2.GaussianBlur(im, (0, 0), rng.uniform(0.5, 2.5))
    if rng.random() < 0.5:
        q = int(rng.integers(35, 90))
        im = cv2.imdecode(cv2.imencode('.jpg', im, [cv2.IMWRITE_JPEG_QUALITY, q])[1], 1)
    return im


def augment(im, rng):
    im = im.copy()
    if rng.random() < 0.5:
        im = im[:, ::-1]
    M = cv2.getRotationMatrix2D((112, 112), rng.uniform(-10, 10), rng.uniform(0.9, 1.1))
    im = cv2.warpAffine(np.ascontiguousarray(im), M, (224, 224), flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REFLECT)
    im = degrade(im, rng)
    return np.clip(im.astype(np.float32) * rng.uniform(0.8, 1.2) + rng.uniform(-20, 20), 0, 255)


def to_t(ims):
    x = (np.stack(ims).astype(np.float32) / 255 - MEAN) / STD
    return torch.from_numpy(x).permute(0, 3, 1, 2).to(dev)


def build(checkpoint=CONVNEXT_TINY):
    m = torchvision.models.convnext_tiny()
    m.load_state_dict(torch.load(checkpoint))
    m.classifier[2] = nn.Linear(768, len(K))
    return m.to(dev)


def train(X, y, seed, epochs=12, bs=32):
    """Class-balanced, label-smoothed cross-entropy on aligned known canvases."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = build()
    head = list(m.classifier[2].parameters())
    hid = {id(p) for p in head}
    opt = torch.optim.AdamW([{'params': [p for p in m.parameters() if id(p) not in hid], 'lr': 1e-4},
                             {'params': head, 'lr': 1e-3}], weight_decay=0.05)
    spe = int(np.ceil(len(y) / bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[1e-4, 1e-3], total_steps=epochs * spe, pct_start=0.1)
    cnt = np.bincount(y, minlength=len(K)).astype(np.float32)
    w = torch.tensor(cnt.sum() / (len(K) * np.maximum(cnt, 1)), device=dev)
    scaler = torch.amp.GradScaler('cuda')
    lossf = nn.CrossEntropyLoss(weight=w, label_smoothing=0.1)
    for ep in range(epochs):
        m.train()
        idx = rng.permutation(len(y))
        for k in range(0, len(idx), bs):
            b = idx[k:k + bs]
            x = to_t([augment(X[i], rng) for i in b])
            t = torch.tensor(y[b], device=dev)
            with torch.autocast('cuda', dtype=torch.float16):
                lo = m(x)
            loss = lossf(lo.float(), t)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
    return m


@torch.no_grad()
def logits(m, ims):
    """Logits averaged over the canvas and its horizontal flip."""
    m.eval()
    out = []
    for k in range(0, len(ims), 64):
        x = to_t(ims[k:k + 64])
        with torch.autocast('cuda', dtype=torch.float16):
            out.append(((m(x) + m(x.flip(3))) / 2).float().cpu())
    return torch.cat(out).numpy()


def margin(z):
    """s_c = z_c - logsumexp_{j != c} z_j for every class column."""
    z = np.asarray(z, np.float64)
    out = np.empty_like(z)
    for c in range(z.shape[1]):
        o = np.delete(z, c, axis=1)
        mx = o.max(1, keepdims=True)
        out[:, c] = z[:, c] - (mx[:, 0] + np.log(np.exp(o - mx).sum(1)))
    return out


def region_evidence(seed_logits, claims, available):
    """Eq. margin: m_r(x, c) for each proposal's claimed class index.

    ``seed_logits`` is a list of (n, |K|) logit arrays (one per seed), ``claims`` the
    claimed class indices and ``available`` a boolean mask of proposals whose region
    geometry exists; unavailable regions contribute 0.
    """
    z = np.mean(np.stack(seed_logits), axis=0)
    m = np.clip(margin(z)[np.arange(len(claims)), np.asarray(claims)], -CLIP, CLIP)
    return np.where(np.asarray(available, bool), m, 0.)
