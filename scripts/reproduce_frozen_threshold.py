"""Reproduce the main VERA-z operating points from the released per-proposal evidence terms.

    python scripts/reproduce_frozen_threshold.py [--bootstrap 2000]

1. fit the five VERA-z scales on development-split localized known proposals;
2. calibrate tau at 90.5% localized-known retention on the development split;
3. apply that tau unchanged to the test split (dev-frozen, the paper's main protocol);
4. for reference, also report the test-calibrated point (tau refit on test known scores).
"""
import argparse, csv, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vera import fusion, evaluation as E  # noqa: E402

KNOWN = E.KNOWN


def load(split):
    d = ROOT / 'results' / 'scores'
    P = list(csv.DictReader(open(d / f'{split}_proposals.csv')))
    G = list(csv.DictReader(open(d / f'{split}_gt_objects.csv')))
    terms = {t: np.array([float(r[t]) for r in P]) for t in fusion.TERMS}
    pred = np.array([KNOWN.index(r['claimed_class']) for r in P])
    scene = dict(label=np.array([int(g['label']) for g in G]), code=np.array([g['aircraft_type_code'] for g in G]),
                 prop=np.array([int(g['matched_proposal_index']) for g in G]), pred=pred)
    family = np.array([g['scene_family_id'] for g in G])
    return terms, scene, family, np.array([float(r['vera_z']) for r in P])


def row(r):
    return (f"retention {100 * r['localized_known_retention']:5.1f}  look-alike rejection {100 * r['G3']['rejection']:5.1f}  "
            f"all-unknown rejection {100 * r['all5']['rejection']:5.1f}  known E2E {100 * r['known_e2e']:5.1f}  tau {r['tau']:.4f}")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--bootstrap', type=int, default=0); a = ap.parse_args()
    dev_terms, dev, _, dev_stored = load('development')
    test_terms, test, test_family, test_stored = load('test')
    known = (dev['label'] >= 0) & (dev['label'] < 10) & (dev['prop'] >= 0)
    kp = dev['prop'][known]
    scales = fusion.fit_scales(*[dev_terms[t][kp] for t in fusion.TERMS])
    print('development scales', {k: round(v, 4) for k, v in scales.items()})
    v_dev = fusion.verification_score(*[dev_terms[t] for t in fusion.TERMS], scales=scales)
    v_test = fusion.verification_score(*[test_terms[t] for t in fusion.TERMS], scales=scales)
    assert np.allclose(v_dev, dev_stored, atol=1e-4) and np.allclose(v_test, test_stored, atol=1e-4)
    d = E.summary(dev, v_dev)
    print('development (calibration)     ', row(d))
    frozen = E.summary(test, v_test, tau=d['tau'])
    print('test, dev-frozen tau  [main]  ', row(frozen))
    print('test, test-calibrated tau     ', row(E.summary(test, v_test)))
    if a.bootstrap:
        draws = E.family_bootstrap(test, v_test, test_family, d['tau'], n=a.bootstrap)
        lo, hi = np.percentile([100 * x['G3_rejection'] for x in draws], [2.5, 97.5])
        print(f'look-alike rejection 95% scene-family bootstrap interval (tau fixed): [{lo:.1f}, {hi:.1f}]')


if __name__ == '__main__':
    main()
