"""Relative (camera-roll-free) trunk asymmetry from the Scoliosis1K-Pose keypoints -> analysis/relative_asym.md

Question: the CNNs read the signed body lean, and the lean of healthy children differs between recording blocks
(subject-id ranges = sessions). A tilted camera rotates the whole body by the same angle, so every single line
(shoulders, hips, trunk axis) tilts by that angle, but the DIFFERENCE between two lines does not change.
Scoliosis tilts body parts by different angles. So:
  global (camera-sensitive) : shoulder line angle, hip line angle, trunk axis lean (hip centre -> shoulder centre)
  relative (roll-invariant) : shoulder - hip angle, trunk axis - hip normal
All signed (the side of the curve is kept), in degrees, image coordinates (y down), median over usable frames.
A frame is usable for a pair when both keypoints have conf >= 0.3 and the pair is wide enough to measure a tilt
(|dx| >= 0.12 x body height, i.e. a front/back view, not a side view).

Checks:
  1. single-feature AUC on test (scoliosis vs negative, neutral vs negative)
  2. recording-block effect: share of the variance among ALL negative children explained by their 100-id block
     (eta^2, one-way ANOVA). Healthy children should not differ by session, so lower = less session effect.
  3. logistic regression (train children) on the global vs the relative set (and DRF PAV for reference):
     test AUC, block eta^2 of the scores of the negatives, the test AUC when every test child gets its own random
     camera roll in [-2, 2] degrees (like a different camera angle per session; mean of 20 draws), and after mirroring.
     A roll r adds exactly r to the three global angles and 0 to the relative ones (checked by rotating the keypoints,
     see transform()), so the rolled features are X + r * [1, 1, 1, 0, 0].

python checkpoint_2/experiments/exp_2/code/relative_asym.py
"""
import json
import pickle
from pathlib import Path

import numpy as np
from scipy.stats import f_oneway
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
REPO = EXP.parents[2]
POSE = REPO / 'output' / 'Scoliosis1K-pose-pkl' / 'Scoliosis1K-pose-pkl'
CACHE = EXP / 'runs' / 'relative_asym_cache.npz'
PAV_CACHE = EXP / 'runs' / 'pav_cache.npz'
LABELS = ['negative', 'neutral', 'positive']
L_SH, R_SH, L_HIP, R_HIP, L_ANK, R_ANK, NOSE = 5, 6, 11, 12, 15, 16, 0
GLOBAL = ['shoulder_angle', 'hip_angle', 'trunk_lean']
RELATIVE = ['shoulder_minus_hip', 'trunk_minus_hip']
NAMES = GLOBAL + RELATIVE


def auc(a, b):
    return 100 * ((a[:, None] > b[None]).mean() + 0.5 * (a[:, None] == b[None]).mean())


def transform(kp, roll=0.0, mirror=False):
    """Rotate every frame about its hip centre by `roll` degrees and/or mirror it left-right (image x -> -x).
    A mirrored video also swaps which keypoint is called left and right."""
    kp = np.array(kp, dtype=np.float64)
    if mirror:
        kp[..., 0] = -kp[..., 0]
        swap = list(range(17))
        for l, r in [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]:
            swap[l], swap[r] = r, l
        kp = kp[:, swap]
    if roll:
        c = (kp[:, L_HIP, :2] + kp[:, R_HIP, :2]) / 2
        t = np.deg2rad(roll)
        R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        kp[..., :2] = (kp[..., :2] - c[:, None]) @ R.T + c[:, None]
    return kp


def line_angle(a, b):
    """Signed tilt in degrees of the line a-b against the image horizontal, independent of which end is which."""
    d = a - b
    d = np.where(d[:, :1] < 0, -d, d)                                   # point the line to +x
    return np.degrees(np.arctan2(d[:, 1], d[:, 0]))


def child_features(kp):
    xy, conf = kp[..., :2], kp[..., 2]
    hip_c = (xy[:, L_HIP] + xy[:, R_HIP]) / 2
    sh_c = (xy[:, L_SH] + xy[:, R_SH]) / 2
    height = np.abs((xy[:, L_ANK, 1] + xy[:, R_ANK, 1]) / 2 - xy[:, NOSE, 1]) + 1e-6
    ok_sh = (conf[:, L_SH] >= .3) & (conf[:, R_SH] >= .3) & (np.abs(xy[:, L_SH, 0] - xy[:, R_SH, 0]) >= .12 * height)
    ok_hip = (conf[:, L_HIP] >= .3) & (conf[:, R_HIP] >= .3) & (np.abs(xy[:, L_HIP, 0] - xy[:, R_HIP, 0]) >= .06 * height)
    sh = line_angle(xy[:, L_SH], xy[:, R_SH])
    hip = line_angle(xy[:, L_HIP], xy[:, R_HIP])
    v = sh_c - hip_c                                                    # trunk axis, points up (y < 0)
    lean = np.degrees(np.arctan2(v[:, 0], -v[:, 1]))                   # 0 = vertical, + = leaning to image right
    both = ok_sh & ok_hip
    f = {'shoulder_angle': sh[ok_sh], 'hip_angle': hip[ok_hip], 'trunk_lean': lean[ok_hip & ok_sh],
         'shoulder_minus_hip': (sh - hip)[both], 'trunk_minus_hip': (lean - hip)[both]}
    # A camera roll by r rotates every vector by r: line angles and the trunk lean all gain +r (checked in main()
    # with the rolled copies), so both differences cancel it.
    return np.array([np.median(f[n]) if len(f[n]) >= 10 else np.nan for n in NAMES])


def build(roll=0.0, mirror=False):
    subs, X, y = [], [], []
    for d in sorted(POSE.iterdir()):
        f = next(d.rglob('*.pkl'))
        kp = transform(pickle.load(open(f, 'rb')), roll, mirror)
        subs.append(d.name); X.append(child_features(kp)); y.append(LABELS.index(f.parts[-3]))
    return np.array(subs), np.array(X), np.array(y)


def splits():
    tv = json.load(open(EXP / 'partitions' / 'Scoliosis1K_118_trainval.json'))
    te = json.load(open(EXP / 'partitions' / 'Scoliosis1K_118_test.json'))
    return tv['TRAIN_SET'], te['TEST_SET']


def eta2(values, groups):
    """Share of the variance explained by the group (one-way ANOVA), and the ANOVA p-value."""
    ok = ~np.isnan(values)
    v, g = values[ok], groups[ok]
    gm = {k: v[g == k].mean() for k in np.unique(g)}
    ss_b = sum((g == k).sum() * (m - v.mean()) ** 2 for k, m in gm.items())
    ss_t = ((v - v.mean()) ** 2).sum()
    p = f_oneway(*[v[g == k] for k in np.unique(g) if (g == k).sum() > 1]).pvalue
    return 100 * ss_b / ss_t, p, gm


def main():
    if not CACHE.exists():
        conds = {'normal': build(), 'roll+5': build(roll=5), 'roll-5': build(roll=-5), 'mirror': build(mirror=True)}
        np.savez(CACHE, subject=conds['normal'][0], y=conds['normal'][2], **{k: v[1] for k, v in conds.items()})
    c = np.load(CACHE)
    subs, y = list(c['subject']), c['y']
    sid = np.array([int(s) for s in subs])
    idx = {s: i for i, s in enumerate(subs)}
    train, test = splits()
    tr, te = [idx[s] for s in train], [idx[s] for s in test]
    med = np.nanmedian(c['normal'][tr], 0)
    X = {k: np.where(np.isnan(c[k]), med, c[k]) for k in ('normal', 'roll+5', 'roll-5', 'mirror')}
    neg = np.where(y == 0)[0]
    block = sid[neg] // 100

    L = ['# Relative trunk asymmetry from pose keypoints (camera-roll-free)', '',
         'Written by `code/relative_asym.py`. Signed angles in degrees, median over usable front/back frames. '
         '"global" features tilt with the camera, "relative" ones are differences of two body lines and do not. '
         f'Children: {len(y)}; test {len(te)}; negatives used for the block check: {len(neg)} (all splits).', '',
         '## 1. Single features: test AUC and the recording-block effect among healthy children', '',
         'AUC > 50: higher value = more scoliotic; < 50: lower value = more scoliotic (the side matters, it is signed). '
         'eta^2 = % of the variance among all negative children explained by their 100-id block (session).', '',
         '| feature | type | mean neg / neu / pos | AUC sco-neg | AUC neu-neg | block eta^2 (negatives) | p |',
         '|---|---|---|---|---|---|---|']
    for j, n in enumerate(NAMES):
        v = X['normal'][:, j]; vt, yt = v[te], y[te]
        e, p, _ = eta2(c['normal'][neg, j], block)
        means = ' / '.join(f'{np.nanmean(c["normal"][y == k, j]):+.2f}' for k in range(3))
        L.append(f"| {n} | {'global' if n in GLOBAL else 'relative'} | {means} | {auc(vt[yt > 0], vt[yt == 0]):.1f} | "
                 f"{auc(vt[yt == 1], vt[yt == 0]):.1f} | {e:.1f} | {p:.1g} |")

    L += ['', '## 2. Mean per 100-id block of the negatives (healthy children), against the class means', '',
          '| group | n | ' + ' | '.join(NAMES) + ' |', '|---|---|' + '---|' * len(NAMES)]
    for b in sorted(np.unique(block)):
        m = neg[block == b]
        L.append(f'| negatives {b * 100}-{b * 100 + 99} | {len(m)} | ' +
                 ' | '.join(f'{np.nanmean(c["normal"][m, j]):+.2f}' for j in range(len(NAMES))) + ' |')
    for k in range(3):
        m = np.where(y == k)[0]
        L.append(f'| all {LABELS[k]} | {len(m)} | ' + ' | '.join(f'{np.nanmean(c["normal"][m, j]):+.2f}' for j in range(len(NAMES))) + ' |')

    # 3. logistic regression: scoliosis (neutral + positive) vs negative score from each feature set
    sets = {'global (shoulder, hip, trunk lean)': [NAMES.index(n) for n in GLOBAL],
            'relative (shoulder-hip, trunk-hip)': [NAMES.index(n) for n in RELATIVE],
            'global + relative': list(range(len(NAMES)))}
    L += ['', '## 3. Logistic regression on the train children (3 classes), score = log p(neu or pos) - log p(neg)', '',
          'Rolled = every test child gets its own random camera roll in [-2, 2] degrees (a different camera angle per '
          'session), mean AUC of 20 draws. Mirrored = left-right flipped test video. Block eta^2 = session effect in the scores '
          'of all negative children (train children included, so it is an in-sample view of the session effect).', '',
          '| features | AUC sco-neg | AUC neu-neg | AUC sco-neg, random camera roll per child (+-2 deg) | AUC sco-neg, mirrored | block eta^2 of negatives |',
          '|---|---|---|---|---|---|']

    def score(model, sc, Z):
        lp = model.predict_log_proba(sc.transform(Z))
        return np.logaddexp(lp[:, 1], lp[:, 2]) - lp[:, 0]

    yt = y[te]
    for name, cols in sets.items():
        sc = StandardScaler().fit(X['normal'][tr][:, cols])
        m = LogisticRegression(max_iter=2000, class_weight='balanced').fit(sc.transform(X['normal'][tr][:, cols]), y[tr])
        s = {k: score(m, sc, X[k][:, cols]) for k in X}
        st = {k: v[te] for k, v in s.items()}
        shift = np.array([1.0 if n in GLOBAL else 0.0 for n in NAMES])[cols]
        rng = np.random.default_rng(0)
        rolled = []
        for _ in range(20):
            r = rng.uniform(-2, 2, size=len(te))[:, None]
            sr = score(m, sc, X['normal'][te][:, cols] + r * shift)
            rolled.append(auc(sr[yt > 0], sr[yt == 0]))
        rolled = np.mean(rolled)
        e, p, _ = eta2(s['normal'][neg], block)
        L.append(f"| {name} | {auc(st['normal'][yt > 0], st['normal'][yt == 0]):.1f} | "
                 f"{auc(st['normal'][yt == 1], st['normal'][yt == 0]):.1f} | {rolled:.1f} | "
                 f"{auc(st['mirror'][yt > 0], st['mirror'][yt == 0]):.1f} | {e:.1f} (p {p:.1g}) |")
    if PAV_CACHE.exists():   # DRF PAV (absolute values) for reference, same split and model
        pc = np.load(PAV_CACHE)
        pidx = {s: i for i, s in enumerate(pc['subject'])}
        P = pc['mean']
        P = np.where(np.isnan(P), np.nanmedian(P[[pidx[s] for s in train]], 0), P)
        ptr, pte = [pidx[s] for s in train], [pidx[s] for s in test]
        sc = StandardScaler().fit(P[ptr])
        m = LogisticRegression(max_iter=2000, class_weight='balanced').fit(sc.transform(P[ptr]), pc['y'][ptr])
        s = score(m, sc, P)
        py = pc['y']; pyt = py[pte]; pneg = np.where(py == 0)[0]
        e, p, _ = eta2(s[pneg], np.array([int(x) for x in pc['subject']])[pneg] // 100)
        L.append(f"| DRF PAV (absolute, reference) | {auc(s[pte][pyt > 0], s[pte][pyt == 0]):.1f} | "
                 f"{auc(s[pte][pyt == 1], s[pte][pyt == 0]):.1f} | not computed | = normal (absolute values) | {e:.1f} (p {p:.1g}) |")
    out = EXP / 'analysis' / 'relative_asym.md'
    out.write_text('\n'.join(L) + '\n', encoding='utf8')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
