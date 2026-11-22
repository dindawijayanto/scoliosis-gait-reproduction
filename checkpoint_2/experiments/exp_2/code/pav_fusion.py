"""PAV (Postural Asymmetry Vector, DRF / "Pose as Clinical Prior", MICCAI 2025) from the Scoliosis1K-Pose keypoints,
and a late fusion with a trained exp_2 model. No GPU, no training of the CNN.

PAV, as in DRF: 8 symmetric COCO pairs (eyes, ears, shoulders, elbows, wrists, hips, knees, ankles) x 3 metrics
  vertical  |y_L - y_R|
  midline   |(x_L + x_R) / 2 - x_hipcentre|
  angle     |atan((y_L - y_R) / (x_L - x_R))|
per frame, after moving the hip centre to the origin and scaling the body height to 1. Then per child: IQR outlier
removal over frames and the mean (= DRF's PAV), plus the std over frames (temporal spread, our extension).
All metrics are absolute values of left/right differences, so a mirrored video gives the same PAV.

Fusion: s = z_model - tau * log(prior) + lam * log p_PAV(class), with z_model = 16 * mean-strip logits of the run,
p_PAV from a logistic regression trained on the exp_2 train children; tau and lam chosen on val by balanced accuracy.

python checkpoint_2/experiments/exp_2/code/pav_fusion.py [--run B_strat_s0 --iter 16000] -> analysis/pav_fusion_<run>.md
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
REPO = EXP.parents[2]
sys.path.insert(0, str(HERE))
from calibrate import LABELS, LOGP, TAUS, metrics, auc   # same metrics as the exp_2 tables

POSE = REPO / 'output' / 'Scoliosis1K-pose-pkl' / 'Scoliosis1K-pose-pkl'
PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
PAIR_NAMES = ['eye', 'ear', 'shoulder', 'elbow', 'wrist', 'hip', 'knee', 'ankle']
CACHE = EXP / 'runs' / 'pav_cache.npz'
LAMS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0]


def frame_pav(kp):
    """kp [T, 17, 3] (x, y, conf) -> [T, 8, 3] metrics, NaN where a keypoint of the pair has conf < 0.3."""
    xy = kp[..., :2].astype(np.float64)
    conf = kp[..., 2]
    hip = (xy[:, 11] + xy[:, 12]) / 2
    ank = (xy[:, 15] + xy[:, 16]) / 2
    height = np.abs(ank[:, 1] - xy[:, 0, 1]) + 1e-6              # nose to ankle centre
    xy = (xy - hip[:, None]) / height[:, None, None]
    out = np.full((len(kp), len(PAIRS), 3), np.nan)
    for j, (l, r) in enumerate(PAIRS):
        dx = xy[:, l, 0] - xy[:, r, 0]
        dy = xy[:, l, 1] - xy[:, r, 1]
        ok = (conf[:, l] >= 0.3) & (conf[:, r] >= 0.3)
        vert = np.abs(dy)
        mid = np.abs((xy[:, l, 0] + xy[:, r, 0]) / 2)                # hip centre is x = 0
        ang = np.abs(np.arctan(dy / np.where(np.abs(dx) < 1e-6, 1e-6, dx)))
        out[ok, j] = np.stack([vert, mid, ang], -1)[ok]
    return out


def child_pav(kp):
    f = frame_pav(kp).reshape(len(kp), -1)                           # [T, 24]
    q1, q3 = np.nanpercentile(f, 25, 0), np.nanpercentile(f, 75, 0)
    lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    f = np.where((f < lo) | (f > hi), np.nan, f)                     # IQR outlier removal, as in DRF
    return np.nanmean(f, 0), np.nanstd(f, 0)


def build_cache():
    subs, mean, std, cls = [], [], [], []
    for d in sorted(POSE.iterdir()):
        f = next(d.rglob('*.pkl'))
        m, s = child_pav(np.asarray(pickle.load(open(f, 'rb'))))
        subs.append(d.name); mean.append(m); std.append(s); cls.append(LABELS.index(f.parts[-3]))
    np.savez(CACHE, subject=np.array(subs), mean=np.array(mean), std=np.array(std), y=np.array(cls))


def split_ids():
    tv = json.load(open(EXP / 'partitions' / 'Scoliosis1K_118_trainval.json'))
    te = json.load(open(EXP / 'partitions' / 'Scoliosis1K_118_test.json'))
    return {'train': tv['TRAIN_SET'], 'val': tv['TEST_SET'], 'test': te['TEST_SET']}


def load_logits(run, split, it):
    z = np.load(EXP / 'runs' / 'logits' / run / split / f'iter_{it:05d}.npz')
    o = np.argsort(z['subject'])
    return np.array(z['subject'])[o], 16 * z['logits'].mean(-1)[o], z['true'][o]


def pav_auc_table(X, y, names):
    rows = []
    for i, n in enumerate(names):
        rows.append((n, auc(X[y > 0, i], X[y == 0, i]), auc(X[y == 1, i], X[y == 0, i])))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='B_strat_s0')
    ap.add_argument('--iter', type=int, default=16000)
    a = ap.parse_args()
    if not CACHE.exists():
        build_cache()
    c = np.load(CACHE)
    idx = {s: i for i, s in enumerate(c['subject'])}
    feats = {'PAV mean (DRF)': c['mean'], 'PAV mean + std over time': np.concatenate([c['mean'], c['std']], 1)}
    names = [f'{p}_{m}' for p in PAIR_NAMES for m in ('vert', 'mid', 'ang')]
    keep = [i for i, n in enumerate(names) if n != 'hip_mid']           # hip midline is 0 by construction
    ids = split_ids()
    lines = [f'# PAV from pose keypoints + late fusion with {a.run} @ {a.iter}', '',
             'Written by `code/pav_fusion.py`. PAV as in DRF (MICCAI 2025): 8 symmetric COCO pairs x (vertical, midline, '
             'angle) differences, hip centre at the origin, height 1, IQR outlier removal, mean over frames. '
             '"+ std" adds the spread over frames. Absolute values: mirror-invariant by construction.', '']

    # 1. single features on test
    Xt = c['mean'][[idx[s] for s in ids['test']]]
    yt = c['y'][[idx[s] for s in ids['test']]]
    tab = sorted(pav_auc_table(Xt[:, keep], yt, [names[i] for i in keep]), key=lambda r: -abs(r[1] - 50))
    lines += ['## 1. Single PAV features on test (AUC, scoliosis vs negative / neutral vs negative)', '',
              '| feature | AUC sco-neg | AUC neu-neg |', '|---|---|---|']
    lines += [f'| {n} | {s:.1f} | {u:.1f} |' for n, s, u in tab[:8]]

    # 2. logistic on PAV alone, and fusion
    _, z_val, y_val = load_logits(a.run, 'val', a.iter)
    s_val = np.array(sorted(ids['val']))
    s_test, z_test, y_test = load_logits(a.run, 'test', a.iter)
    _, z_mir, _ = load_logits(a.run, 'test_mirror', a.iter)
    assert list(s_test) == sorted(ids['test'])
    lines += ['', f'## 2. PAV alone and fused with {a.run} (test; tau, lam chosen on val)', '',
              '| model | tau | lam | acc | Sen | Spe | rec_neu | rec_pos | bal | AUC_sn | AUC_nn | mirror AUC_sn |',
              '|---|---|---|---|---|---|---|---|---|---|---|---|']
    results = {}
    for fname, F in feats.items():
        Ftr = F[[idx[s] for s in ids['train']]][:, keep] if F.shape[1] == 24 else F[[idx[s] for s in ids['train']]]
        cols = keep if F.shape[1] == 24 else keep + [24 + i for i in keep]
        def X(split_ids_):
            return F[[idx[s] for s in split_ids_]][:, cols]
        Xtr, ytr = X(ids['train']), c['y'][[idx[s] for s in ids['train']]]
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        clf = LogisticRegression(max_iter=5000, C=0.3, class_weight='balanced').fit((Xtr - mu) / sd, ytr)
        lp_val = clf.predict_log_proba((X(s_val) - mu) / sd)
        lp_test = clf.predict_log_proba((X(s_test) - mu) / sd)
        # PAV alone (class-balanced weights already target a uniform prior, so no tau)
        m = metrics(y_test, lp_test)
        lines.append(f'| {fname} alone | - | - | {m["acc"]:.1f} | {m["Sen"]:.1f} | {m["Spe"]:.1f} | {m["rec_neu"]:.1f} | '
                     f'{m["rec_pos"]:.1f} | {m["bal"]:.1f} | {m["AUC_sn"]:.1f} | {m["AUC_nn"]:.1f} | (same) |')
        for lams, label in [([0.0], f'{a.run} val-tau'), (LAMS, f'{a.run} + {fname}')]:
            best = None
            for t in TAUS:
                for lam in lams:
                    mv = metrics(y_val, z_val - t * LOGP + lam * lp_val)
                    key = (round(mv['bal'], 6), -abs(t), -lam)
                    if best is None or key > best[0]:
                        best = (key, t, lam)
            _, t, lam = best
            s_t = z_test - t * LOGP + lam * lp_test
            m = metrics(y_test, s_t)
            mm = metrics(y_test, z_mir - t * LOGP + lam * lp_test)
            results[label] = (s_t.argmax(-1) == y_test)
            if label in [r.split(' | ')[0] for r in lines]:
                continue
            lines.append(f'| {label} | {t:g} | {lam:g} | {m["acc"]:.1f} | {m["Sen"]:.1f} | {m["Spe"]:.1f} | {m["rec_neu"]:.1f} | '
                         f'{m["rec_pos"]:.1f} | {m["bal"]:.1f} | {m["AUC_sn"]:.1f} | {m["AUC_nn"]:.1f} | {mm["AUC_sn"]:.1f} |')
    lines = [l for i, l in enumerate(lines) if l not in lines[:i] or not l.startswith('|')]

    # 3. paired bootstrap of the fusion gain
    rng = np.random.default_rng(0)
    base = results[f'{a.run} val-tau']
    lines += ['', f'## 3. Paired bootstrap of balanced accuracy vs {a.run} val-tau (2000 resamples of test children)', '']
    for label, ok in results.items():
        if label == f'{a.run} val-tau':
            continue
        g = []
        for _ in range(2000):
            i = rng.integers(0, len(y_test), len(y_test))
            g.append(np.mean([100 * (ok[i][y_test[i] == k].mean() - base[i][y_test[i] == k].mean()) for k in range(3)]))
        lines.append(f'- {label}: {np.mean(g):+.1f} [{np.percentile(g, 2.5):+.1f}, {np.percentile(g, 97.5):+.1f}]')

    # 4. recording-block check inside the negatives (notebook 08: lean varies between id blocks)
    neg = [s for s in ids['train'] + ids['val'] + ids['test'] if c['y'][idx[s]] == 0]
    Fall = c['mean'][:, keep]
    mu, sd = Fall.mean(0), Fall.std(0) + 1e-9
    Xn = (Fall[[idx[s] for s in neg]] - mu) / sd
    blocks = np.array([int(s) // 100 for s in neg])
    Xs = (Fall[[idx[s] for s in ids['test']]] - mu) / sd
    w = LogisticRegression(max_iter=5000, C=0.3, class_weight='balanced').fit(
        (Fall[[idx[s] for s in ids['train']]] - mu) / sd, (c['y'][[idx[s] for s in ids['train']]] > 0).astype(int))
    sn = w.decision_function(Xn)
    ss = w.decision_function(Xs)
    lines += ['', '## 4. Recording-block check (scoliosis score of a PAV-only logistic, trained on the train children)', '',
              'Mean score per block of negative subject ids, against the class means on test. If the negative blocks '
              'spread as much as the classes, PAV also carries a recording effect.', '',
              '| group | n | mean score |', '|---|---|---|']
    for b in sorted(set(blocks)):
        lines.append(f'| negatives, ids {b * 100}-{b * 100 + 99} | {np.sum(blocks == b)} | {sn[blocks == b].mean():+.2f} |')
    for k, n in enumerate(LABELS):
        lines.append(f'| test {n} | {np.sum(yt == k)} | {ss[yt == k].mean():+.2f} |')
    out = EXP / 'analysis' / f'pav_fusion_{a.run}.md'
    out.write_text('\n'.join(lines) + '\n', encoding='utf8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
