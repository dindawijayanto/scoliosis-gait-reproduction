"""Choose checkpoint, tau and aux-head fusion on VAL only, then report once on the official TEST split.

Input: checkpoint_2/experiments/exp_2/runs/logits/<run>/{val,test}/iter_XXXXX.npz (from export_logits.py)
Score: z = 16 * mean-over-strips logits. Logit adjustment at test: z_c - tau * log(prior_c),
prior = reduced train set (476 / 59 / 59). LA-trained models already target a balanced prior, so tau
may be negative; the grid is -0.5..1.5.
Aux fusion (only runs with a head), weight lam chosen on val:
  bam     : z_neu += lam * 16 * mean(b_neu - b_rest)
  ordinal : z     += lam * log P_ord(y), P_ord from the two cumulative heads
Rules (ties -> smaller |tau|, smaller lam, later checkpoint):
  fixed      argmax at 16k (or the last checkpoint of a short run), no val use
  val-tau    checkpoint + tau chosen by val balanced accuracy
  val-fuse   checkpoint + tau + lam chosen by val balanced accuracy (= val-tau for runs without a head)
Beyond accuracy (analysis checkpoint, threshold-free):
  probes      AUC scoliosis-vs-negative and neutral-vs-negative on the normal, mirrored and deskewed test set
  curves      val / test balanced accuracy (val tau) and AUC per checkpoint -> analysis/curves.json
  embedding   5-NN leave-one-out balanced accuracy on the 256x16 test embeddings (how separable the classes are)
Usage: python checkpoint_2/experiments/exp_2/code/calibrate.py   ->  prints per-run and per-variant tables, writes checkpoint_2/experiments/exp_2/analysis/summary.{json,md}
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent            # exp_2/code
EXP = HERE.parent                                   # checkpoint_2/experiments/exp_2
LABELS = ['negative', 'neutral', 'positive']
TAUS = np.round(np.arange(-0.5, 1.501, 0.05), 2)
LAMS = [0.0, 0.25, 0.5, 1.0, 2.0]
LOGP = np.log(np.array([476, 59, 59]) / 594)
KEYS = ['acc', 'Sen', 'Spe', 'rec_neg', 'rec_neu', 'rec_pos', 'bal', 'AUC_nn', 'AUC_sn']


def load(d):
    out = {}
    for f in sorted(Path(d).glob('iter_*.npz')):
        z = np.load(f)
        o = np.argsort(z['subject'])
        aux = z['aux'][o] if 'aux' in z.files else None
        out[int(f.stem[5:])] = (16 * z['logits'].mean(-1)[o], aux, z['true'][o])
    return out


def aux_score(aux):
    """aux [N, T, 2, p] -> additive term [N, 3] for lam = 1."""
    s = 16 * (aux[:, :, 1] - aux[:, :, 0]).mean(-1)              # [N, T] binary logit of each head
    if aux.shape[1] == 1:                                         # bam
        return np.stack([np.zeros(len(s)), s[:, 0], np.zeros(len(s))], 1)
    p1, p2 = 1 / (1 + np.exp(-s[:, 0])), 1 / (1 + np.exp(-s[:, 1]))
    probs = np.stack([1 - p1, np.clip(p1 - p2, 1e-4, None), p2], 1)
    return np.log(np.clip(probs, 1e-4, None))


def scores(z, aux, tau, lam):
    s = z - tau * LOGP
    return s + lam * aux_score(aux) if (aux is not None and lam) else s


def auc(a, b):
    return 100 * ((a[:, None] > b[None]).mean() + 0.5 * (a[:, None] == b[None]).mean())


def metrics(y, s):
    p = s.argmax(-1)
    m = {'acc': 100 * np.mean(y == p), 'Sen': 100 * np.mean(p[y == 0] == 0), 'Spe': 100 * np.mean(p[y > 0] != 0)}
    for c, n in enumerate(LABELS):
        m[f'rec_{n[:3]}'] = 100 * np.mean(p[y == c] == c)
    m['bal'] = np.mean([m['rec_neg'], m['rec_neu'], m['rec_pos']])
    m['AUC_nn'] = auc((s[:, 1] - s[:, 0])[y == 1], (s[:, 1] - s[:, 0])[y == 0])        # neutral vs negative
    sc = np.logaddexp(s[:, 1], s[:, 2]) - s[:, 0]                                      # scoliosis vs negative
    m['AUC_sn'] = auc(sc[y > 0], sc[y == 0])
    return m


def choose(val, lams):
    best = None
    for it, (z, aux, y) in sorted(val.items()):
        for t in TAUS:
            for lam in (lams if aux is not None else [0.0]):
                m = metrics(y, scores(z, aux, t, lam))
                key = (round(m['bal'], 6), -abs(t), -lam, it)
                if best is None or key > best[0]:
                    best = (key, it, t, lam, m['bal'])
    return best[1:]


def evaluate_run(run_dir):
    val, test = load(run_dir / 'val'), load(run_dir / 'test')
    rows = {}
    last = 16000 if 16000 in test else max(test)          # paper protocol, or the last checkpoint of a short run
    z, aux, y = test[last]
    rows['fixed'] = dict(iter=last, tau=0.0, lam=0.0, **metrics(y, scores(z, aux, 0, 0)))
    for rule, lams in [('val-tau', [0.0]), ('val-fuse', LAMS)]:
        it, t, lam, vb = choose(val, lams)
        z, aux, y = test[it]
        rows[rule] = dict(iter=it, tau=float(t), lam=lam, val_bal=vb, **metrics(y, scores(z, aux, t, lam)))
    rows['_pred'] = {r: (scores(*test[v['iter']][:2], v['tau'], v['lam']).argmax(-1) == test[v['iter']][2])
                     for r, v in rows.items()}
    rows['_true'] = test[rows['val-tau']['iter']][2]
    return rows


def threshold_free(z, y):
    s = z - z.max(-1, keepdims=True)
    sc = np.logaddexp(s[:, 1], s[:, 2]) - s[:, 0]
    return {'AUC_sn': auc(sc[y > 0], sc[y == 0]), 'AUC_nn': auc((s[:, 1] - s[:, 0])[y == 1], (s[:, 1] - s[:, 0])[y == 0])}


def beyond_accuracy(run_dir, rows):
    """Probes, learning curve and embedding separability of one run."""
    out = {}
    test = load(run_dir / 'test')
    it = rows['fixed']['iter']
    for split in ('test', 'test_mirror', 'test_deskew'):
        f = run_dir / split / f'iter_{it:05d}.npz'
        if f.exists():
            d = np.load(f)
            out[split] = threshold_free(16 * d['logits'].mean(-1), d['true'])
    f = run_dir / 'test' / f'iter_{it:05d}.npz'
    d = np.load(f)
    if 'embed' in d.files:
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.model_selection import cross_val_predict
        x = d['embed'].astype(np.float32).reshape(len(d['true']), -1)
        x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
        p = cross_val_predict(KNeighborsClassifier(5), x, d['true'], cv=10)
        out['knn_bal'] = float(np.mean([100 * np.mean(p[d['true'] == c] == c) for c in range(3)]))
    val = load(run_dir / 'val')
    curve = []
    for i in sorted(set(val) & set(test)):
        zv, av, yv = val[i]
        t = max(TAUS, key=lambda t: metrics(yv, scores(zv, None, t, 0))['bal'])
        zt, at, yt = test[i]
        m = metrics(yt, scores(zt, None, t, 0))
        curve.append({'iter': i, 'val_bal': metrics(yv, scores(zv, None, t, 0))['bal'], 'test_bal': m['bal'],
                      'test_rec_neu': m['rec_neu'], 'test_AUC_nn': m['AUC_nn'], 'tau': float(t)})
    out['curve'] = curve
    return out


def paired_bootstrap(a_rows, b_rows, rule, n=2000, seed=0):
    """Child-level paired bootstrap of the seed-averaged balanced-accuracy gain b - a on test."""
    rng = np.random.default_rng(seed)
    y = a_rows[0]['_true']
    A = np.stack([r['_pred'][rule] for r in a_rows])
    B = np.stack([r['_pred'][rule] for r in b_rows])
    gains = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        g = [100 * (B[:, i][:, y[i] == c].mean() - A[:, i][:, y[i] == c].mean()) for c in range(3)]
        gains.append(np.mean(g))
    return np.mean(gains), np.percentile(gains, 2.5), np.percentile(gains, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=str(EXP / 'runs' / 'logits'))
    ap.add_argument('--ref', default='A_base', help='variant the others are compared with')
    a = ap.parse_args()
    runs = sorted(p for p in Path(a.root).iterdir() if (p / 'val').is_dir() and (p / 'test').is_dir())
    assert runs, f'no runs under {a.root}'
    by_variant = {}
    for r in runs:
        variant = re.sub(r'_s\d+$', '', r.name)
        rows = evaluate_run(r)
        rows['_beyond'] = beyond_accuracy(r, rows)
        by_variant.setdefault(variant, []).append(rows)

    lines, summary = [], {}
    for rule in ['fixed', 'val-tau', 'val-fuse']:
        lines += [f'\n### {rule} (test, mean +- std over seeds)', '',
                  '| variant | n | ' + ' | '.join(KEYS) + ' | tau | lam | iter (k) |',
                  '|---' * (len(KEYS) + 5) + '|']
        for v, rows in by_variant.items():
            ms = [r[rule] for r in rows if rule in r]
            if not ms:
                continue
            cells = [f'{np.mean([m[k] for m in ms]):.1f} +- {np.std([m[k] for m in ms]):.1f}' for k in KEYS]
            extra = [', '.join(f'{m[k] + 0:g}' for m in ms) for k in ('tau', 'lam')] + \
                    [', '.join(f"{m['iter'] / 1000:g}" for m in ms)]
            lines.append(f'| {v} | {len(ms)} | ' + ' | '.join(cells + extra) + ' |')
            summary.setdefault(rule, {})[v] = {k: [m[k] for m in ms] for k in KEYS + ['tau', 'lam', 'iter']}
    lines.append('\nTG-MILNet (paper, 1 run, full train set): acc 89.9  Sen 99.5  Spe 86.4  neu 70.6  pos 91.2')

    if a.ref in by_variant:
        lines += [f'\n### paired bootstrap of balanced accuracy vs {a.ref} (val-fuse, seed-averaged, 95% CI)', '']
        for v, rows in by_variant.items():
            if v == a.ref or len(rows) != len(by_variant[a.ref]):
                continue
            g, lo, hi = paired_bootstrap(by_variant[a.ref], rows, 'val-fuse')
            lines.append(f'- {v}: {g:+.1f}  [{lo:+.1f}, {hi:+.1f}]')

    lines += ['\n### beyond accuracy (analysis checkpoint, mean over seeds)', '',
              'AUC_sn = scoliosis vs negative, AUC_nn = neutral vs negative, on the normal / mirrored / deskewed test set.',
              'A model that reads the image lean drops on mirror; deskew shows what is left without the global lean.', '',
              '| variant | AUC_sn | AUC_sn mirror | AUC_sn deskew | AUC_nn | AUC_nn mirror | AUC_nn deskew | 5-NN bal (embedding) |',
              '|---|---|---|---|---|---|---|---|']
    curves = {}
    for v, rows in by_variant.items():
        b = [r['_beyond'] for r in rows]
        cells = []
        for k in ('AUC_sn', 'AUC_nn'):
            for sp in ('test', 'test_mirror', 'test_deskew'):
                vals = [x[sp][k] for x in b if sp in x]
                cells.append(f'{np.mean(vals):.1f}' if vals else '-')
        knn = [x['knn_bal'] for x in b if 'knn_bal' in x]
        cells.append(f'{np.mean(knn):.1f}' if knn else '-')
        lines.append(f'| {v} | ' + ' | '.join(cells) + ' |')
        curves[v] = [x['curve'] for x in b]

    lines += ['\n### learning curve (seed-averaged test balanced accuracy / neutral recall, tau chosen on val per checkpoint)', '']
    for v, cs in curves.items():
        n = min(len(c) for c in cs)
        lines.append(f'- {v}: ' + '  '.join(
            f'{cs[0][i]["iter"] / 1000:g}k {np.mean([c[i]["test_bal"] for c in cs]):.0f}/{np.mean([c[i]["test_rec_neu"] for c in cs]):.0f}'
            for i in range(n)))
    text = '\n'.join(lines)
    print(text)
    out = EXP / 'analysis'
    out.mkdir(exist_ok=True)
    (out / 'summary.md').write_text(text, encoding='utf8')
    json.dump(summary, open(out / 'summary.json', 'w'), indent=1)
    json.dump(curves, open(out / 'curves.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
