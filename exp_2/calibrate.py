"""Choose checkpoint, tau and aux-head fusion on VAL only, then report once on the official TEST split.

Input: exp_2/logits/<run>/{val,test}/iter_XXXXX.npz (from export_logits.py)
Score: z = 16 * mean-over-strips logits. Logit adjustment at test: z_c - tau * log(prior_c),
prior = reduced train set (476 / 59 / 59). LA-trained models already target a balanced prior, so tau
may be negative; the grid is -0.5..1.5.
Aux fusion (only runs with a head), weight lam chosen on val:
  bam     : z_neu += lam * 16 * mean(b_neu - b_rest)
  ordinal : z     += lam * log P_ord(y), P_ord from the two cumulative heads
Rules (ties -> smaller |tau|, smaller lam, later checkpoint):
  fixed16k   argmax at 16k, no val use (paper protocol)
  val-tau    checkpoint + tau chosen by val balanced accuracy
  val-fuse   checkpoint + tau + lam chosen by val balanced accuracy (= val-tau for runs without a head)
Usage: python exp_2/calibrate.py   ->  prints per-run and per-variant tables, writes exp_2/results/summary.{json,md}
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
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
    if 16000 in test:
        z, aux, y = test[16000]
        rows['fixed16k'] = dict(iter=16000, tau=0.0, lam=0.0, **metrics(y, scores(z, aux, 0, 0)))
    for rule, lams in [('val-tau', [0.0]), ('val-fuse', LAMS)]:
        it, t, lam, vb = choose(val, lams)
        z, aux, y = test[it]
        rows[rule] = dict(iter=it, tau=float(t), lam=lam, val_bal=vb, **metrics(y, scores(z, aux, t, lam)))
    rows['_pred'] = {r: (scores(*test[v['iter']][:2], v['tau'], v['lam']).argmax(-1) == test[v['iter']][2])
                     for r, v in rows.items()}
    rows['_true'] = test[rows['val-tau']['iter']][2]
    return rows


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
    ap.add_argument('--root', default=str(HERE / 'logits'))
    ap.add_argument('--ref', default='A_base', help='variant the others are compared with')
    a = ap.parse_args()
    runs = sorted(p for p in Path(a.root).iterdir() if (p / 'val').is_dir() and (p / 'test').is_dir())
    assert runs, f'no runs under {a.root}'
    by_variant = {}
    for r in runs:
        variant = re.sub(r'_s\d+$', '', r.name)
        by_variant.setdefault(variant, []).append(evaluate_run(r))

    lines, summary = [], {}
    for rule in ['fixed16k', 'val-tau', 'val-fuse']:
        lines += [f'\n### {rule} (test, mean +- std over seeds)', '',
                  '| variant | n | ' + ' | '.join(KEYS) + ' | tau | lam | iter (k) |',
                  '|---' * (len(KEYS) + 5) + '|']
        for v, rows in by_variant.items():
            ms = [r[rule] for r in rows if rule in r]
            if not ms:
                continue
            cells = [f'{np.mean([m[k] for m in ms]):.1f} +- {np.std([m[k] for m in ms]):.1f}' for k in KEYS]
            extra = [', '.join(f'{m[k] + 0:g}' for m in ms) for k in ('tau', 'lam')] + \
                    [', '.join(str(m['iter'] // 1000) for m in ms)]
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
    text = '\n'.join(lines)
    print(text)
    out = HERE / 'results'
    out.mkdir(exist_ok=True)
    (out / 'summary.md').write_text(text, encoding='utf8')
    json.dump(summary, open(out / 'summary.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
