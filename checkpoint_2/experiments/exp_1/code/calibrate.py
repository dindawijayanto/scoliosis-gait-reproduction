"""Pick checkpoint + decision threshold on the held-out VAL split only, then report once on the official TEST split.

Input: val_logits/<save_name>/{val,test}/iter_XXXXX.npz  (from export_logits.py)
Score: z = 16 * mean-over-strips logits (ScoNet CE scale). Logit adjustment: z_c - tau * log(prior_c),
prior = class frequencies of the REDUCED train set (476 / 59 / 59).
Rules (all chosen on val, never on test; ties -> smaller tau, then later checkpoint):
  fixed16k      argmax at 16k (paper protocol, no val use)
  val-iter      argmax, checkpoint chosen by val balanced accuracy
  val-tau       checkpoint + tau chosen by val balanced accuracy
  val-tau@Sen97 checkpoint + tau maximising val balanced acc subject to val Sen >= 97 (screening constraint)
Usage: python calibrate.py [--root val_logits]      |  python calibrate.py --smoke  (pipeline check on the old run)
"""
import argparse, json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent            # exp_1/code
EXP = HERE.parent                                   # checkpoint_2/experiments/exp_1
LABELS = ['negative', 'neutral', 'positive']
TAUS = np.round(np.arange(0, 1.501, 0.05), 2)

def load(d):
    out = {}
    for f in sorted(Path(d).glob('iter_*.npz')):
        z = np.load(f); o = np.argsort(z['subject'])
        out[int(f.stem[5:])] = (16 * z['logits'].mean(-1)[o], z['true'][o])
    return out

def metrics(y, p, s=None):
    m = {'acc': 100 * np.mean(y == p), 'Sen': 100 * np.mean(p[y == 0] == 0), 'Spe': 100 * np.mean(p[y > 0] != 0)}
    for c, n in enumerate(LABELS): m[f'rec_{n[:3]}'] = 100 * np.mean(p[y == c] == c)
    m['bal'] = np.mean([m['rec_neg'], m['rec_neu'], m['rec_pos']])
    if s is not None:   # AUC neutral vs negative on (z_neu - z_neg); threshold-free
        a, b = s[y == 1], s[y == 0]; m['AUC_nn'] = 100 * ((a[:, None] > b[None]).mean() + 0.5 * (a[:, None] == b[None]).mean())
    return m

def predict(z, tau, logp): return (z - tau * logp).argmax(-1)

def choose(val, logp, taus, constraint=None):
    best = None
    for it in sorted(val):
        z, y = val[it]
        for t in taus:
            m = metrics(y, predict(z, t, logp))
            if constraint and m['Sen'] < constraint: continue
            key = (round(m['bal'], 6), -t, it)
            if best is None or key > best[0]: best = (key, it, t, m)
    return best

def run(name, val, test, logp):
    rows = {}
    z, y = test[16000]; rows['fixed16k'] = (16000, 0.0, metrics(y, predict(z, 0, logp), z[:, 1] - z[:, 0]))
    for rule, taus, con in [('val-iter', [0.0], None), ('val-tau', TAUS, None), ('val-tau@Sen97', TAUS, 97.0)]:
        b = choose(val, logp, taus, con)
        if b is None: rows[rule] = None; continue
        _, it, t, mv = b; z, y = test[it]
        rows[rule] = (it, t, metrics(y, predict(z, t, logp), z[:, 1] - z[:, 0]), mv)
    print(f'\n== {name}')
    keys = ['acc', 'Sen', 'Spe', 'rec_neg', 'rec_neu', 'rec_pos', 'bal', 'AUC_nn']
    print(f'{"rule":15s}{"iter":>6s}{"tau":>6s}' + ''.join(f'{k:>9s}' for k in keys) + f'{"val bal":>9s}')
    for r, v in rows.items():
        if v is None: print(f'{r:15s}  (no checkpoint meets the constraint on val)'); continue
        it, t, m = v[:3]; vb = f'{v[3]["bal"]:9.1f}' if len(v) > 3 else f'{"-":>9s}'
        print(f'{r:15s}{it//1000:5d}k{t:6.2f}' + ''.join(f'{m[k]:9.1f}' for k in keys) + vb)
    return rows

def paired_bootstrap(test_by_seed, rows_by_seed, a='fixed16k', b='val-tau', n=2000, seed=0):
    """Child-level paired bootstrap of the seed-averaged accuracy / neutral-recall gain b - a on the test set."""
    rng = np.random.default_rng(seed); preds = {}
    for s, rows in rows_by_seed.items():
        for r in (a, b):
            it, t = rows[r][:2]; z, y = test_by_seed[s][it]; preds[(s, r)] = (predict(z, t, LOGP) == y)
    y = next(iter(test_by_seed.values()))[16000][1]; N = len(y); gains, ngains = [], []
    for _ in range(n):
        i = rng.integers(0, N, N); g = np.mean([preds[(s, b)][i].mean() - preds[(s, a)][i].mean() for s in rows_by_seed])
        neu = i[y[i] == 1]; ng = np.mean([preds[(s, b)][neu].mean() - preds[(s, a)][neu].mean() for s in rows_by_seed])
        gains.append(100 * g); ngains.append(100 * ng)
    for lab, g in [('acc', gains), ('neutral recall', ngains)]:
        print(f'  {b} - {a} {lab}: {np.mean(g):+.1f}  95% CI [{np.percentile(g, 2.5):+.1f}, {np.percentile(g, 97.5):+.1f}]')

ap = argparse.ArgumentParser(); ap.add_argument('--root', default=str(EXP / 'runs' / 'val_logits')); ap.add_argument('--smoke', action='store_true')
a = ap.parse_args()
counts = np.array([476, 59, 59]); LOGP = np.log(counts / counts.sum())

if a.smoke:   # old full-train run, test used as "val": only checks the code reproduces known numbers (tau 0.5 -> 92.0 acc)
    old = load(EXP.parents[2] / 'checkpoint_1/reproduce/result_sconet/ScoNet_MT/local_check')
    full = np.array([596, 74, 74]); lp = np.log(full / full.sum())
    z, y = old[16000]
    for t in (0, 0.1, 0.3, 0.5):
        m = metrics(y, predict(z, t, lp)); print(f'smoke tau {t}: acc {m["acc"]:.1f} Sen {m["Sen"]:.1f} Spe {m["Spe"]:.1f} neu {m["rec_neu"]:.1f}')
    LOGP = lp; run('smoke (test used as val - NOT a result)', old, old, lp); raise SystemExit

runs = sorted(p for p in Path(a.root).iterdir() if (p / 'val').is_dir())
assert runs, f'no runs under {a.root}'
tests, allrows, summary = {}, {}, {}
for r in runs:
    val, test = load(r / 'val'), load(r / 'test')
    assert len(next(iter(val.values()))[1]) == 150 and len(test[16000][1]) == 749
    tests[r.name] = test; allrows[r.name] = run(r.name, val, test, LOGP)

print('\n== mean +- std over seeds (test)')
keys = ['acc', 'Sen', 'Spe', 'rec_neu', 'rec_pos', 'AUC_nn']
print(f'{"rule":15s}' + ''.join(f'{k:>14s}' for k in keys))
for rule in ['fixed16k', 'val-iter', 'val-tau', 'val-tau@Sen97']:
    ms = [allrows[s][rule][2] for s in allrows if allrows[s][rule]]
    if not ms: continue
    print(f'{rule:15s}' + ''.join(f'{np.mean([m[k] for m in ms]):8.1f} +-{np.std([m[k] for m in ms]):4.1f}' for k in keys) + f'   (n={len(ms)})')
    summary[rule] = {k: [m[k] for m in ms] for k in keys}
print('TG-MILNet (paper, 1 run)      acc 89.9  Sen 99.5  Spe 86.4  neu 70.6  pos 91.2')
print('\n== paired bootstrap over test children (seed-averaged)')
paired_bootstrap(tests, allrows)
(EXP / 'analysis').mkdir(exist_ok=True)
json.dump(summary, open(EXP / 'analysis' / 'calibration_summary.json', 'w'), indent=1)
