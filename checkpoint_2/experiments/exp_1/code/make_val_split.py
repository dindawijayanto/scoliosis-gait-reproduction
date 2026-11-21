"""Carve a stratified validation set out of the official 1:1:8 train split (Scoliosis1K_118.json).

Writes two partition files (OpenGait reads only TRAIN_SET in --phase train, only TEST_SET in --phase test):
  Scoliosis1K_118_trainval.json : TRAIN_SET = train minus val, TEST_SET = val   (train + validation logits)
  Scoliosis1K_118_test.json     : TRAIN_SET = same reduced train, TEST_SET = official test (final report only)
The val set keeps the 1:1:8 ratio. Selection is seeded and saved, so every seed/model uses the SAME val set.
Usage: python make_val_split.py [--neu 15 --pos 15 --neg 120 --seed 0]
"""
import argparse, json, random
from pathlib import Path

HERE = Path(__file__).resolve().parent            # exp_1/code
EXP = HERE.parent                                   # checkpoint_2/experiments/exp_1
REPO = EXP.parents[2]
SPLIT = REPO / 'datasets' / 'Scoliosis1K' / 'Scoliosis1K_118.json'
PKL = REPO / 'datasets' / 'Scoliosis1K' / 'Scoliosis1K-sil-pkl'

ap = argparse.ArgumentParser()
ap.add_argument('--neg', type=int, default=120); ap.add_argument('--neu', type=int, default=15)
ap.add_argument('--pos', type=int, default=15); ap.add_argument('--seed', type=int, default=0)
a = ap.parse_args()

split = json.load(open(SPLIT))
label = {d.name: next(d.iterdir()).name.lower() for d in PKL.iterdir() if d.is_dir()}
train = sorted(split['TRAIN_SET'])
rng = random.Random(a.seed)
val = []
for cls, n in [('negative', a.neg), ('neutral', a.neu), ('positive', a.pos)]:
    ids = [s for s in train if label[s] == cls]
    val += rng.sample(ids, n)
    print(f'{cls:9s} train {len(ids):4d} -> val {n:3d}, remaining {len(ids) - n}')
val = sorted(val); rest = sorted(set(train) - set(val))
assert not set(val) & set(split['TEST_SET'])
json.dump({'TRAIN_SET': rest, 'TEST_SET': val}, open(EXP / 'partitions' / 'Scoliosis1K_118_trainval.json', 'w'), indent=1)
json.dump({'TRAIN_SET': rest, 'TEST_SET': sorted(split['TEST_SET'])}, open(EXP / 'partitions' / 'Scoliosis1K_118_test.json', 'w'), indent=1)
print(f'train {len(rest)}  val {len(val)}  test {len(split["TEST_SET"])}  -> written to {EXP / 'partitions'}')
