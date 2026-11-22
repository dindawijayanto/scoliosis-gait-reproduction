"""Shared code for the checkpoint_2 self_learning notebooks.

Everything in here was already explained in checkpoint_1 (data loading, ResNet9, ScoNet head,
triplet + cross entropy, the training loop). The NEW parts of each notebook (strip weights,
temporal attention, fusion, BAM, ...) are written inside the notebook itself, so I can read them
next to the explanation.

Sections:
  1. paths and labels
  2. data: split json, picking children, loading .pkl, transform, batches
  3. saved logits of my full ScoNet-MT runs (checkpoint_1)
  4. metrics (paper convention)
  5. model pieces (ResNet9, SeparateFCs, SeparateBNNecks, horizontal pooling)
  6. losses (triplet, cross entropy)
  7. mini training loop + evaluation
"""
import json
import os
import pickle
import random
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------------------------
# 1. paths and labels
# ---------------------------------------------------------------------------------------------
TOOLS = Path(__file__).resolve().parent
SL = TOOLS.parent                                    # checkpoint_2/self_learning
REPO = SL.parents[1]                                 # OpenGait
DATA = REPO / 'datasets' / 'Scoliosis1K'
PKL_ROOT = DATA / 'Scoliosis1K-sil-pkl'
SPLITS = {r: DATA / f'Scoliosis1K_{r}.json' for r in ['112', '114', '118', '1116']}
RESULT = REPO / 'checkpoint_1' / 'reproduce' / 'result_sconet'
LOGIT_DIRS = {'official': RESULT / 'ScoNet_MT' / 'local_check',
              'fromraw': RESULT / 'result_fromraw' / 'local_check' / 'fromraw_on_official'}
SCONET_16K = RESULT / 'ScoNet_MT' / 'output' / 'Scoliosis1K' / 'ScoNet' / 'ScoNet' / 'checkpoints' / 'ScoNet-16000.pt'
GAITBASE_ZIP = REPO / 'output' / 'pretrained_casiab_gaitbase.zip'
GAITBASE_PT = TOOLS / 'pretrained' / 'GaitBase_DA-60000.pt'

LABEL_NAMES = ['negative', 'neutral', 'positive']
LABELS = {name: i for i, name in enumerate(LABEL_NAMES)}
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------------------------
# 2. data
# ---------------------------------------------------------------------------------------------
def load_split(ratio='118'):
    with open(SPLITS[ratio]) as f:
        return json.load(f)


def subject_class(subject):
    """The class is the folder name inside the subject folder: <subject>/<class>/000_180/000_180.pkl"""
    return os.listdir(PKL_ROOT / subject)[0]


def children_by_class(subjects):
    by_class = {c: [] for c in LABEL_NAMES}
    for s in sorted(subjects):
        by_class[subject_class(s)].append(s)
    return by_class


def pick_children(subjects, counts, seed=0):
    """counts = {'negative': n, 'neutral': n, 'positive': n}. Random but fixed by seed."""
    by_class = children_by_class(subjects)
    rng = random.Random(seed)
    return {c: sorted(rng.sample(by_class[c], counts[c])) for c in LABEL_NAMES}


def load_seq(subject, cls=None):
    cls = cls or subject_class(subject)
    with open(PKL_ROOT / subject / cls / '000_180' / '000_180.pkl', 'rb') as f:
        return np.asarray(pickle.load(f))


def load_items(picked):
    """One dict per child: subject, class, label, pid (0..n-1), seq uint8 (frames, 64, 64)."""
    items = []
    for cls in LABEL_NAMES:
        for s in picked[cls]:
            items.append({'subject': s, 'class': cls, 'label': LABELS[cls], 'pid': len(items),
                          'seq': load_seq(s, cls)})
    return items


def sil_transform(seq):
    """BaseSilCuttingTransform: cut 10 columns each side (64 -> 44), / 255."""
    return seq[..., 10:-10].astype(np.float32) / 255.0


def make_batch(items, n_per_class=4, copies=2, frames=20, balanced=True, n_children=12):
    """balanced=True : n_per_class children of every class (like DiseaseTripletSampler).
       balanced=False: n_children random children, whatever class (like OpenGait TripletSampler).
       Every child is used `copies` times with different random frames (needed for the MT triplet)."""
    if balanced:
        chosen = []
        for c in range(3):
            pool = [it for it in items if it['label'] == c]
            chosen += random.sample(pool, n_per_class)
    else:
        chosen = random.sample(items, n_children)
    seqs, pids, labels = [], [], []
    for it in chosen:
        for _ in range(copies):
            idx = np.random.choice(len(it['seq']), frames, replace=len(it['seq']) < frames)
            seqs.append(sil_transform(it['seq'][idx]))
            pids.append(it['pid'])
            labels.append(it['label'])
    return (torch.tensor(np.stack(seqs)), torch.tensor(pids), torch.tensor(labels))


# ---------------------------------------------------------------------------------------------
# 3. saved logits of my full runs (checkpoint_1, full 1:1:8 test set, 749 children)
# ---------------------------------------------------------------------------------------------
def load_saved_logits(run='official', it=16000):
    """logits [749, 3, 16] (class x strip, BNNeck cosines), true [749], subject [749]."""
    d = np.load(LOGIT_DIRS[run] / f'iter_{it:05d}.npz')
    return d['logits'].astype(np.float64), d['true'].astype(int), d['subject']


# ---------------------------------------------------------------------------------------------
# 4. metrics
# ---------------------------------------------------------------------------------------------
def metrics(y, p):
    """Paper convention (ScoNet-MT, TG-MILNet, and ScoliGaitX's note under its Table 3):
       Sen = recall of NEGATIVE, Spe = share of neutral + positive NOT called negative.
       Plus per-class recall and balanced accuracy (mean of the 3 recalls)."""
    y, p = np.asarray(y), np.asarray(p)
    out = {'n': len(y), 'acc': 100 * (y == p).mean(),
           'Sen': 100 * (p[y == 0] == 0).mean(),
           'Spe': 100 * (p[y > 0] != 0).mean()}
    for c, name in enumerate(LABEL_NAMES):
        out['rec_' + name[:3]] = 100 * (p[y == c] == c).mean() if (y == c).any() else np.nan
    out['bal'] = np.nanmean([out['rec_neg'], out['rec_neu'], out['rec_pos']])
    return out


def print_metrics(rows, title=None):
    """rows = list of (name, metrics dict)."""
    keys = ['acc', 'Sen', 'Spe', 'rec_neg', 'rec_neu', 'rec_pos', 'bal']
    if title:
        print(title)
    print(f'{"":30s}' + ''.join(f'{k:>9s}' for k in keys))
    for name, m in rows:
        print(f'{name:30s}' + ''.join(f'{m[k]:9.1f}' for k in keys))


def auc(pos_scores, neg_scores):
    """Chance that a random 'pos' item scores higher than a random 'neg' item (ties count 0.5).
       Same as the loop version in notebook 03 of the old checkpoint, written with numpy broadcasting."""
    a = np.asarray(pos_scores)[:, None]
    b = np.asarray(neg_scores)[None, :]
    return float((a > b).mean() + 0.5 * (a == b).mean())


# ---------------------------------------------------------------------------------------------
# 5. model pieces (same as checkpoint_1 / OpenGait)
# ---------------------------------------------------------------------------------------------
class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride, bias=False), nn.BatchNorm2d(out_ch))

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity)


class ResNet9(nn.Module):
    """OpenGait ResNet9: channels 64-128-256-512, strides 1-2-2-1. Input [N, in_ch, 64, 44] -> [N, 512, 16, 11]."""
    def __init__(self, in_ch=1, channels=(64, 128, 256, 512), strides=(1, 2, 2, 1)):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, channels[0], 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels[0])
        self.layer1 = BasicBlock(channels[0], channels[0], strides[0])
        self.layer2 = BasicBlock(channels[0], channels[1], strides[1])
        self.layer3 = BasicBlock(channels[1], channels[2], strides[2])
        self.layer4 = BasicBlock(channels[2], channels[3], strides[3])

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        return self.layer4(self.layer3(self.layer2(self.layer1(x))))


def encode_frames(backbone, sils):
    """Every frame alone through the backbone (SetBlockWrapper).
       sils [n, s, h, w] (or [n, s, c, h, w]) -> [n, s, 512, 16, 11]"""
    if sils.dim() == 4:
        sils = sils.unsqueeze(2)
    n, s = sils.shape[:2]
    x = backbone(sils.flatten(0, 1))
    return x.view(n, s, *x.shape[1:])


def horizontal_pooling(x, bins=16):
    """[n, c, h, w] -> [n, c, bins]: mean + max over every horizontal strip."""
    n, c = x.shape[:2]
    z = x.reshape(n, c, bins, -1)
    return z.mean(-1) + z.max(-1)[0]


class SeparateFCs(nn.Module):
    """Every strip its own linear layer: [n, c_in, p] -> [n, c_out, p]."""
    def __init__(self, parts, in_ch, out_ch):
        super().__init__()
        self.fc_bin = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(parts, in_ch, out_ch)))

    def forward(self, x):
        return x.permute(2, 0, 1).matmul(self.fc_bin).permute(1, 2, 0).contiguous()


class SeparateBNNecks(nn.Module):
    """BN over all c*p numbers, then per strip: cosine between the feature and every class weight.
       [n, c, p] -> logits [n, classes, p] (between -1 and 1)."""
    def __init__(self, parts, in_ch, class_num):
        super().__init__()
        self.fc_bin = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(parts, in_ch, class_num)))
        self.bn1d = nn.BatchNorm1d(in_ch * parts)

    def forward(self, x):
        n, c, p = x.shape
        x = self.bn1d(x.reshape(n, -1)).reshape(n, c, p)
        feature = F.normalize(x.permute(2, 0, 1), dim=-1)
        logits = feature.matmul(F.normalize(self.fc_bin, dim=1))
        return logits.permute(1, 2, 0).contiguous()


def init_parameters(model):
    """Same as OpenGait BaseModel.init_parameters."""
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.Conv1d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.BatchNorm3d)) and m.affine:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)


def _opengait_to_mine(key):
    """'Backbone.forward_block.conv1.conv.weight' -> 'conv1.weight', 'layer1.0.conv1.weight' -> 'layer1.conv1.weight'"""
    k = key.replace('Backbone.forward_block.', '').replace('conv1.conv.', 'conv1.')
    for i in range(1, 5):
        k = k.replace(f'layer{i}.0.', f'layer{i}.')
    return k


def load_gaitbase_backbone(backbone):
    """Copy the ResNet9 weights of GaitBase trained for gait RECOGNITION on CASIA-B (OpenGait model zoo).
       It never saw Scoliosis1K, so it is a fair starting point; it only knows 'what a walking person looks like'.
       Only works for in_ch = 1."""
    if not GAITBASE_PT.exists():
        GAITBASE_PT.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(GAITBASE_ZIP) as z:
            name = [n for n in z.namelist() if n.endswith('.pt')][0]
            GAITBASE_PT.write_bytes(z.read(name))
    sd = torch.load(GAITBASE_PT, map_location='cpu', weights_only=False)['model']
    mine = {_opengait_to_mine(k): v for k, v in sd.items() if k.startswith('Backbone.')}
    missing, unexpected = backbone.load_state_dict(mine, strict=False)
    missing = [m for m in missing if not m.endswith('num_batches_tracked')]
    return missing, unexpected


class ScoNetMT(nn.Module):
    """The baseline from checkpoint_1: ResNet9 -> max over frames -> 16 strips -> SeparateFCs -> BNNeck.
       forward returns a dict, so every notebook can add more outputs."""
    def __init__(self, parts=16, embed_dim=256, num_classes=3):
        super().__init__()
        self.backbone = ResNet9()
        self.fcs = SeparateFCs(parts, 512, embed_dim)
        self.bnnecks = SeparateBNNecks(parts, embed_dim, num_classes)
        init_parameters(self)

    def forward(self, sils):
        f = encode_frames(self.backbone, sils)          # [n, s, 512, 16, 11]
        f = f.max(1)[0]                                 # temporal max pooling
        strips = horizontal_pooling(f)                  # [n, 512, 16]
        embed = self.fcs(strips)                        # [n, 256, 16]
        logits = self.bnnecks(embed)                    # [n, 3, 16]
        return {'embed': embed, 'logits': logits}


def load_sconet_16k():
    """My full ScoNet-MT run from checkpoint_1 (official pkl, 16000 iterations). For analysis only."""
    sd = torch.load(SCONET_16K, map_location='cpu', weights_only=False)['model']
    model = ScoNetMT()
    mine = {}
    for k, v in sd.items():
        if k.startswith('Backbone.'):
            mine['backbone.' + _opengait_to_mine(k)] = v
        elif k.startswith('FCs.'):
            mine[k.replace('FCs.', 'fcs.')] = v
        elif k.startswith('BNNecks.'):
            mine[k.replace('BNNecks.', 'bnnecks.')] = v
    model.load_state_dict(mine)
    return model.to(DEVICE).eval()


# ---------------------------------------------------------------------------------------------
# 6. losses
# ---------------------------------------------------------------------------------------------
def triplet_loss(embed, ids, margin=0.2):
    """OpenGait TripletLoss, per strip, averaged over the non-zero triplets. embed [n, c, p]."""
    x = embed.permute(2, 0, 1).float()
    x2 = (x ** 2).sum(-1)
    dist = torch.sqrt(F.relu(x2.unsqueeze(2) + x2.unsqueeze(1) - 2 * x.matmul(x.transpose(1, 2))) + 1e-12)
    same = ids.unsqueeze(1) == ids.unsqueeze(0)
    p, n, _ = dist.shape
    ap = dist[:, same].view(p, n, -1, 1)
    an = dist[:, ~same].view(p, n, 1, -1)
    loss = F.relu(ap - an + margin).view(p, -1)
    num = (loss != 0).sum(-1).float()
    avg = loss.sum(-1) / (num + 1e-9)
    avg[num == 0] = 0
    return avg.mean()


def ce_loss(logits, labels, scale=16, eps=0.1, weight=None):
    """Cross entropy on every strip with the same label. logits [n, 3, p] or [n, 3]."""
    if logits.dim() == 2:
        logits = logits.unsqueeze(-1)
    p = logits.shape[-1]
    return F.cross_entropy(logits.float() * scale, labels.unsqueeze(1).repeat(1, p),
                           label_smoothing=eps, weight=weight)


def default_loss(out, labels, pids):
    """ScoNet-MT: triplet on child id + cross entropy on class."""
    return triplet_loss(out['embed'], pids) + ce_loss(out['logits'], labels)


# ---------------------------------------------------------------------------------------------
# 7. mini training + evaluation
# ---------------------------------------------------------------------------------------------
MINI = {'iters': 500, 'lr': 1e-4, 'head_lr': 1e-3, 'weight_decay': 5e-4, 'frames': 16, 'n_per_class': 3,
        'copies': 2, 'milestones': [350, 450], 'test_frames': 300, 'test_clips': 10}


def train_mini(model, items, loss_fn=default_loss, cfg=None, balanced=True, seed=0, log_every=100, verbose=True):
    """Adam (ScoliGaitX uses Adam 1e-4; SGD 0.1 blew up in checkpoint_1), fp16, MultiStepLR.
       The pretrained backbone(s) learn with lr 1e-4, everything new after it (head, strip weights, attention,
       fusion) with head_lr 1e-3: with 1e-4 a new module barely moves in 500 iterations.
       Returns the training history."""
    cfg = {**MINI, **(cfg or {})}
    set_seed(seed)
    model = model.to(DEVICE).train()
    backbone = [p for n, p in model.named_parameters() if n.startswith('backbone')]
    rest = [p for n, p in model.named_parameters() if not n.startswith('backbone')]
    opt = torch.optim.Adam([{'params': backbone, 'lr': cfg['lr']}, {'params': rest, 'lr': cfg['head_lr']}],
                           weight_decay=cfg['weight_decay'])
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, cfg['milestones'], 0.1)
    scaler = torch.amp.GradScaler('cuda', enabled=DEVICE == 'cuda')
    hist = {'loss': []}
    t0 = time.time()
    for it in range(1, cfg['iters'] + 1):
        sils, pids, labels = make_batch(items, cfg['n_per_class'], cfg['copies'], cfg['frames'], balanced,
                                        n_children=3 * cfg['n_per_class'])
        sils, pids, labels = sils.to(DEVICE), pids.to(DEVICE), labels.to(DEVICE)
        with torch.autocast(device_type=DEVICE, enabled=DEVICE == 'cuda'):
            out = model(sils)
        loss = loss_fn(out, labels, pids)
        if not torch.isfinite(loss):
            raise RuntimeError(f'loss became {loss.item()} at iteration {it}')
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        hist['loss'].append(loss.item())
        if verbose and it % log_every == 0:
            print(f'iter {it:4d} | loss {np.mean(hist["loss"][-log_every:]):.4f} | {time.time() - t0:.0f}s')
    return hist


@torch.no_grad()
def predict_mini(model, items, out_key='logits', n_clips=None, clip_frames=None, n_frames=None, seed=0):
    """Test like training: n_clips random clips of clip_frames frames per child (the same number of frames as in
       training), and the class scores are averaged over the clips. Why: the model is trained on 16 frames; with
       all 300 frames the max over frames gives much bigger values than it ever saw in training, and the mini
       models then called almost every child 'negative' (notebook 03, first version).
       raw = the outputs of one pass with ALL frames (max n_frames, in order), only for looking inside the model
       (attention weights, gates).
       Returns y [n], scores [n, 3] (= logits averaged over strips, the ScoNet decision), raw."""
    n_clips = n_clips or MINI['test_clips']
    clip_frames = clip_frames or MINI['frames']
    n_frames = n_frames or MINI['test_frames']
    model = model.to(DEVICE).eval()
    rng = np.random.default_rng(seed)
    ys, scores, raw = [], [], []
    for it in items:
        seq = it['seq']
        clips = np.stack([sil_transform(seq[np.sort(rng.choice(len(seq), clip_frames, replace=False))])
                          for _ in range(n_clips)])
        with torch.autocast(device_type=DEVICE, enabled=DEVICE == 'cuda'):
            out = model(torch.tensor(clips).to(DEVICE))
        z = out[out_key].float().cpu().numpy()
        z = z.mean(-1) if z.ndim == 3 else z                     # [clips, 3]
        scores.append(z.mean(0))
        x = torch.tensor(sil_transform(seq[:n_frames])).unsqueeze(0).to(DEVICE)
        with torch.autocast(device_type=DEVICE, enabled=DEVICE == 'cuda'):
            out = model(x)
        raw.append({k: v.float().cpu().numpy()[0] for k, v in out.items() if torch.is_tensor(v)})
        ys.append(it['label'])
    return np.array(ys), np.array(scores), raw
