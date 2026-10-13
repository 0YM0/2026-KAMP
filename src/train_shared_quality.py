"""Rebuild D2 features or train the fixed 48-epoch shared-quality model."""
import argparse
import time
import numpy as np
import torch
from torch import nn
from PIL import Image
from run import ROOT, read, save, rows, m, sha
from shared_quality import Quality, variants, component_features


def build_features(output):
    cases = read(ROOT / 'data/shared_quality_oof301.json')
    byid = {r['image_id']: r for r in rows('train')}
    assert len(cases) == 301 and {c['image_id'] for c in cases} <= set(byid)
    features, targets, identity = [], [], []
    offset = 0
    for ci, case in enumerate(cases):
        row = byid[case['image_id']]
        gt = m.targets(row)
        np.testing.assert_allclose(gt, case['targets'], atol=1e-6)
        gray = Image.open(row['image_path']).convert('L')
        for proposal in case['predictions']:
            candidates = variants(proposal, gray.size)
            features.append(component_features(gray, proposal, candidates))
            gi = int(np.argmax(m.iou(proposal[:4], gt)))
            targets.extend(float(m.iou(q[:4], [gt[gi]])[0]) for q in candidates)
            identity.extend([offset] * len(candidates))
            offset += len(candidates)
        if (ci + 1) % 60 == 0:
            print('features', ci + 1, flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, peak=np.concatenate(features), bound=np.asarray(targets, np.float32), identity=np.asarray(identity, int))
    print('saved', output, offset, flush=True)


def fit(features, output):
    assert not output.exists(), 'Choose a new --output directory to preserve previous training.'
    output.mkdir(parents=True)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    data = np.load(features)
    x, y, identity = data['peak'], data['bound'], data['identity']
    assert x.shape == (86913, 84) and len(y) == len(identity) == len(x)
    mean = x.mean(0)
    std = np.maximum(x.std(0), .1)
    tx = torch.from_numpy(np.clip((x - mean) / std, -8, 8))
    ox = torch.from_numpy(np.clip((x[identity] - mean) / std, -8, 8))
    target = torch.from_numpy(y)
    delta = target - torch.from_numpy(y[identity])
    torch.manual_seed(20261008)
    model = Quality(84)
    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    history = []
    start = time.monotonic()
    for epoch in range(1, 49):
        order = torch.randperm(len(x))
        total = 0
        for off in range(0, len(x), 4096):
            ids = order[off:off + 4096]
            opt.zero_grad()
            q = model(tx[ids])
            dq = q - model(ox[ids])
            loss = nn.functional.smooth_l1_loss(q, target[ids], beta=.1) + nn.functional.smooth_l1_loss(dq, delta[ids], beta=.1)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            opt.step()
            total += loss.item() * len(ids)
        history.append(dict(epoch=epoch, fit_loss=total / len(x)))
        if epoch % 12 == 0:
            print(history[-1], flush=True)
    torch.save(dict(state=model.state_dict(), mean=mean.tolist(), std=std.tolist(), features=84), output / 'quality.pt')
    save(output / 'history.json', history)
    save(output / 'status.json', dict(epochs=48, fit_images=301, fit_candidates=len(x), seconds=time.monotonic() - start, checkpoint_sha256=sha(output / 'quality.pt'), last_epoch_fixed=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['features', 'train'])
    parser.add_argument('--features', default='data/shared_quality_features.npz')
    parser.add_argument('--output', default='retraining/shared_quality')
    args = parser.parse_args()
    if args.stage == 'features':
        build_features(ROOT / args.features)
    else:
        fit(ROOT / args.features, ROOT / args.output)
