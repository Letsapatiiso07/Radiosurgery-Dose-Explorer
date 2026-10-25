"""Train the coarse or the fine tumour model on the training patients.

    python -m radiosurgery.seg.train coarse [--threads 4]
    python -m radiosurgery.seg.train fine [--threads 8]

Each step trains on a small batch of patches cut from random training
patients. Most patches are centred near the tumour (it fills well under 1%
of the head, so random patches would rarely show it); the rest come from
around it and from anywhere in the head, so the model also learns what is
not tumour. Patches are flipped left to right (the tumour can be on either
side), sometimes rotated and scaled in-plane, and their brightness and
contrast are jittered.

The loss is cross-entropy plus soft Dice. Cross-entropy scores every voxel;
Dice scores the overlap and so does not let the vast background drown the
tumour.

Every random draw comes from a generator seeded by the step number, so a run
that stops can resume from its last checkpoint and continue the same way.
Checkpoints and the log are written to models/seg/<stage>/ (not in git).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import data, splits
from .infer import sliding_window
from .unet import UNet

MODELS = Path("models") / "seg"
SEED = 2026

STAGES = {
    "coarse": dict(channels=(8, 16, 32, 64), kernels=((3, 3, 3),) * 4, pools=((2, 2, 2),) * 3,
                   patch=(48, 80, 80), batch=2, steps=3000, lr=1e-3, p_tumour=0.5, p_near=0.0, near_mm=0,
                   val_every=500),
    "fine": dict(channels=(16, 32, 64, 128), kernels=((1, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)),
                 pools=((1, 2, 2), (2, 2, 2), (2, 2, 2)), patch=(24, 112, 112), batch=2, steps=12000, lr=1e-3,
                 p_tumour=0.6, p_near=0.25, near_mm=30, val_every=500),
}


def build(stage):
    c = STAGES[stage]
    return UNet(c["channels"], c["kernels"], c["pools"])


class Patches:
    """Random training patches with their soft labels, from memory-mapped caches."""

    def __init__(self, patients, kind, cfg):
        self.kind, self.cfg = kind, cfg
        self.spacing = np.array(data.FINE if kind == "fine" else data.COARSE)
        self.items = []
        for p in patients:
            img, lbl = data.load_cached(p, kind)
            m = data.meta(p)
            tumour = np.argwhere(np.asarray(lbl) >= 128)
            ci, _ = data.load_cached(p, "coarse", mmap=False)
            head = np.argwhere(ci > ci.min() + 0.5)  # bounding box of the head, from the small coarse image
            ca, fa = data.axes_of(m, "coarse"), data.axes_of(m, kind)
            lo = [np.searchsorted(f, c[i]) for f, c, i in zip(fa, ca, head.min(0))]
            hi = [np.searchsorted(f, c[i]) for f, c, i in zip(fa, ca, head.max(0))]
            self.items.append({"patient": p, "img": img, "lbl": lbl, "tumour": tumour, "lo": np.array(lo),
                               "hi": np.array(hi), "pad": float(ci.min())})

    def centre(self, item, rng):
        patch = np.array(self.cfg["patch"])
        u = rng.random()
        if u < self.cfg["p_tumour"] and len(item["tumour"]):
            c = item["tumour"][rng.integers(len(item["tumour"]))]
            return c + rng.integers(-(patch // 3), patch // 3 + 1)  # the tumour anywhere in the middle of the patch
        if u < self.cfg["p_tumour"] + self.cfg["p_near"] and len(item["tumour"]):
            reach = (self.cfg["near_mm"] / self.spacing).astype(int)
            return item["tumour"].mean(0).astype(int) + rng.integers(-reach, reach + 1)
        return np.array([rng.integers(a, max(a + 1, b)) for a, b in zip(item["lo"], item["hi"])])

    def crop(self, arr, centre, size, pad):
        lo = centre - size // 2
        src = tuple(slice(max(l, 0), min(l + s, n)) for l, s, n in zip(lo, size, arr.shape))
        out = np.full(size, pad, np.float32)
        dst = tuple(slice(s.start - l, s.stop - l) for s, l in zip(src, lo))
        out[dst] = arr[src]
        return out

    def sample(self, rng):
        item = self.items[rng.integers(len(self.items))]
        patch = np.array(self.cfg["patch"])
        c = self.centre(item, rng)
        spatial = rng.random() < 0.3
        size = patch.copy()
        if spatial:  # room to rotate up to 15 degrees and zoom out up to 15% in-plane
            size[1:] = np.ceil(patch[1:] * 1.45).astype(int) // 2 * 2
        img = self.crop(item["img"], c, size, item["pad"])
        lbl = self.crop(item["lbl"], c, size, 0) / 255.0
        img, lbl = torch.from_numpy(img)[None, None], torch.from_numpy(lbl.astype(np.float32))[None, None]
        if spatial:
            a, s = np.deg2rad(rng.uniform(-15, 15)), rng.uniform(0.85, 1.15)
            r = patch[1] / size[1] * s
            theta = torch.tensor([[[r * np.cos(a), -r * np.sin(a), 0, 0],
                                   [r * np.sin(a), r * np.cos(a), 0, 0],
                                   [0, 0, 1, 0]]], dtype=torch.float32)
            grid = F.affine_grid(theta, (1, 1, *patch), align_corners=False)
            img = F.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=False)
            lbl = F.grid_sample(lbl, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        if rng.random() < 0.5:
            img, lbl = img.flip(-1), lbl.flip(-1)
        img = img * rng.uniform(0.9, 1.1) + rng.uniform(-0.1, 0.1)
        if rng.random() < 0.15:
            img = img + torch.from_numpy(rng.normal(0, rng.uniform(0, 0.1), img.shape).astype(np.float32))
        return img[0], lbl[0].clamp(0, 1)

    def batch(self, step, n):
        rng = np.random.default_rng([SEED, step])
        pairs = [self.sample(rng) for _ in range(n)]
        return torch.stack([p[0] for p in pairs]), torch.stack([p[1] for p in pairs])


def loss_fn(logits, target):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    dice = (2 * (p * target).sum() + 1e-5) / (p.sum() + target.sum() + 1e-5)
    return bce + (1 - dice), float(dice.detach())


def dice(a, b):
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else float(2 * (a & b).sum() / s)


def validate(model, stage, patients):
    """Dice on the stage's own grid: the coarse model on the whole head, the fine model in a box around the tumour."""
    cfg = STAGES[stage]
    scores, found = [], 0
    for p in patients:
        img, lbl = data.load_cached(p, stage, mmap=False)
        truth = lbl >= 128
        if stage == "fine":  # the localiser's job is not tested here
            nz = np.argwhere(truth)
            lo = np.maximum(nz.min(0) - np.array([8, 40, 40]), 0)
            hi = nz.max(0) + np.array([9, 41, 41])
            box = tuple(slice(a, b) for a, b in zip(lo, hi))
            img, truth = img[box], truth[box]
        pred = sliding_window(model, img.astype(np.float32), cfg["patch"]) >= 0.5
        scores.append(dice(pred, truth))
        found += bool((pred & truth).any())
    return float(np.mean(scores)), found


def save(path, **state):
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=STAGES)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--steps", type=int, default=None, help="override the number of steps (for a quick test)")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    cfg = dict(STAGES[args.stage])
    if args.steps:
        cfg["steps"] = args.steps
    out = MODELS / args.stage
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "train.log", "a")

    def say(msg):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    s = splits.load()
    ready = lambda ps: [p for p in ps if (data.CACHE / p / "meta.json").exists()]
    train, val = ready(s["train"]), ready(s["val"])
    torch.manual_seed(SEED)
    model = build(args.stage)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-5)
    step, best, history = 0, -1.0, []
    if (out / "last.pt").exists():
        state = torch.load(out / "last.pt", weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        step, best, history = state["step"], state["best"], state["history"]
        say(f"resumed at step {step}")
    else:
        say(f"start {args.stage}: {len(train)} training and {len(val)} validation patients, config {json.dumps(cfg)}")
    patches = Patches(train, args.stage, cfg)
    say(f"loaded {len(patches.items)} training patients")
    t0, losses, dices = time.time(), [], []
    while step < cfg["steps"]:
        lr = cfg["lr"] * min(1.0, (step + 1) / 200) * (1 - step / cfg["steps"]) ** 0.9  # warm-up, then poly decay
        for g in opt.param_groups:
            g["lr"] = lr
        model.train()
        x, y = patches.batch(step, cfg["batch"])
        with torch.autocast("cpu", dtype=torch.bfloat16):
            logits = model(x)
        loss, d = loss_fn(logits.float(), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        step += 1
        losses.append(float(loss))
        dices.append(d)
        if step % 50 == 0:
            say(f"step {step} loss {np.mean(losses):.4f} batch dice {np.mean(dices):.3f} lr {lr:.2e} "
                f"{(time.time() - t0) / len(losses):.2f} s/step")
            t0, losses, dices = time.time(), [], []
        if step % cfg["val_every"] == 0 or step == cfg["steps"]:
            v, found = validate(model, args.stage, val)
            history.append({"step": step, "val_dice": v, "val_found": found})
            say(f"validation at step {step}: mean dice {v:.4f}, tumour found in {found} of {len(val)}")
            if v > best:
                best = v
                save(out / "best.pt", model=model.state_dict(), step=step, val_dice=v)
            save(out / "last.pt", model=model.state_dict(), opt=opt.state_dict(), step=step, best=best, history=history)
            t0 = time.time()
        elif step % 250 == 0:
            save(out / "last.pt", model=model.state_dict(), opt=opt.state_dict(), step=step, best=best, history=history)
    say(f"finished {args.stage} at step {step}, best validation dice {best:.4f}")


if __name__ == "__main__":
    main()
