"""From an MRI to a tumour outline, in two steps.

1. The coarse model reads the whole head on the 2 mm grid and marks where
   the tumour probably is. The largest marked region, by summed
   probability, is the candidate.
2. The fine model reads a box around that region on the 0.5 mm grid and
   outlines the tumour in it.

Both models see the image in overlapping windows; each voxel's probability
is the average of the windows that cover it, weighted towards window centres
(where the network sees the most context). Mirroring left and right and
averaging is optional.

The result is a probability on the fine grid, inside the box. to_native puts
it back on the MRI's own grid, where it is compared with the doctors' outline.
"""

from dataclasses import dataclass

import numpy as np
import torch
from scipy import ndimage

from .data import resample


def window_weights(patch):
    g = [np.exp(-0.5 * ((np.arange(n) - (n - 1) / 2) / (n / 8)) ** 2) for n in patch]
    w = g[0][:, None, None] * g[1][None, :, None] * g[2][None, None, :]
    return np.maximum(w / w.max(), 1e-3).astype(np.float32)


def starts(size, patch, step):
    if size <= patch:
        return [0]
    s = list(range(0, size - patch, step)) + [size - patch]
    return sorted(set(s))


@torch.no_grad()
def sliding_window(model, image, patch, overlap=0.5, mirror=False):
    """Tumour probability for every voxel of `image` [z, y, x]."""
    model.eval()
    pad = [(0, max(0, p - s)) for s, p in zip(image.shape, patch)]
    img = np.pad(np.asarray(image, np.float32), pad, mode="edge")
    out = np.zeros(img.shape, np.float32)
    norm = np.zeros(img.shape, np.float32)
    w = window_weights(patch)
    steps = [max(1, int(p * (1 - overlap))) for p in patch]
    for z in starts(img.shape[0], patch[0], steps[0]):
        for y in starts(img.shape[1], patch[1], steps[1]):
            for x in starts(img.shape[2], patch[2], steps[2]):
                sl = (slice(z, z + patch[0]), slice(y, y + patch[1]), slice(x, x + patch[2]))
                t = torch.from_numpy(np.ascontiguousarray(img[sl]))[None, None]
                p = torch.sigmoid(model(t))
                if mirror:
                    p = 0.5 * (p + torch.sigmoid(model(t.flip(-1))).flip(-1))
                out[sl] += p[0, 0].numpy() * w
                norm[sl] += w
    out /= norm
    return out[:image.shape[0], :image.shape[1], :image.shape[2]]


@dataclass
class Settings:
    coarse_patch: tuple = (48, 80, 80)
    fine_patch: tuple = (24, 112, 112)
    margin_mm: float = 12.0  # added around the coarse candidate before the fine step
    mirror: bool = False
    candidates: int = 1  # how many coarse regions the fine model looks at


def candidates(prob, threshold=0.5, keep=1):
    """Connected regions of the coarse probability, largest summed probability first, as index boxes."""
    lab, n = ndimage.label(prob >= threshold)
    if n == 0:  # nothing confident: look around the single most likely voxel
        c = np.unravel_index(int(np.argmax(prob)), prob.shape)
        return [tuple(slice(int(i), int(i) + 1) for i in c)]
    mass = ndimage.sum(prob, lab, index=np.arange(1, n + 1))
    boxes = ndimage.find_objects(lab)
    order = np.argsort(mass)[::-1][:keep]
    return [boxes[i] for i in order]


def box_on(axes_from, box, axes_to, margin_mm, min_size):
    """An index box on axes_to covering `box` on axes_from plus a margin, at least min_size voxels."""
    out = []
    for a, b, s, m in zip(axes_from, axes_to, box, min_size):
        lo_mm, hi_mm = a[s.start] - margin_mm, a[s.stop - 1] + margin_mm
        step = b[1] - b[0]
        lo, hi = int(np.floor((lo_mm - b[0]) / step)), int(np.ceil((hi_mm - b[0]) / step)) + 1
        if hi - lo < m:
            mid = (lo + hi) // 2
            lo, hi = mid - m // 2, mid - m // 2 + m
        lo, hi = max(lo, 0), min(hi, len(b))
        if hi - lo < m:  # against an edge of the image
            lo, hi = (0, min(m, len(b))) if lo == 0 else (max(len(b) - m, 0), len(b))
        out.append(slice(lo, hi))
    return tuple(out)


def outline(coarse_model, fine_model, coarse_img, fine_img, coarse_axes, fine_axes, settings=Settings()):
    """Tumour probability on the fine grid: (probability in the box, the box as index slices)."""
    pc = sliding_window(coarse_model, coarse_img, settings.coarse_patch, mirror=settings.mirror)
    full = None
    for cand in candidates(pc, keep=settings.candidates):
        box = box_on(coarse_axes, cand, fine_axes, settings.margin_mm, settings.fine_patch)
        p = sliding_window(fine_model, fine_img[box], settings.fine_patch, mirror=settings.mirror)
        if full is None:
            full = (p, box)
        elif p.max() > full[0].max():  # several candidates: keep the one the fine model is surest of
            full = (p, box)
    return full


def to_native(prob, box, fine_axes, native_axes):
    """Probability on the MRI's own grid, inside the matching box: (probability, box)."""
    src = [a[s] for a, s in zip(fine_axes, box)]
    nbox = []
    for a, s in zip(native_axes, src):
        lo = int(np.searchsorted(a, s[0], side="left"))
        hi = int(np.searchsorted(a, s[-1], side="right"))
        nbox.append(slice(lo, hi))
    nbox = tuple(nbox)
    dst = [a[s] for a, s in zip(native_axes, nbox)]
    return resample(prob, src, dst), nbox
