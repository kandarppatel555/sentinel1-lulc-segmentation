"""
Dataset loading, PyTorch datasets, metrics and the train/eval loops.
Kept in a module (not inside the notebook) so it can be tested and reused.

On-disk layout produced by notebooks/01_build_dataset.ipynb:
    data/regions/<name>/s1.npy     (12, 2, H, W) float16, VV/VH in dB, NaN = no data
    data/regions/<name>/label.npy  (H, W) uint8, merged classes, 255 = ignore
    data/regions/<name>/meta.json  grid, split, scenes per month
    data/norm_stats.json           per-channel mean/std over the training split
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

import config as C
from s1_data import make_features, normalise


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_regions(data_dir, split: str | None = None) -> list[dict]:
    regions = []
    for d in sorted(Path(data_dir, "regions").iterdir()):
        meta_path = d / "meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        if split and meta["split"] != split:
            continue
        regions.append({
            "name": meta["name"],
            "split": meta["split"],
            "s1": np.load(d / "s1.npy", mmap_mode=None),     # (12, 2, H, W) float16
            "label": np.load(d / "label.npy"),               # (H, W) uint8
            "meta": meta,
        })
    return regions


def compute_norm_stats(regions: list[dict], max_pixels_per_image: int = 20000, seed: int = 0):
    """Per-channel mean/std of the model input features over valid training pixels."""
    rng = np.random.default_rng(seed)
    samples = []
    for r in regions:
        for t in range(r["s1"].shape[0]):
            f = make_features(r["s1"][t].astype(np.float32)).reshape(len(C.INPUT_CHANNELS), -1)
            ok = np.isfinite(f).all(0)
            idx = np.flatnonzero(ok)
            if idx.size == 0:
                continue
            idx = rng.choice(idx, size=min(max_pixels_per_image, idx.size), replace=False)
            samples.append(f[:, idx])
    allpix = np.concatenate(samples, axis=1)
    return allpix.mean(1).tolist(), allpix.std(1).tolist()


def class_counts(regions: list[dict]) -> np.ndarray:
    counts = np.zeros(C.NUM_CLASSES, dtype=np.int64)
    for r in regions:
        lab = r["label"]
        counts += np.bincount(lab[lab != C.IGNORE_INDEX].ravel(), minlength=C.NUM_CLASSES)[: C.NUM_CLASSES]
    return counts


def class_weights_from_counts(counts: np.ndarray) -> torch.Tensor:
    """Inverse square-root frequency, normalised to mean 1 (milder than pure inverse)."""
    freq = counts / counts.sum()
    w = 1.0 / np.sqrt(np.maximum(freq, 1e-6))
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


def _sample(s1_month: np.ndarray, label: np.ndarray, mean, std):
    feats = make_features(s1_month.astype(np.float32))
    x, valid = normalise(feats, mean, std)
    y = label.astype(np.int64).copy()
    y[~valid] = C.IGNORE_INDEX
    return x, y


def usable_months(region: dict, min_valid: float = 0.5) -> list[int]:
    s1 = region["s1"]
    return [t for t in range(s1.shape[0]) if np.isfinite(s1[t, 0]).mean() >= min_valid]


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
class RandomCropDataset(Dataset):
    """Training: random region, random month, random 256 crop, random flips/rotations.
    Every month of 2021 is paired with the same WorldCover label, so the model
    is pushed to learn season-invariant features."""

    def __init__(self, regions, mean, std, patch=C.PATCH_SIZE_PX, samples_per_epoch=2000,
                 augment=True, seed=None):
        self.regions = regions
        self.mean, self.std = mean, std
        self.patch = patch
        self.n = samples_per_epoch
        self.augment = augment
        self.index = [(i, t) for i, r in enumerate(regions) for t in usable_months(r)]
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.n

    def __getitem__(self, _):
        i, t = self.index[self.rng.integers(len(self.index))]
        r = self.regions[i]
        H, W = r["label"].shape
        y0 = self.rng.integers(0, H - self.patch + 1)
        x0 = self.rng.integers(0, W - self.patch + 1)
        sl = (slice(y0, y0 + self.patch), slice(x0, x0 + self.patch))
        x, y = _sample(r["s1"][t][:, sl[0], sl[1]], r["label"][sl], self.mean, self.std)
        if self.augment:
            k = self.rng.integers(4)
            x, y = np.rot90(x, k, axes=(1, 2)), np.rot90(y, k)
            if self.rng.random() < 0.5:
                x, y = x[:, :, ::-1], y[:, ::-1]
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(y))


class TiledDataset(Dataset):
    """Validation/test: every usable month of every region, cut into non-overlapping tiles."""

    def __init__(self, regions, mean, std, patch=C.PATCH_SIZE_PX):
        self.regions, self.mean, self.std, self.patch = regions, mean, std, patch
        self.items = []
        for i, r in enumerate(regions):
            H, W = r["label"].shape
            for t in usable_months(r):
                for y0 in range(0, H - patch + 1, patch):
                    for x0 in range(0, W - patch + 1, patch):
                        self.items.append((i, t, y0, x0))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        i, t, y0, x0 = self.items[k]
        r = self.regions[i]
        p = self.patch
        x, y = _sample(r["s1"][t][:, y0:y0 + p, x0:x0 + p], r["label"][y0:y0 + p, x0:x0 + p],
                       self.mean, self.std)
        return torch.from_numpy(x), torch.from_numpy(y)


def seed_worker(worker_id):
    """Give each DataLoader worker its own RNG stream for RandomCropDataset."""
    info = torch.utils.data.get_worker_info()
    ds = info.dataset
    if hasattr(ds, "rng"):
        ds.rng = np.random.default_rng(torch.initial_seed() % 2**32)


# ---------------------------------------------------------------------------
# GPU data pipeline
# All training data lives in GPU memory; cropping, feature computation,
# normalisation and augmentation run as PyTorch operations on the GPU.
# Same sampling logic as RandomCropDataset, but no CPU workers and no shared
# memory are needed (fixes the Docker /dev/shm limit and keeps the GPU busy).
# ---------------------------------------------------------------------------
def _features_torch(s1: torch.Tensor, mean, std):
    """s1: (B, 2, H, W) dB, NaN = no data  ->  x (B, 3, H, W) standardised, valid (B, H, W)."""
    s1 = s1.float()
    vv, vh = s1[:, 0], s1[:, 1]
    feats = torch.stack([vv, vh, vv - vh], dim=1)
    for i, name in enumerate(C.INPUT_CHANNELS):
        lo, hi = C.DB_CLIP[name]
        feats[:, i] = feats[:, i].clamp(lo, hi)          # clamp keeps NaN as NaN
    valid = torch.isfinite(feats).all(dim=1)
    m = torch.as_tensor(mean, dtype=torch.float32, device=s1.device).view(1, -1, 1, 1)
    s = torch.as_tensor(std, dtype=torch.float32, device=s1.device).view(1, -1, 1, 1)
    x = (feats - m) / s
    x = torch.where(valid.unsqueeze(1), x, torch.zeros_like(x))
    return x, valid


class GPUCropSampler:
    """Iterable that yields `steps` random training batches per epoch, already on `device`."""

    def __init__(self, regions, mean, std, device, batch_size=64, steps=187,
                 patch=C.PATCH_SIZE_PX, augment=True, seed=0):
        self.mean, self.std, self.device = mean, std, device
        self.bs, self.steps, self.patch, self.augment = batch_size, steps, patch, augment
        # (R, 12, 2, H, W) float16 and (R, H, W) uint8 on the GPU (~0.6 GB for 40 regions)
        self.s1 = torch.from_numpy(np.stack([r["s1"] for r in regions])).to(device)
        self.label = torch.from_numpy(np.stack([r["label"] for r in regions])).to(device)
        pairs = [(i, t) for i, r in enumerate(regions) for t in usable_months(r)]
        self.pairs = torch.tensor(pairs, dtype=torch.long, device=device)
        self.H, self.W = self.label.shape[1:]
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(seed)
        self.ar = torch.arange(patch, device=device)

    def __len__(self):
        return self.steps

    def _randint(self, high, n):
        return torch.randint(0, high, (n,), generator=self.gen, device=self.device)

    def batch(self):
        B, p = self.bs, self.patch
        pick = self.pairs[self._randint(len(self.pairs), B)]
        ri, ti = pick[:, 0], pick[:, 1]
        y0 = self._randint(self.H - p + 1, B)
        x0 = self._randint(self.W - p + 1, B)
        rows = (y0[:, None] + self.ar)[:, :, None]        # (B, p, 1)
        cols = (x0[:, None] + self.ar)[:, None, :]        # (B, 1, p)
        b = ri[:, None, None]
        s1 = torch.stack([self.s1[b, ti[:, None, None], c, rows, cols] for c in range(2)], dim=1)
        y = self.label[b, rows, cols].long()
        x, valid = _features_torch(s1, self.mean, self.std)
        y[~valid] = C.IGNORE_INDEX
        if self.augment:
            k = self._randint(4, B)
            flip = self._randint(2, B).bool()
            for kk in range(1, 4):
                sel = k == kk
                if sel.any():
                    x[sel] = torch.rot90(x[sel], kk, dims=(2, 3))
                    y[sel] = torch.rot90(y[sel], kk, dims=(1, 2))
            x[flip] = x[flip].flip(3)
            y[flip] = y[flip].flip(2)
        return x.contiguous(), y.contiguous()

    def __iter__(self):
        for _ in range(self.steps):
            yield self.batch()


class GPUTiles:
    """Validation/test tiles (same tiling as TiledDataset), precomputed once on `device`."""

    def __init__(self, regions, mean, std, device, batch_size=64, patch=C.PATCH_SIZE_PX):
        ds = TiledDataset(regions, mean, std, patch)
        xs, ys = zip(*(ds[k] for k in range(len(ds))))
        self.x = torch.stack(xs).to(device)
        self.y = torch.stack(ys).to(device)
        self.bs = batch_size

    def __len__(self):
        return (len(self.x) + self.bs - 1) // self.bs

    def __iter__(self):
        for i in range(0, len(self.x), self.bs):
            yield self.x[i:i + self.bs], self.y[i:i + self.bs]


# ---------------------------------------------------------------------------
# Optimizers
# ---------------------------------------------------------------------------
def make_optimizer(name: str, params, lr: float, weight_decay: float = 1e-4):
    name = name.lower()
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "adam":   # classic Adam: weight decay as L2 term coupled to the adaptive step
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, nesterov=True, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer '{name}'")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def confusion_matrix(pred: torch.Tensor, target: torch.Tensor, n: int = C.NUM_CLASSES) -> torch.Tensor:
    m = target != C.IGNORE_INDEX
    idx = target[m] * n + pred[m]
    return torch.bincount(idx, minlength=n * n).reshape(n, n)


def metrics_from_cm(cm) -> dict:
    """cm[true, pred]. Returns OA, per-class IoU, precision, recall, F1 and their means.
    Classes absent from both target and prediction get NaN and are excluded from means."""
    cm = np.asarray(cm, dtype=np.float64)
    tp = np.diag(cm)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = tp / (tp + fp + fn)
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        f1 = 2 * prec * rec / (prec + rec)
    return {
        "overall_accuracy": float(tp.sum() / max(cm.sum(), 1)),
        "mIoU": float(np.nanmean(iou)),
        "mean_F1": float(np.nanmean(f1)),
        "IoU": iou.tolist(), "precision": prec.tolist(), "recall": rec.tolist(), "F1": f1.tolist(),
    }


# ---------------------------------------------------------------------------
# Loops
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, loss_fn, optimizer, scheduler, scaler, device):
    model.train()
    total, n = 0.0, 0
    cm = torch.zeros(C.NUM_CLASSES, C.NUM_CLASSES, dtype=torch.long, device=device)
    use_amp = device.startswith("cuda")
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(x)
            loss = loss_fn(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()
        total += loss.item() * x.size(0)
        n += x.size(0)
        cm += confusion_matrix(logits.argmax(1), y)
    out = metrics_from_cm(cm.cpu().numpy())
    out["loss"] = total / n
    return out


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total, n = 0.0, 0
    cm = torch.zeros(C.NUM_CLASSES, C.NUM_CLASSES, dtype=torch.long, device=device)
    use_amp = device.startswith("cuda")
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(x)
            loss = loss_fn(logits, y)
        total += loss.item() * x.size(0)
        n += x.size(0)
        cm += confusion_matrix(logits.argmax(1), y)
    out = metrics_from_cm(cm.cpu().numpy())
    out["loss"] = total / n
    out["confusion_matrix"] = cm.cpu().numpy().tolist()
    return out


class CombinedLoss(torch.nn.Module):
    """Weighted cross-entropy + Dice. CE drives per-pixel accuracy; Dice counteracts
    class imbalance by optimising overlap per class."""

    def __init__(self, class_weights, dice_weight=0.5):
        super().__init__()
        import segmentation_models_pytorch as smp
        self.ce = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=C.IGNORE_INDEX)
        self.dice = smp.losses.DiceLoss(mode="multiclass", ignore_index=C.IGNORE_INDEX)
        self.dice_weight = dice_weight

    def forward(self, logits, target):
        return self.ce(logits, target) + self.dice_weight * self.dice(logits, target)
