"""
Model definition, checkpoint I/O, inference and area statistics.
Shared by the training notebooks and the Streamlit prototype.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp

import config as C
from s1_data import make_features, normalise


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
ARCHITECTURES = ["Unet", "UnetPlusPlus", "DeepLabV3Plus", "FPN", "Segformer", "UPerNet", "PSPNet", "Linknet"]


def build_model(encoder: str = "resnet34", encoder_weights: str | None = "imagenet",
                arch: str = "Unet", strict_weights: bool = False) -> torch.nn.Module:
    """Any segmentation_models_pytorch architecture, e.g. arch="DeepLabV3Plus".
    strict_weights=True raises instead of silently training from scratch when the
    pretrained weights cannot be downloaded (used for fair model comparisons)."""
    cls = getattr(smp, arch)
    kwargs = dict(encoder_name=encoder, in_channels=len(C.INPUT_CHANNELS), classes=C.NUM_CLASSES)
    try:
        return cls(encoder_weights=encoder_weights, **kwargs)
    except Exception as e:  # e.g. no internet access to download pretrained weights
        if encoder_weights is None or strict_weights:
            raise
        print(f"WARNING: could not load '{encoder_weights}' encoder weights ({e.__class__.__name__}). "
              "Training from scratch instead; expect slower convergence.")
        return cls(encoder_weights=None, **kwargs)


def save_checkpoint(path, model, encoder, mean, std, extra: dict | None = None, arch: str = "Unet"):
    ckpt = {
        "state_dict": model.state_dict(),
        "arch": arch,
        "encoder": encoder,
        "mean": list(map(float, mean)),
        "std": list(map(float, std)),
        "class_names": C.CLASS_NAMES,
        "input_channels": C.INPUT_CHANNELS,
    }
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, path)


def load_checkpoint(path, device: str = "cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ckpt["encoder"], encoder_weights=None, arch=ckpt.get("arch", "Unet"))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict(model, s1: np.ndarray, mean, std, device: str = "cpu") -> np.ndarray:
    """s1: (2, H, W) VV,VH in dB (NaN = no data).
    Returns (H, W) uint8 class map, IGNORE_INDEX where there is no radar data."""
    feats = make_features(s1)
    x, valid = normalise(feats, mean, std)
    h, w = x.shape[1:]
    ph, pw = (-h) % 32, (-w) % 32  # U-Net needs sizes divisible by 32
    t = torch.from_numpy(x)[None].to(device)
    if ph or pw:
        t = F.pad(t, (0, pw, 0, ph), mode="reflect")
    logits = model(t)[..., :h, :w]
    pred = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
    pred[~valid] = C.IGNORE_INDEX
    return pred


def temporal_majority(maps: np.ndarray, window: int = 3) -> np.ndarray:
    """Optional smoothing: each month becomes the majority class over a centred
    window of months. maps: (T, H, W) uint8. Ignored pixels do not vote."""
    T = maps.shape[0]
    half = window // 2
    onehot = np.zeros((T, C.NUM_CLASSES) + maps.shape[1:], dtype=np.uint8)
    for k in range(C.NUM_CLASSES):
        onehot[:, k] = maps == k
    out = np.empty_like(maps)
    for t in range(T):
        lo, hi = max(0, t - half), min(T, t + half + 1)
        votes = onehot[lo:hi].sum(0)
        out[t] = votes.argmax(0)
        out[t][votes.sum(0) == 0] = C.IGNORE_INDEX
    return out


# ---------------------------------------------------------------------------
# Area statistics
# ---------------------------------------------------------------------------
def class_percentages(class_map: np.ndarray) -> np.ndarray:
    """Share of each class among valid pixels, in percent."""
    valid = class_map != C.IGNORE_INDEX
    n = valid.sum()
    if n == 0:
        return np.full(C.NUM_CLASSES, np.nan)
    counts = np.bincount(class_map[valid].ravel(), minlength=C.NUM_CLASSES)[: C.NUM_CLASSES]
    return 100.0 * counts / n


def area_statistics(maps: np.ndarray, month_labels: list[str]):
    """maps: (12, H, W). Returns three DataFrames:
      pct   : % of area per class per month                     (rows = months)
      delta : month-to-month change in percentage points        (rows = months 2..12)
      total : change first -> last month in percentage points   (one row per class)
    """
    pct = pd.DataFrame([class_percentages(m) for m in maps],
                       index=month_labels, columns=C.CLASS_NAMES)
    delta = pct.diff().iloc[1:]
    delta.index = [f"{a} → {b}" for a, b in zip(month_labels[:-1], month_labels[1:])]
    total = pd.DataFrame({
        f"{month_labels[0]} (%)": pct.iloc[0],
        f"{month_labels[-1]} (%)": pct.iloc[-1],
        "Change (pp)": pct.iloc[-1] - pct.iloc[0],
    })
    return pct, delta, total


def valid_fraction(maps: np.ndarray) -> list[float]:
    return [float((m != C.IGNORE_INDEX).mean()) for m in maps]
