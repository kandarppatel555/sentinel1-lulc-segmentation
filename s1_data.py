"""
Data access for Sentinel-1 and ESA WorldCover via Google Earth Engine.

Used by BOTH the dataset notebook and the prototype app, so training data and
app inputs are produced by exactly the same code path (same compositing,
same masking, same grid).

Requires:  pip install earthengine-api pyproj numpy
One-time:  earthengine authenticate   (or ee.Authenticate() in a notebook)
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from pyproj import Transformer

import config as C

try:
    import ee
except ImportError:  # allows importing helpers (e.g. for tests) without EE installed
    ee = None


# ---------------------------------------------------------------------------
# Earth Engine initialisation
# ---------------------------------------------------------------------------
def init_ee(project: str | None = None) -> None:
    """Initialise Earth Engine. `project` = your Google Cloud project ID
    registered for Earth Engine (noncommercial use is free)."""
    if ee is None:
        raise ImportError("earthengine-api is not installed: pip install earthengine-api")
    try:
        ee.Initialize(project=project)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project)


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------
@dataclass
class Grid:
    """A north-up pixel grid in a UTM projection, centred on (lat, lon)."""
    crs: str          # e.g. "EPSG:32632"
    x0: float         # left edge (m)
    y0: float         # top edge (m)
    size_px: int
    pixel_size: float = C.PIXEL_SIZE_M

    @property
    def bounds(self):
        """(xmin, ymin, xmax, ymax) in the grid CRS."""
        ext = self.size_px * self.pixel_size
        return (self.x0, self.y0 - ext, self.x0 + ext, self.y0)

    def to_ee_grid(self) -> dict:
        return {
            "dimensions": {"width": self.size_px, "height": self.size_px},
            "affineTransform": {
                "scaleX": self.pixel_size, "shearX": 0, "translateX": self.x0,
                "shearY": 0, "scaleY": -self.pixel_size, "translateY": self.y0,
            },
            "crsCode": self.crs,
        }

    def ee_geometry(self):
        xmin, ymin, xmax, ymax = self.bounds
        return ee.Geometry.Rectangle([xmin, ymin, xmax, ymax], proj=self.crs, geodesic=False)

    def latlon_bounds(self):
        """(south, west, north, east) in WGS84, for map previews."""
        t = Transformer.from_crs(self.crs, "EPSG:4326", always_xy=True)
        xmin, ymin, xmax, ymax = self.bounds
        lons, lats = t.transform([xmin, xmax, xmin, xmax], [ymin, ymin, ymax, ymax])
        return min(lats), min(lons), max(lats), max(lons)

    def to_dict(self) -> dict:
        return {"crs": self.crs, "x0": self.x0, "y0": self.y0,
                "size_px": self.size_px, "pixel_size": self.pixel_size}


def utm_epsg(lat: float, lon: float) -> str:
    zone = int((lon + 180) // 6) + 1
    return f"EPSG:{32600 + zone if lat >= 0 else 32700 + zone}"


def make_grid(lat: float, lon: float, size_px: int) -> Grid:
    """Square grid of `size_px` pixels centred on (lat, lon), snapped to the 10 m lattice."""
    crs = utm_epsg(lat, lon)
    x, y = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(lon, lat)
    half = size_px * C.PIXEL_SIZE_M / 2
    x0 = round((x - half) / C.PIXEL_SIZE_M) * C.PIXEL_SIZE_M
    y0 = round((y + half) / C.PIXEL_SIZE_M) * C.PIXEL_SIZE_M
    return Grid(crs=crs, x0=float(x0), y0=float(y0), size_px=size_px)


def add_months(year: int, month: int, n: int) -> tuple[int, int]:
    idx = year * 12 + (month - 1) + n
    return idx // 12, idx % 12 + 1


def month_range(start_year: int, start_month: int, n: int = 12) -> list[tuple[int, int]]:
    return [add_months(start_year, start_month, i) for i in range(n)]


# ---------------------------------------------------------------------------
# Earth Engine image builders
# ---------------------------------------------------------------------------
def _mask_edges(img):
    angle = img.select("angle")
    keep = angle.gt(C.ANGLE_MIN).And(angle.lt(C.ANGLE_MAX))
    return img.updateMask(keep).select(C.S1_BANDS)


def s1_collection(geometry, year: int, month: int):
    """All dual-pol IW GRD scenes over `geometry` in the given calendar month."""
    y2, m2 = add_months(year, month, 1)
    start = f"{year:04d}-{month:02d}-01"
    end = f"{y2:04d}-{m2:02d}-01"
    return (ee.ImageCollection(C.S1_COLLECTION)
            .filterBounds(geometry)
            .filterDate(start, end)
            .filter(ee.Filter.eq("instrumentMode", C.INSTRUMENT_MODE))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
            .map(_mask_edges))


def s1_monthly_composite(geometry, year: int, month: int):
    """Monthly median composite in dB (both orbit directions).
    Median of dB == dB of median (log is monotonic), so compositing in dB is exact."""
    coll = s1_collection(geometry, year, month)
    return coll.median().toFloat(), coll.size()


def worldcover_image():
    return ee.ImageCollection(C.WORLDCOVER_ASSET).first().select("Map")


# ---------------------------------------------------------------------------
# Pixel download
# ---------------------------------------------------------------------------
def _compute_pixels(image, grid: Grid, retries: int = 5):
    req = {"expression": image, "fileFormat": "NUMPY_NDARRAY", "grid": grid.to_ee_grid()}
    for attempt in range(retries):
        try:
            return ee.data.computePixels(req)
        except Exception as e:  # rate limits / transient errors
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt
            print(f"  EE error ({e.__class__.__name__}: {str(e)[:80]}), retry in {wait}s")
            time.sleep(wait)


def fetch_s1_month(grid: Grid, year: int, month: int) -> tuple[np.ndarray, int]:
    """Returns (array of shape (2, H, W) float32 in dB with NaN = no data, n_scenes)."""
    geom = grid.ee_geometry()
    comp, size = s1_monthly_composite(geom, year, month)
    n_scenes = int(size.getInfo())
    shape = (len(C.S1_BANDS), grid.size_px, grid.size_px)
    if n_scenes == 0:
        return np.full(shape, np.nan, dtype=np.float32), 0
    arr = _compute_pixels(comp.unmask(C.NODATA_VALUE), grid)
    out = np.stack([arr[b].astype(np.float32) for b in C.S1_BANDS])
    out[out <= C.NODATA_VALUE + 1] = np.nan
    return out, n_scenes


def fetch_worldcover(grid: Grid) -> np.ndarray:
    """Returns merged class labels (H, W) uint8, IGNORE_INDEX where unmapped."""
    arr = _compute_pixels(worldcover_image().unmask(0).toUint8(), grid)
    return remap_worldcover(arr["Map"])


def remap_worldcover(codes: np.ndarray) -> np.ndarray:
    lut = np.full(256, C.IGNORE_INDEX, dtype=np.uint8)
    for code, cls in C.WORLDCOVER_TO_CLASS.items():
        lut[code] = cls
    return lut[codes.astype(np.uint8)]


# ---------------------------------------------------------------------------
# Feature engineering (shared by training and inference)
# ---------------------------------------------------------------------------
def make_features(s1: np.ndarray) -> np.ndarray:
    """(2, H, W) VV,VH in dB  ->  (3, H, W) VV, VH, VV-VH, clipped. NaN preserved."""
    vv, vh = s1[0].astype(np.float32), s1[1].astype(np.float32)
    feats = np.stack([vv, vh, vv - vh])
    for i, name in enumerate(C.INPUT_CHANNELS):
        lo, hi = C.DB_CLIP[name]
        feats[i] = np.clip(feats[i], lo, hi)  # np.clip keeps NaN as NaN
    return feats


def normalise(feats: np.ndarray, mean, std) -> tuple[np.ndarray, np.ndarray]:
    """Standardise per channel. Returns (normalised array with NaN -> 0, valid mask)."""
    mean = np.asarray(mean, dtype=np.float32)[:, None, None]
    std = np.asarray(std, dtype=np.float32)[:, None, None]
    valid = np.isfinite(feats).all(axis=0)
    x = (feats - mean) / std
    x[:, ~valid] = 0.0
    return x.astype(np.float32), valid
