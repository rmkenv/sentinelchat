"""
sentinel_core.py
----------------
Sentinel-2 L2A pipeline via Microsoft Planetary Computer STAC.
No GEE. No sign-in. Read-only access via SAS token signing.

Data: Microsoft Planetary Computer
      https://planetarycomputer.microsoft.com
      Collection: sentinel-2-l2a
"""

import json
import logging
import os
import re
import warnings
from dataclasses import dataclass, field
from typing import Optional

import httpx
import numpy as np
import requests

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("sentinel")

try:
    import rasterio
    from rasterio.crs import CRS
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds as window_from_bounds
    import rasterio.mask
except ImportError as e:
    raise ImportError("pip install rasterio") from e

try:
    from pyproj import Transformer
    from shapely.geometry import box, mapping
except ImportError as e:
    raise ImportError("pip install pyproj shapely") from e

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PC_STAC_URL   = "https://planetarycomputer.microsoft.com/api/stac/v1"
PC_TOKEN_URL  = "https://planetarycomputer.microsoft.com/api/sas/v1/token"
COLLECTION    = "sentinel-2-l2a"

# Band → asset key in PC S2 items
BAND_ASSETS = {
    "B02": "B02",   # Blue      10m
    "B03": "B03",   # Green     10m
    "B04": "B04",   # Red       10m
    "B08": "B08",   # NIR       10m
    "B11": "B11",   # SWIR-1    20m
    "B12": "B12",   # SWIR-2    20m
    "SCL": "SCL",   # Scene Classification 20m
}

# Spectral indices computed by this pipeline
INDICES = ["NDVI", "NDWI", "NDBI", "EVI", "SAVI"]

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class AOI:
    lon_min: float
    lat_min: float
    lon_max: float
    lat_max: float
    name: str = "AOI"

    def center(self):
        return (self.lon_min + self.lon_max) / 2, (self.lat_min + self.lat_max) / 2

    def bbox(self):
        return [self.lon_min, self.lat_min, self.lon_max, self.lat_max]

    def to_shapely(self):
        return box(self.lon_min, self.lat_min, self.lon_max, self.lat_max)


@dataclass
class SceneResult:
    """One Sentinel-2 scene's worth of extracted data."""
    bands: dict           # band_name → (H, W) float32 array, scaled 0–1
    indices: dict         # index_name → (H, W) float32 array
    mask: np.ndarray      # (H, W) bool — True = valid (not cloud/shadow/nodata)
    item_id: str
    date: str
    cloud_cover: float
    aoi: AOI
    crs: str
    stats: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# SAS token signing (PC read access, no auth required for public data)
# ---------------------------------------------------------------------------

_token_cache: dict = {}

def get_sas_token(collection: str = COLLECTION) -> str:
    """Fetch a short-lived SAS token for PC blob storage access."""
    if collection in _token_cache:
        return _token_cache[collection]
    url = f"{PC_TOKEN_URL}/{collection}"
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        token = r.json().get("token", "")
        _token_cache[collection] = token
        log.info("Got SAS token for %s", collection)
        return token
    except Exception as e:
        log.warning("SAS token fetch failed (%s) — URLs may still work unsigned", e)
        return ""


def sign_url(href: str, token: str) -> str:
    """Append SAS token to a PC blob storage URL."""
    if not token or "?" in href:
        return href
    return f"{href}?{token}"


# ---------------------------------------------------------------------------
# STAC search
# ---------------------------------------------------------------------------

def search_scenes(
    aoi: AOI,
    date_range: str,          # e.g. "2023-06-01/2023-09-01"
    max_cloud: float = 20.0,
    max_items: int = 5,
) -> list[dict]:
    """
    Search PC STAC for Sentinel-2 L2A scenes over an AOI.
    Returns list of STAC item dicts, sorted by cloud cover ascending.
    """
    payload = {
        "collections": [COLLECTION],
        "bbox": aoi.bbox(),
        "datetime": date_range,
        "query": {"eo:cloud_cover": {"lt": max_cloud}},
        "sortby": [{"field": "eo:cloud_cover", "direction": "asc"}],
        "limit": max_items,
    }

    url = f"{PC_STAC_URL}/search"
    log.info("Searching PC STAC: %s  clouds<%.0f%%  dates=%s", aoi.name, max_cloud, date_range)
    r = requests.post(url, json=payload, timeout=30)
    r.raise_for_status()

    items = r.json().get("features", [])
    log.info("Found %d scene(s)", len(items))
    return items


def best_scene(items: list[dict]) -> Optional[dict]:
    """Return the scene with lowest cloud cover."""
    if not items:
        return None
    return sorted(items, key=lambda x: x.get("properties", {}).get("eo:cloud_cover", 99))[0]


# ---------------------------------------------------------------------------
# Band reading
# ---------------------------------------------------------------------------

def read_band(
    href: str,
    aoi: AOI,
    target_epsg: int = 4326,
    overview_level: int = 0,
) -> Optional[np.ndarray]:
    """
    Read a single band COG for the AOI window.
    Returns (H, W) float32 scaled 0–1, or None on failure.
    """
    try:
        with rasterio.open(href) as src:
            native_crs = src.crs.to_epsg() or 4326

            # Reproject AOI bbox to band's native CRS
            if native_crs != 4326:
                tr = Transformer.from_crs(4326, native_crs, always_xy=True)
                x0, y0 = tr.transform(aoi.lon_min, aoi.lat_min)
                x1, y1 = tr.transform(aoi.lon_max, aoi.lat_max)
            else:
                x0, y0 = aoi.lon_min, aoi.lat_min
                x1, y1 = aoi.lon_max, aoi.lat_max

            left, right  = min(x0, x1), max(x0, x1)
            bottom, top  = min(y0, y1), max(y0, y1)

            win = window_from_bounds(left, bottom, right, top, src.transform)
            win = win.intersection(rasterio.windows.Window(0, 0, src.width, src.height))

            data = src.read(1, window=win).astype(np.float32)

            # S2 L2A DN → reflectance (divide by 10000)
            data = data / 10000.0
            data = np.clip(data, 0, 1)

            return data

    except Exception as e:
        log.warning("Band read failed for %s: %s", href.split("/")[-1], e)
        return None


def read_scl(href: str, aoi: AOI) -> Optional[np.ndarray]:
    """Read Scene Classification Layer (SCL) — uint8 class labels."""
    try:
        with rasterio.open(href) as src:
            native_crs = src.crs.to_epsg() or 4326
            if native_crs != 4326:
                tr = Transformer.from_crs(4326, native_crs, always_xy=True)
                x0, y0 = tr.transform(aoi.lon_min, aoi.lat_min)
                x1, y1 = tr.transform(aoi.lon_max, aoi.lat_max)
            else:
                x0, y0 = aoi.lon_min, aoi.lat_min
                x1, y1 = aoi.lon_max, aoi.lat_max

            left, right  = min(x0, x1), max(x0, x1)
            bottom, top  = min(y0, y1), max(y0, y1)

            win = window_from_bounds(left, bottom, right, top, src.transform)
            win = win.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
            return src.read(1, window=win).astype(np.uint8)
    except Exception as e:
        log.warning("SCL read failed: %s", e)
        return None


def scl_to_mask(scl: Optional[np.ndarray]) -> np.ndarray:
    """
    Convert SCL to a valid-pixel boolean mask.
    SCL classes kept as valid: 4=Vegetation, 5=Not-vegetated, 6=Water,
    7=Unclassified, 11=Snow/ice.
    Classes masked out: 0=NoData, 1=Saturated, 2=Dark, 3=CloudShadow,
    8=MedCloud, 9=HighCloud, 10=ThinCircus.
    """
    if scl is None:
        return None
    valid_classes = {4, 5, 6, 7, 11}
    mask = np.zeros(scl.shape, dtype=bool)
    for cls in valid_classes:
        mask |= (scl == cls)
    return mask


def align_to_shape(arr: Optional[np.ndarray], target_shape: tuple) -> Optional[np.ndarray]:
    """Resize array to target shape via nearest-neighbor (for 20m→10m bands)."""
    if arr is None:
        return None
    if arr.shape == target_shape:
        return arr
    from skimage.transform import resize
    return resize(arr, target_shape, order=0, anti_aliasing=False,
                  preserve_range=True).astype(arr.dtype)


# ---------------------------------------------------------------------------
# Spectral indices
# ---------------------------------------------------------------------------

def safe_ratio(num: np.ndarray, denom: np.ndarray) -> np.ndarray:
    """Compute ratio, returning 0 where denominator is near-zero."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(np.abs(denom) > 1e-6, num / denom, 0.0)
    return result.astype(np.float32)


def compute_indices(bands: dict) -> dict:
    """
    Compute spectral indices from band reflectance arrays.
    All inputs are float32 0–1.
    """
    indices = {}

    B02 = bands.get("B02")  # Blue
    B03 = bands.get("B03")  # Green
    B04 = bands.get("B04")  # Red
    B08 = bands.get("B08")  # NIR
    B11 = bands.get("B11")  # SWIR-1
    B12 = bands.get("B12")  # SWIR-2

    # NDVI — vegetation vigour
    if B08 is not None and B04 is not None:
        indices["NDVI"] = safe_ratio(B08 - B04, B08 + B04)

    # NDWI — open water (McFeeters)
    if B03 is not None and B08 is not None:
        indices["NDWI"] = safe_ratio(B03 - B08, B03 + B08)

    # NDBI — built-up / impervious
    if B11 is not None and B08 is not None:
        indices["NDBI"] = safe_ratio(B11 - B08, B11 + B08)

    # EVI — enhanced vegetation index
    if B08 is not None and B04 is not None and B02 is not None:
        num = 2.5 * (B08 - B04)
        den = B08 + 6.0 * B04 - 7.5 * B02 + 1.0
        indices["EVI"] = safe_ratio(num, den)

    # SAVI — soil-adjusted vegetation (L=0.5)
    if B08 is not None and B04 is not None:
        L = 0.5
        indices["SAVI"] = safe_ratio(
            1.5 * (B08 - B04),
            B08 + B04 + L
        )

    # MNDWI — modified NDWI (Xu 2006) uses Green/SWIR-1
    if B03 is not None and B11 is not None:
        indices["MNDWI"] = safe_ratio(B03 - B11, B03 + B11)

    return {k: np.clip(v, -1, 1) for k, v in indices.items()}


def true_color_rgb(bands: dict, gamma: float = 2.2) -> Optional[np.ndarray]:
    """Return uint8 (H, W, 3) true-color RGB from B04/B03/B02."""
    r, g, b = bands.get("B04"), bands.get("B03"), bands.get("B02")
    if r is None or g is None or b is None:
        return None
    # Stretch to 2–98th percentile then gamma
    def stretch(arr):
        p2, p98 = np.percentile(arr[arr > 0], [2, 98]) if (arr > 0).any() else (0, 1)
        arr = np.clip((arr - p2) / max(p98 - p2, 1e-6), 0, 1)
        return (arr ** (1 / gamma) * 255).astype(np.uint8)
    return np.stack([stretch(r), stretch(g), stretch(b)], axis=-1)


def false_color_rgb(bands: dict) -> Optional[np.ndarray]:
    """NIR/Red/Green false-color composite — vegetation in red."""
    nir, r, g = bands.get("B08"), bands.get("B04"), bands.get("B03")
    if nir is None or r is None or g is None:
        return None
    def stretch(arr):
        p2, p98 = np.percentile(arr[arr > 0], [2, 98]) if (arr > 0).any() else (0, 1)
        return np.clip((arr - p2) / max(p98 - p2, 1e-6), 0, 1)
    rgb = np.stack([stretch(nir), stretch(r), stretch(g)], axis=-1)
    return (rgb * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Scene extraction
# ---------------------------------------------------------------------------

def extract_scene(item: dict, aoi: AOI, token: str = "") -> Optional[SceneResult]:
    """
    Given a STAC item, read all bands and compute indices for the AOI.
    Returns a SceneResult.
    """
    assets = item.get("assets", {})
    props  = item.get("properties", {})
    item_id    = item.get("id", "unknown")
    date       = props.get("datetime", "")[:10]
    cloud_cover = float(props.get("eo:cloud_cover", -1))

    log.info("Extracting scene %s (%s, %.1f%% cloud)", item_id, date, cloud_cover)

    # Read SCL first to get CRS/shape reference
    scl_href = assets.get("SCL", {}).get("href", "")
    if scl_href and token:
        scl_href = sign_url(scl_href, token)

    # Read 10m bands
    bands = {}
    ref_shape = None
    for band in ["B04", "B03", "B02", "B08"]:
        href = assets.get(band, {}).get("href", "")
        if not href:
            continue
        if token:
            href = sign_url(href, token)
        arr = read_band(f"/vsicurl/{href}", aoi)
        if arr is not None:
            bands[band] = arr
            if ref_shape is None:
                ref_shape = arr.shape

    if not bands:
        log.warning("No bands read for scene %s", item_id)
        return None

    if ref_shape is None:
        return None

    # Read 20m bands and upscale
    for band in ["B11", "B12"]:
        href = assets.get(band, {}).get("href", "")
        if not href:
            continue
        if token:
            href = sign_url(href, token)
        arr = read_band(f"/vsicurl/{href}", aoi)
        if arr is not None:
            bands[band] = align_to_shape(arr, ref_shape)

    # SCL mask
    scl = None
    if scl_href:
        scl = read_scl(f"/vsicurl/{scl_href}", aoi)
        if scl is not None:
            scl = align_to_shape(scl, ref_shape)

    mask = scl_to_mask(scl)
    if mask is None:
        # Fallback: mask where all 10m bands are > 0
        mask = np.ones(ref_shape, dtype=bool)
        for b in ["B04", "B03", "B02", "B08"]:
            if b in bands:
                mask &= (bands[b] > 0)

    # Indices
    indices = compute_indices(bands)

    # Get CRS from first band href
    crs_str = "EPSG:32618"  # default
    try:
        first_href = list(assets.values())[0].get("href", "")
        if first_href and token:
            first_href = sign_url(first_href, token)
        with rasterio.open(f"/vsicurl/{first_href}") as src:
            crs_str = src.crs.to_string()
    except Exception:
        pass

    result = SceneResult(
        bands=bands,
        indices=indices,
        mask=mask,
        item_id=item_id,
        date=date,
        cloud_cover=cloud_cover,
        aoi=aoi,
        crs=crs_str,
    )
    result.stats = compute_stats(result)
    return result


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_stats(scene: SceneResult) -> dict:
    """Compute per-index and per-band statistics for LLM consumption."""
    mask = scene.mask
    n_valid = int(mask.sum())
    n_total = int(mask.size)

    stats = {
        "aoi": scene.aoi.name,
        "date": scene.date,
        "item_id": scene.item_id,
        "cloud_cover_pct": round(scene.cloud_cover, 1),
        "valid_px": n_valid,
        "total_px": n_total,
        "coverage_pct": round(100 * n_valid / max(n_total, 1), 1),
        "bands_read": sorted(scene.bands.keys()),
        "indices": {},
    }

    for name, arr in scene.indices.items():
        valid = arr[mask]
        if len(valid) == 0:
            continue
        stats["indices"][name] = {
            "mean":   round(float(valid.mean()), 4),
            "std":    round(float(valid.std()),  4),
            "median": round(float(np.median(valid)), 4),
            "p10":    round(float(np.percentile(valid, 10)), 4),
            "p90":    round(float(np.percentile(valid, 90)), 4),
            "pct_positive": round(float((valid > 0).mean() * 100), 1),
        }

    # Land cover proxies from index thresholds
    if "NDVI" in scene.indices:
        ndvi = scene.indices["NDVI"][mask]
        stats["land_cover_proxy"] = {
            "dense_veg_pct":   round(float((ndvi > 0.5).mean() * 100), 1),
            "sparse_veg_pct":  round(float(((ndvi > 0.2) & (ndvi <= 0.5)).mean() * 100), 1),
            "barren_pct":      round(float(((ndvi >= -0.1) & (ndvi <= 0.2)).mean() * 100), 1),
            "water_snow_pct":  round(float((ndvi < -0.1).mean() * 100), 1),
        }
    if "NDWI" in scene.indices:
        ndwi = scene.indices["NDWI"][mask]
        stats["water_pct"] = round(float((ndwi > 0).mean() * 100), 1)

    if "NDBI" in scene.indices:
        ndbi = scene.indices["NDBI"][mask]
        stats["built_up_pct"] = round(float((ndbi > 0).mean() * 100), 1)

    return stats


def compute_change_stats(s1: SceneResult, s2: SceneResult) -> dict:
    """Compare two scenes: per-index mean change and area shifts."""
    stats = {
        "aoi":    s1.aoi.name,
        "date_t1": s1.date,
        "date_t2": s2.date,
        "index_changes": {},
        "land_cover_change": {},
    }

    for idx in set(s1.indices) & set(s2.indices):
        a1 = s1.indices[idx][s1.mask]
        a2 = s2.indices[idx][s2.mask]
        if len(a1) == 0 or len(a2) == 0:
            continue
        stats["index_changes"][idx] = {
            "mean_t1":   round(float(a1.mean()), 4),
            "mean_t2":   round(float(a2.mean()), 4),
            "delta":     round(float(a2.mean() - a1.mean()), 4),
            "delta_pct": round(float((a2.mean() - a1.mean()) / max(abs(a1.mean()), 1e-6) * 100), 1),
        }

    lc1 = s1.stats.get("land_cover_proxy", {})
    lc2 = s2.stats.get("land_cover_proxy", {})
    for k in lc1:
        if k in lc2:
            stats["land_cover_change"][k] = {
                "t1_pct": lc1[k], "t2_pct": lc2[k],
                "delta":  round(lc2[k] - lc1[k], 1),
            }

    return stats


# ---------------------------------------------------------------------------
# Ollama Cloud
# ---------------------------------------------------------------------------

def query_ollama(
    stats: dict,
    change_stats: Optional[dict] = None,
    model: str = "llama3.2",
    host: str = "https://api.ollama.ai",
    api_key: Optional[str] = None,
    task: str = "interpret",
) -> str:
    """
    Send spectral stats to Ollama Cloud (or local) for interpretation.

    Cloud (api_key set): POST /v1/chat/completions with Bearer auth
    Local (no api_key):  POST /api/chat
    """
    if task == "change" and change_stats:
        prompt = f"""You are a remote sensing analyst interpreting Sentinel-2 spectral change signals.

Spectral statistics — {change_stats['date_t1']} vs {change_stats['date_t2']}:
{json.dumps(change_stats, indent=2)}

Scene details t1:
{json.dumps({k: v for k, v in stats.items() if k != 'indices'}, indent=2)}

Provide:
1. What the index changes suggest about land-cover or environmental change
2. Which indices show the strongest signal and what that implies
3. Likely drivers of any vegetation, water, or built-up shifts
4. Confidence and caveats (cloud cover, seasonality, etc.)
5. Recommended follow-up analyses

Be quantitative. Reference specific index values. Do not fabricate causes."""

    elif task == "report":
        prompt = f"""You are a remote sensing scientist. Write a concise technical paragraph (4-6 sentences)
for a methods/results section based on this Sentinel-2 L2A spectral analysis:

{json.dumps(stats, indent=2)}

Include: AOI, acquisition date, cloud cover, bands used, key index values, and landscape interpretation."""

    else:
        prompt = f"""You are a remote sensing analyst. Interpret these Sentinel-2 L2A spectral statistics
for a study area. All indices are computed from surface reflectance (L2A).

{json.dumps(stats, indent=2)}

Provide:
1. Overall landscape character based on NDVI, NDWI, NDBI values
2. Estimated land cover composition (vegetation, water, built-up, bare soil)
3. Data quality assessment (cloud cover, coverage)
4. Notable spectral patterns or anomalies
5. Suggested follow-up analyses"""

    messages = [{"role": "user", "content": prompt}]

    if api_key:
        url = f"{host.rstrip('/')}/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {"model": model, "messages": messages, "stream": False}
        try:
            r = httpx.post(url, json=payload, headers=headers, timeout=120)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as e:
            return f"[HTTP {e.response.status_code}]: {e.response.text[:300]}"
        except Exception as e:
            return f"[Ollama Cloud error]: {e}"
    else:
        url = f"{host.rstrip('/')}/api/chat"
        payload = {"model": model, "messages": messages, "stream": False}
        try:
            r = httpx.post(url, json=payload, timeout=120)
            r.raise_for_status()
            return r.json()["message"]["content"]
        except httpx.ConnectError:
            return f"[Ollama not reachable at {host} — ensure 'ollama serve' is running]"
        except Exception as e:
            return f"[Ollama local error]: {e}"


# ---------------------------------------------------------------------------
# High-level pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    aoi: AOI,
    date_range: str,
    max_cloud: float = 20.0,
    compare_date_range: Optional[str] = None,
    ollama_model: str = "llama3.2",
    ollama_host: str = "https://api.ollama.ai",
    ollama_api_key: Optional[str] = None,
    task: str = "interpret",
) -> dict:
    """
    Full pipeline:
      1. STAC search on Planetary Computer
      2. SAS token sign
      3. Read bands + compute indices for best scene
      4. Optionally repeat for a second date range (change detection)
      5. Compute stats and query Ollama
    """
    results = {"aoi": aoi}

    token = get_sas_token()

    # Primary scene
    items1 = search_scenes(aoi, date_range, max_cloud=max_cloud)
    if not items1:
        results["error"] = f"No scenes found for {aoi.name} in {date_range} with cloud < {max_cloud}%"
        return results

    item1 = best_scene(items1)
    scene1 = extract_scene(item1, aoi, token)
    if scene1 is None:
        results["error"] = "Scene extraction failed"
        return results

    results["scene1"] = scene1
    results["items1"] = items1

    # Optional comparison scene
    scene2 = None
    if compare_date_range:
        items2 = search_scenes(aoi, compare_date_range, max_cloud=max_cloud)
        if items2:
            item2 = best_scene(items2)
            scene2 = extract_scene(item2, aoi, token)
            results["scene2"] = scene2
            results["items2"] = items2

    # Stats
    change_stats = None
    if scene2:
        change_stats = compute_change_stats(scene1, scene2)
        results["change_stats"] = change_stats

    # Ollama
    effective_task = "change" if change_stats else task
    llm_response = query_ollama(
        scene1.stats,
        change_stats=change_stats,
        model=ollama_model,
        host=ollama_host,
        api_key=ollama_api_key,
        task=effective_task,
    )
    results["llm_response"] = llm_response

    return results
