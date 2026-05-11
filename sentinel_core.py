"""
sentinel_core.py  v2
--------------------
Sentinel-2 L2A via Microsoft Planetary Computer STAC.
No GEE. No sign-in. SAS token public read access.
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
    from rasterio.windows import from_bounds as window_from_bounds
    import rasterio.windows
except ImportError as e:
    raise ImportError("pip install rasterio") from e

try:
    from pyproj import Transformer
    from shapely.geometry import box
except ImportError as e:
    raise ImportError("pip install pyproj shapely") from e

PC_STAC_URL  = "https://planetarycomputer.microsoft.com/api/stac/v1"
PC_TOKEN_URL = "https://planetarycomputer.microsoft.com/api/sas/v1/token"
COLLECTION   = "sentinel-2-l2a"
SCL_VALID    = {4, 5, 6, 7, 11}
HIST_BINS    = np.linspace(-1, 1, 41)

AOI_CONTEXT = {
    "Catonsville, MD":            "Mid-Atlantic suburban/urban fringe, mixed deciduous forest and residential development.",
    "Baltimore, MD":              "Dense Mid-Atlantic city with active port, urban heat island, Chesapeake Bay tributaries.",
    "Washington DC":              "Dense federal urban core, Potomac/Anacostia rivers, significant impervious cover.",
    "New York City":              "High-density coastal metropolis, Hudson/East rivers, major urban heat island.",
    "Chesapeake Bay":             "Largest US estuary, mixed estuarine water, tidal wetlands, agricultural watershed.",
    "Atlanta, GA":                "Rapidly urbanising Southern city, mixed pine/hardwood forest, strong urban heat island.",
    "Houston, TX":                "Low-elevation Gulf Coast city, petrochemical industry, highly flood-prone, urban sprawl.",
    "Skaftafellsjökull, Iceland": "Active outlet glacier on Vatnajökull ice cap. Expect NDSI>0.4 for clean ice/snow, low NDVI, meltwater channels (high NDWI near terminus), proglacial lake. Key signals: glacier retreat, bare ice exposure, supraglacial melt ponds.",
    "Strait of Hormuz":           "Strategic 55km-wide waterway between Iran and Oman. Arid rocky coastline, shallow turbid water, strategic oil infrastructure. Expect low NDVI, high NDBI near ports, MNDWI distinguishes water from bare rock. Monitor ship traffic, coastal turbidity, shoreline change.",
}


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

    def area_km2(self):
        import math
        lon_span = abs(self.lon_max - self.lon_min)
        lat_span = abs(self.lat_max - self.lat_min)
        lat_mid  = (self.lat_min + self.lat_max) / 2
        km_per_lon = 111.32 * abs(math.cos(math.radians(lat_mid)))
        return round(lon_span * km_per_lon * lat_span * 111.32, 1)

    def context(self):
        return AOI_CONTEXT.get(self.name, "")


@dataclass
class SceneResult:
    bands:       dict
    indices:     dict
    mask:        np.ndarray
    item_id:     str
    date:        str
    cloud_cover: float
    aoi:         AOI
    crs:         str
    stats:       dict = field(default_factory=dict)


_token_cache: dict = {}

def get_sas_token(collection: str = COLLECTION) -> str:
    if collection in _token_cache:
        return _token_cache[collection]
    try:
        r = requests.get(f"{PC_TOKEN_URL}/{collection}", timeout=15)
        r.raise_for_status()
        token = r.json().get("token", "")
        _token_cache[collection] = token
        return token
    except Exception as e:
        log.warning("SAS token failed: %s", e)
        return ""

def sign_url(href: str, token: str) -> str:
    if not token or "?" in href:
        return href
    return f"{href}?{token}"

def search_scenes(aoi, date_range, max_cloud=20.0, max_items=8):
    payload = {
        "collections": [COLLECTION],
        "bbox": aoi.bbox(),
        "datetime": date_range,
        "query": {"eo:cloud_cover": {"lt": max_cloud}},
        "sortby": [{"field": "eo:cloud_cover", "direction": "asc"}],
        "limit": max_items,
    }
    log.info("STAC search: %s %s cloud<%.0f%%", aoi.name, date_range, max_cloud)
    r = requests.post(f"{PC_STAC_URL}/search", json=payload, timeout=30)
    r.raise_for_status()
    items = r.json().get("features", [])
    log.info("Found %d scene(s)", len(items))
    return items

def best_scene(items):
    if not items:
        return None
    return sorted(items, key=lambda x: x["properties"].get("eo:cloud_cover", 99))[0]

def scene_list_summary(items):
    out = []
    for it in items:
        p = it.get("properties", {})
        out.append({"id": it.get("id",""), "date": p.get("datetime","")[:10],
                    "cloud": round(p.get("eo:cloud_cover", -1), 1)})
    return sorted(out, key=lambda x: x["date"])

def _reproject_bbox(aoi, native_epsg):
    if native_epsg == 4326:
        return aoi.lon_min, aoi.lat_min, aoi.lon_max, aoi.lat_max
    tr = Transformer.from_crs(4326, native_epsg, always_xy=True)
    x0, y0 = tr.transform(aoi.lon_min, aoi.lat_min)
    x1, y1 = tr.transform(aoi.lon_max, aoi.lat_max)
    return min(x0,x1), min(y0,y1), max(x0,x1), max(y0,y1)

def read_band(href, aoi, scale=10000.0):
    try:
        with rasterio.open(href) as src:
            epsg = src.crs.to_epsg() or 4326
            left, bottom, right, top = _reproject_bbox(aoi, epsg)
            win = window_from_bounds(left, bottom, right, top, src.transform)
            win = win.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
            if win.width < 1 or win.height < 1:
                return None
            data = src.read(1, window=win).astype(np.float32)
            return np.clip(data / scale, 0, 1)
    except Exception as e:
        log.debug("Band read failed %s: %s", href.split("/")[-1], e)
        return None

def read_scl(href, aoi):
    try:
        with rasterio.open(href) as src:
            epsg = src.crs.to_epsg() or 4326
            left, bottom, right, top = _reproject_bbox(aoi, epsg)
            win = window_from_bounds(left, bottom, right, top, src.transform)
            win = win.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
            if win.width < 1 or win.height < 1:
                return None
            return src.read(1, window=win).astype(np.uint8)
    except Exception as e:
        log.debug("SCL failed: %s", e)
        return None

def align_shape(arr, target):
    if arr is None or arr.shape == target:
        return arr
    try:
        from skimage.transform import resize
        return resize(arr, target, order=0, anti_aliasing=False,
                      preserve_range=True).astype(arr.dtype)
    except ImportError:
        ry = max(1, target[0] // arr.shape[0])
        rx = max(1, target[1] // arr.shape[1])
        return np.repeat(np.repeat(arr, ry, 0), rx, 1)[:target[0], :target[1]]

def scl_to_mask(scl):
    if scl is None:
        return None
    mask = np.zeros(scl.shape, dtype=bool)
    for cls in SCL_VALID:
        mask |= (scl == cls)
    return mask

def _ratio(num, denom):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(denom) > 1e-6, num / denom, 0.0).astype(np.float32)

def compute_indices(bands):
    B02=bands.get("B02"); B03=bands.get("B03"); B04=bands.get("B04")
    B08=bands.get("B08"); B11=bands.get("B11"); B12=bands.get("B12")
    idx = {}
    if B08 is not None and B04 is not None:
        idx["NDVI"]  = _ratio(B08-B04, B08+B04)
    if B08 is not None and B04 is not None and B02 is not None:
        idx["EVI"]   = _ratio(2.5*(B08-B04), B08+6*B04-7.5*B02+1)
    if B08 is not None and B04 is not None:
        idx["SAVI"]  = _ratio(1.5*(B08-B04), B08+B04+0.5)
    if B03 is not None and B08 is not None:
        idx["NDWI"]  = _ratio(B03-B08, B03+B08)
    if B03 is not None and B11 is not None:
        idx["MNDWI"] = _ratio(B03-B11, B03+B11)
    if B11 is not None and B08 is not None:
        idx["NDBI"]  = _ratio(B11-B08, B11+B08)
    if B04 is not None and B08 is not None and B02 is not None and B11 is not None:
        idx["BSI"]   = _ratio((B11+B04)-(B08+B02), (B11+B04)+(B08+B02))
    if B03 is not None and B11 is not None:
        idx["NDSI"]  = _ratio(B03-B11, B03+B11)
    if B08 is not None and B12 is not None:
        idx["NBR"]   = _ratio(B08-B12, B08+B12)
    if B04 is not None and B08 is not None:
        denom = (0.1-B04)**2 + (0.06-B08)**2
        with np.errstate(divide="ignore", invalid="ignore"):
            bai = np.where(denom > 1e-8, 1.0/denom, 0.0).astype(np.float32)
        idx["BAI"] = np.clip(np.log1p(bai) / 10.0, 0, 1)
    return {k: np.clip(v, -1, 1) for k, v in idx.items()}

def _stretch(arr, p_low=2, p_high=98):
    valid = arr[arr > 0]
    if len(valid) == 0:
        return arr
    lo, hi = np.percentile(valid, [p_low, p_high])
    return np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1)

def true_color_rgb(bands):
    r,g,b = bands.get("B04"), bands.get("B03"), bands.get("B02")
    if r is None or g is None or b is None:
        return None
    rgb = np.stack([_stretch(r), _stretch(g), _stretch(b)], axis=-1)
    return (np.clip(rgb**0.454, 0, 1) * 255).astype(np.uint8)

def false_color_rgb(bands):
    nir,r,g = bands.get("B08"), bands.get("B04"), bands.get("B03")
    if nir is None or r is None or g is None:
        return None
    rgb = np.stack([_stretch(nir), _stretch(r), _stretch(g)], axis=-1)
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

def swir_composite(bands):
    s1,nir,r = bands.get("B11"), bands.get("B08"), bands.get("B04")
    if s1 is None or nir is None or r is None:
        return None
    rgb = np.stack([_stretch(s1), _stretch(nir), _stretch(r)], axis=-1)
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

def index_colormap(arr, mask, cmap_name, vmin=-1, vmax=1):
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    masked = np.where(mask, arr, np.nan)
    norm   = mcolors.Normalize(vmin=vmin, vmax=vmax)
    rgba   = cm.get_cmap(cmap_name)(norm(masked), bytes=True)
    rgba[~mask] = [15, 20, 30, 255]
    return rgba[:, :, :3]

def extract_scene(item, aoi, token=""):
    assets = item.get("assets", {})
    props  = item.get("properties", {})
    item_id     = item.get("id", "unknown")
    date        = props.get("datetime", "")[:10]
    cloud_cover = float(props.get("eo:cloud_cover", -1))
    log.info("Extracting %s (%s %.1f%% cloud)", item_id, date, cloud_cover)

    def href(b):
        h = assets.get(b, {}).get("href", "")
        return f"/vsicurl/{sign_url(h, token)}" if h else ""

    bands = {}
    ref_shape = None
    for b in ["B04","B03","B02","B08"]:
        h = href(b)
        if not h: continue
        arr = read_band(h, aoi)
        if arr is not None and arr.size > 0:
            bands[b] = arr
            if ref_shape is None: ref_shape = arr.shape

    if ref_shape is None or ref_shape[0] < 2 or ref_shape[1] < 2:
        log.warning("No usable 10m bands for %s", item_id)
        return None

    for b in ["B11","B12"]:
        h = href(b)
        if not h: continue
        arr = read_band(h, aoi)
        if arr is not None:
            bands[b] = align_shape(arr, ref_shape)

    scl_h = href("SCL")
    scl   = read_scl(scl_h, aoi) if scl_h else None
    if scl is not None:
        scl = align_shape(scl, ref_shape)
    mask = scl_to_mask(scl)
    if mask is None:
        mask = np.ones(ref_shape, dtype=bool)
        for b in ["B04","B03","B02"]:
            if b in bands: mask &= (bands[b] > 0)

    indices = compute_indices(bands)

    crs_str = "EPSG:32618"
    try:
        first_h = href(list(assets.keys())[0])
        if first_h:
            with rasterio.open(first_h) as src:
                crs_str = src.crs.to_string()
    except Exception:
        pass

    result = SceneResult(bands=bands, indices=indices, mask=mask,
                         item_id=item_id, date=date, cloud_cover=cloud_cover,
                         aoi=aoi, crs=crs_str)
    result.stats = compute_stats(result)
    return result

def compute_stats(scene):
    mask    = scene.mask
    n_valid = int(mask.sum())
    n_total = int(mask.size)
    stats = {
        "aoi":          scene.aoi.name,
        "aoi_area_km2": scene.aoi.area_km2(),
        "aoi_context":  scene.aoi.context(),
        "date":         scene.date,
        "item_id":      scene.item_id,
        "cloud_pct":    round(scene.cloud_cover, 1),
        "valid_px":     n_valid,
        "total_px":     n_total,
        "coverage_pct": round(100 * n_valid / max(n_total, 1), 1),
        "pixel_res_m":  10,
        "bands_read":   sorted(scene.bands.keys()),
        "indices":      {},
        "histograms":   {},
    }
    for name, arr in scene.indices.items():
        valid = arr[mask]
        if len(valid) < 4: continue
        hist, _ = np.histogram(valid, bins=HIST_BINS)
        stats["indices"][name] = {
            "mean":         round(float(valid.mean()), 4),
            "std":          round(float(valid.std()),  4),
            "median":       round(float(np.median(valid)), 4),
            "p10":          round(float(np.percentile(valid, 10)), 4),
            "p25":          round(float(np.percentile(valid, 25)), 4),
            "p75":          round(float(np.percentile(valid, 75)), 4),
            "p90":          round(float(np.percentile(valid, 90)), 4),
            "pct_positive": round(float((valid > 0).mean() * 100), 1),
        }
        stats["histograms"][name] = hist.tolist()

    if "NDVI" in scene.indices:
        v = scene.indices["NDVI"][mask]
        stats["land_cover"] = {
            "dense_veg_pct":  round(float((v > 0.5).mean()  * 100), 1),
            "sparse_veg_pct": round(float(((v>0.2)&(v<=0.5)).mean()*100),1),
            "barren_pct":     round(float(((v>=-0.1)&(v<=0.2)).mean()*100),1),
            "water_snow_pct": round(float((v<-0.1).mean()*100),1),
        }
    if "NDWI" in scene.indices:
        stats["water_pct"]    = round(float((scene.indices["NDWI"][mask]>0).mean()*100),1)
    if "NDBI" in scene.indices:
        stats["built_up_pct"] = round(float((scene.indices["NDBI"][mask]>0).mean()*100),1)
    if "NDSI" in scene.indices:
        stats["snow_ice_pct"] = round(float((scene.indices["NDSI"][mask]>0.4).mean()*100),1)
    if "NBR" in scene.indices:
        stats["burn_pct"]     = round(float((scene.indices["NBR"][mask]<-0.1).mean()*100),1)
    return stats

def _days_apart(d1, d2):
    try:
        from datetime import date
        return abs((date.fromisoformat(d2)-date.fromisoformat(d1)).days)
    except Exception:
        return -1

def compute_change_stats(s1, s2):
    out = {
        "aoi":              s1.aoi.name,
        "aoi_context":      s1.aoi.context(),
        "date_t1":          s1.date,
        "date_t2":          s2.date,
        "days_apart":       _days_apart(s1.date, s2.date),
        "index_changes":    {},
        "land_cover_change":{},
    }
    for idx in sorted(set(s1.indices) & set(s2.indices)):
        a1 = s1.indices[idx][s1.mask]
        a2 = s2.indices[idx][s2.mask]
        if len(a1) < 4 or len(a2) < 4: continue
        delta = float(a2.mean() - a1.mean())
        out["index_changes"][idx] = {
            "mean_t1":   round(float(a1.mean()),4),
            "mean_t2":   round(float(a2.mean()),4),
            "delta":     round(delta,4),
            "delta_pct": round(delta/max(abs(float(a1.mean())),1e-6)*100,1),
            "std_t1":    round(float(a1.std()),4),
            "std_t2":    round(float(a2.std()),4),
        }
    lc1 = s1.stats.get("land_cover",{})
    lc2 = s2.stats.get("land_cover",{})
    for k in lc1:
        if k in lc2:
            out["land_cover_change"][k] = {"t1":lc1[k],"t2":lc2[k],"delta":round(lc2[k]-lc1[k],1)}
    return out

def change_map(s1, s2, index="NDVI"):
    i1 = s1.indices.get(index)
    i2 = s2.indices.get(index)
    if i1 is None or i2 is None: return None
    if i1.shape != i2.shape:
        i2 = align_shape(i2, i1.shape)
    joint = s1.mask & align_shape(s2.mask.astype(np.uint8), s1.mask.shape).astype(bool)
    return np.where(joint, i2-i1, np.nan).astype(np.float32)

_SYSTEM = (
    "You are an expert remote sensing scientist specialising in Sentinel-2 spectral analysis. "
    "You give precise, quantitative, actionable interpretations grounded in the data provided. "
    "You never fabricate causes or invent data not present. "
    "You acknowledge uncertainty and seasonal confounders where relevant."
)

def query_ollama(stats, change_stats=None, model="gpt-oss:20b-cloud",
                 host="https://ollama.com", api_key=None, task="interpret"):
    aoi_ctx  = stats.get("aoi_context","")
    aoi_name = stats.get("aoi","")
    ctx_line = f"\nKnown AOI context: {aoi_ctx}" if aoi_ctx else ""
    clean_stats = {k:v for k,v in stats.items() if k not in ("histograms","aoi_context")}

    if task == "change" and change_stats:
        prompt = f"""Sentinel-2 L2A multi-temporal analysis.
AOI: {aoi_name}{ctx_line}
Period: {change_stats['date_t1']} → {change_stats['date_t2']} ({change_stats['days_apart']} days)

Index changes:
{json.dumps(change_stats['index_changes'], indent=2)}

Land cover shifts:
{json.dumps(change_stats.get('land_cover_change',{}), indent=2)}

Baseline stats ({change_stats['date_t1']}):
{json.dumps(clean_stats, indent=2)}

Structure your response with these headers:
## Magnitude
## Interpretation
## Seasonality
## Confidence & Caveats
## Priority Follow-ups
## Headline (one sentence)"""

    elif task == "report":
        prompt = f"""Write a technical remote sensing results paragraph (5-7 sentences) for a scientific report.
AOI: {aoi_name}{ctx_line}
Data: Sentinel-2 L2A via Microsoft Planetary Computer
{json.dumps(clean_stats, indent=2)}
Use passive academic voice. Cover: date, cloud cover, resolution, bands, all index values, land cover, data quality."""

    else:
        prompt = f"""Interpret Sentinel-2 L2A surface reflectance statistics for {aoi_name}.{ctx_line}

{json.dumps(clean_stats, indent=2)}

Structure your response with these headers:
## Landscape Character
## Land Cover Composition
## Environmental Signals
## Data Quality
## Recommended Next Steps
## Headline (one sentence)"""

    messages = [
        {"role":"system","content":_SYSTEM},
        {"role":"user",  "content":prompt},
    ]
    url     = f"{host.rstrip('/')}/api/chat"
    headers = {"Content-Type":"application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        r = httpx.post(url, json={"model":model,"messages":messages,"stream":False},
                       headers=headers, timeout=180)
        r.raise_for_status()
        return r.json()["message"]["content"]
    except httpx.HTTPStatusError as e:
        return f"[HTTP {e.response.status_code}]: {e.response.text[:400]}"
    except Exception as e:
        return f"[Ollama error]: {e}"

def run_pipeline(aoi, date_range, max_cloud=20.0, compare_date_range=None,
                 ollama_model="gpt-oss:20b-cloud", ollama_host="https://ollama.com",
                 ollama_api_key=None, task="interpret"):
    results = {"aoi": aoi}
    token   = get_sas_token()
    items1  = search_scenes(aoi, date_range, max_cloud=max_cloud)
    if not items1:
        results["error"] = f"No scenes for {aoi.name} in {date_range} cloud<{max_cloud}%"
        return results
    item1  = best_scene(items1)
    scene1 = extract_scene(item1, aoi, token)
    if scene1 is None:
        results["error"] = "Scene extraction failed"
        return results
    results.update(scene1=scene1, items1=items1, scene_catalog=scene_list_summary(items1))
    scene2 = None
    change_stats = None
    if compare_date_range:
        items2 = search_scenes(aoi, compare_date_range, max_cloud=max_cloud)
        if items2:
            scene2 = extract_scene(best_scene(items2), aoi, token)
            if scene2:
                results.update(scene2=scene2, items2=items2)
                change_stats = compute_change_stats(scene1, scene2)
                results["change_stats"] = change_stats
    results["llm_response"] = query_ollama(
        scene1.stats, change_stats=change_stats, model=ollama_model,
        host=ollama_host, api_key=ollama_api_key,
        task="change" if change_stats else task)
    return results
