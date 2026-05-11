"""
Sentinel-2 Explorer · Microsoft Planetary Computer · Ollama Cloud
No GEE. No sign-in required.
"""

import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import streamlit as st

st.set_page_config(
    page_title="Sentinel-2 Explorer",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

:root {
    --bg:      #060a10;
    --surface: #0d1420;
    --border:  #162032;
    --accent:  #38bdf8;
    --green:   #4ade80;
    --amber:   #fbbf24;
    --red:     #f87171;
    --text:    #cbd5e1;
    --muted:   #475569;
}
html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background: var(--bg);
    color: var(--text);
}
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}
.title-block {
    border-left: 3px solid var(--accent);
    padding: 10px 0 10px 18px;
    margin-bottom: 20px;
}
.title-main {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.5rem;
    font-weight: 600;
    color: #f1f5f9;
    letter-spacing: -0.02em;
}
.title-sub {
    font-size: 0.8rem;
    color: var(--muted);
    margin-top: 3px;
}
.mono { font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; }
.section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--muted);
    border-bottom: 1px solid var(--border);
    padding-bottom: 5px;
    margin: 18px 0 10px;
}
.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
    gap: 10px;
    margin: 14px 0;
}
.metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-top: 2px solid var(--accent);
    border-radius: 6px;
    padding: 12px 14px;
}
.metric-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--muted);
}
.metric-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.3rem;
    color: var(--accent);
    margin-top: 2px;
}
.metric-sub { font-size: 0.68rem; color: var(--muted); }
.index-row {
    display: flex;
    align-items: center;
    padding: 6px 0;
    border-bottom: 1px solid var(--border);
    gap: 10px;
}
.index-name {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    width: 56px;
    color: var(--accent);
}
.index-bar-wrap { flex:1; background: var(--border); border-radius: 2px; height: 6px; position: relative; }
.index-bar { height: 6px; border-radius: 2px; }
.index-val {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    width: 52px;
    text-align: right;
    color: var(--text);
}
.delta-pos { color: var(--green); font-family:'IBM Plex Mono',monospace; font-size:0.72rem; }
.delta-neg { color: var(--red);   font-family:'IBM Plex Mono',monospace; font-size:0.72rem; }
.llm-box {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--green);
    border-radius: 6px;
    padding: 18px 22px;
    font-size: 0.88rem;
    line-height: 1.75;
    white-space: pre-wrap;
}
.badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 20px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem;
    font-weight: 600;
    letter-spacing: 0.04em;
}
.b-blue  { background:rgba(56,189,248,.12); color:var(--accent); border:1px solid rgba(56,189,248,.25); }
.b-green { background:rgba(74,222,128,.12); color:var(--green);  border:1px solid rgba(74,222,128,.25); }
.b-amber { background:rgba(251,191,36,.12); color:var(--amber);  border:1px solid rgba(251,191,36,.25); }
.b-red   { background:rgba(248,113,113,.12);color:var(--red);    border:1px solid rgba(248,113,113,.25);}
.stButton > button {
    background: var(--accent);
    color: #060a10;
    border: none;
    border-radius: 5px;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    font-size: 0.8rem;
    padding: 10px 20px;
    width: 100%;
    letter-spacing: 0.03em;
    transition: opacity .15s;
}
.stButton > button:hover { opacity: .85; }
div[data-testid="stNumberInput"] label,
div[data-testid="stTextInput"] label,
div[data-testid="stSelectbox"] label,
div[data-testid="stCheckbox"] label,
div[data-testid="stSlider"] label {
    font-size: 0.68rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--muted) !important;
    font-family: 'IBM Plex Mono', monospace !important;
}
</style>
""", unsafe_allow_html=True)


# ── Imports ───────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _import_core():
    from sentinel_core import (
        AOI, search_scenes, best_scene, extract_scene, get_sas_token,
        compute_change_stats, query_ollama, true_color_rgb, false_color_rgb,
        run_pipeline,
    )
    return dict(
        AOI=AOI, search_scenes=search_scenes, best_scene=best_scene,
        extract_scene=extract_scene, get_sas_token=get_sas_token,
        compute_change_stats=compute_change_stats, query_ollama=query_ollama,
        true_color_rgb=true_color_rgb, false_color_rgb=false_color_rgb,
        run_pipeline=run_pipeline,
    )


# ── Presets ───────────────────────────────────────────────────────────────────
PRESETS = {
    "Custom":                    None,
    "Catonsville, MD":           (-76.755, 39.240, -76.680, 39.290),
    "Baltimore, MD":             (-76.720, 39.250, -76.550, 39.380),
    "Washington DC":             (-77.120, 38.800, -76.910, 38.990),
    "New York City":             (-74.050, 40.680, -73.920, 40.800),
    "Chesapeake Bay":            (-76.500, 38.700, -76.200, 39.000),
    "Atlanta, GA":               (-84.500, 33.650, -84.300, 33.850),
    "Houston, TX":               (-95.500, 29.650, -95.200, 29.850),
    "Skaftafellsjökull, Iceland":(-17.050, 63.980, -16.850, 64.080),
    "Strait of Hormuz":          ( 56.050, 26.200,  57.050, 26.800),
}

TASKS = {
    "Landscape interpretation": "interpret",
    "Change detection":         "change",
    "Technical report":         "report",
}

INDEX_COLORS = {
    "NDVI":  "#4ade80",
    "NDWI":  "#38bdf8",
    "NDBI":  "#f87171",
    "EVI":   "#86efac",
    "SAVI":  "#6ee7b7",
    "MNDWI": "#7dd3fc",
}


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="title-block">
  <div class="title-main">🌍 Sentinel-2 Explorer</div>
  <div class="title-sub">
    Microsoft Planetary Computer · Sentinel-2 L2A · Ollama Cloud · No GEE
  </div>
</div>
<div style="margin-bottom:18px">
  <span class="badge b-blue">S2 L2A</span>&nbsp;
  <span class="badge b-green">NDVI · NDWI · NDBI · EVI</span>&nbsp;
  <span class="badge b-amber">PC STAC</span>&nbsp;
  <span class="badge b-blue">Ollama Cloud</span>
</div>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="section-label">Area of Interest</div>', unsafe_allow_html=True)
    preset = st.selectbox("Preset", list(PRESETS.keys()), index=1)

    if preset == "Custom":
        c1, c2 = st.columns(2)
        lon_min = c1.number_input("Lon min", value=-76.755, format="%.4f")
        lat_min = c2.number_input("Lat min", value=39.240,  format="%.4f")
        lon_max = c1.number_input("Lon max", value=-76.680, format="%.4f")
        lat_max = c2.number_input("Lat max", value=39.290,  format="%.4f")
        aoi_name = st.text_input("Name", "Custom AOI")
    else:
        lon_min, lat_min, lon_max, lat_max = PRESETS[preset]
        aoi_name = preset

    st.markdown('<div class="section-label">Date Range</div>', unsafe_allow_html=True)
    date1_start = st.date_input("Start date",  value=__import__("datetime").date(2023, 6, 1))
    date1_end   = st.date_input("End date",    value=__import__("datetime").date(2023, 9, 1))
    date_range1 = f"{date1_start}/{date1_end}"

    max_cloud = st.slider("Max cloud cover %", 0, 60, 20)

    enable_change = st.checkbox("Enable change detection")
    date_range2 = None
    if enable_change:
        st.markdown('<div class="section-label">Comparison Period</div>', unsafe_allow_html=True)
        date2_start = st.date_input("Compare start", value=__import__("datetime").date(2019, 6, 1))
        date2_end   = st.date_input("Compare end",   value=__import__("datetime").date(2019, 9, 1))
        date_range2 = f"{date2_start}/{date2_end}"

    st.markdown('<div class="section-label">Ollama</div>', unsafe_allow_html=True)

    _secret_key = ""
    try:
        _secret_key = st.secrets.get("OLLAMA_API_KEY", "")
    except Exception:
        pass

    use_cloud = st.checkbox("Ollama Cloud", value=bool(_secret_key))
    if use_cloud:
        ollama_host = st.text_input("Host", value="https://ollama.com")
        ollama_api_key = st.text_input("API key", value=_secret_key, type="password",
                                        help="Or set OLLAMA_API_KEY in Streamlit secrets")
    else:
        ollama_host    = st.text_input("Host", value="http://localhost:11434")
        ollama_api_key = ""

    ollama_model = st.text_input("Model", value="gpt-oss:20b-cloud",
                                  help="Cloud: gpt-oss:20b-cloud, llama3.2:cloud etc. Local: llama3.2")
    task_label   = st.selectbox("Task", list(TASKS.keys()),
                                 index=1 if enable_change else 0)
    task = TASKS[task_label]

    skip_llm = st.checkbox("Skip Ollama (imagery only)")

    st.markdown("---")
    run_btn = st.button("▶  Analyze", use_container_width=True)

    st.markdown("""
    <div style="font-size:0.6rem;color:#334155;margin-top:14px;line-height:1.6">
    Data: Microsoft Planetary Computer<br>
    Collection: sentinel-2-l2a<br>
    No sign-in required
    </div>""", unsafe_allow_html=True)


# ── Idle state ────────────────────────────────────────────────────────────────
if not run_btn:
    st.markdown("""
    <div style="background:#0d1420;border:1px dashed #162032;border-radius:10px;
                padding:52px 32px;text-align:center;margin-top:20px">
      <div style="font-size:3rem;margin-bottom:14px">🛰️</div>
      <div style="font-family:'IBM Plex Mono',monospace;font-size:1rem;color:#f1f5f9;margin-bottom:10px">
        Configure your AOI and hit Analyze
      </div>
      <div style="font-size:0.82rem;color:#475569;max-width:500px;margin:0 auto;line-height:1.7">
        Pulls Sentinel-2 L2A imagery directly from Microsoft Planetary Computer via STAC,
        computes NDVI · NDWI · NDBI · EVI · SAVI · MNDWI, and sends spectral statistics
        to Ollama for interpretation. No GEE. No authentication.
      </div>
      <div style="margin-top:22px;display:flex;gap:10px;justify-content:center;flex-wrap:wrap">
        <span class="badge b-blue">True color RGB</span>
        <span class="badge b-green">False color NIR</span>
        <span class="badge b-green">6 spectral indices</span>
        <span class="badge b-amber">Change detection</span>
        <span class="badge b-blue">LLM interpretation</span>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ── Run ───────────────────────────────────────────────────────────────────────
core = _import_core()
AOI  = core["AOI"]
aoi  = AOI(lon_min, lat_min, lon_max, lat_max, aoi_name)

status   = st.empty()
progress = st.progress(0)

def step(msg, pct):
    status.markdown(f'<div class="mono" style="color:#38bdf8">⟳ {msg}</div>',
                    unsafe_allow_html=True)
    progress.progress(pct)

try:
    step("Fetching SAS token from Planetary Computer…", 5)
    token = core["get_sas_token"]()

    step(f"Searching Sentinel-2 scenes: {date_range1} · cloud < {max_cloud}%…", 15)
    items1 = core["search_scenes"](aoi, date_range1, max_cloud=max_cloud)
    if not items1:
        progress.empty(); status.empty()
        st.error(f"No scenes found for {aoi_name} in {date_range1} with cloud < {max_cloud}%. "
                 "Try widening the date range or increasing max cloud cover.")
        st.stop()

    item1 = core["best_scene"](items1)
    scene_date1  = item1["properties"]["datetime"][:10]
    scene_cloud1 = item1["properties"].get("eo:cloud_cover", 0)

    step(f"Reading bands for {scene_date1} ({scene_cloud1:.1f}% cloud)…", 30)
    scene1 = core["extract_scene"](item1, aoi, token)
    if scene1 is None:
        progress.empty(); status.empty()
        st.error("Band extraction failed. Try a different date range.")
        st.stop()

    scene2 = None
    change_stats = None
    if enable_change and date_range2:
        step(f"Searching comparison scenes: {date_range2}…", 50)
        items2 = core["search_scenes"](aoi, date_range2, max_cloud=max_cloud)
        if items2:
            item2  = core["best_scene"](items2)
            step(f"Reading bands for comparison scene {item2['properties']['datetime'][:10]}…", 62)
            scene2 = core["extract_scene"](item2, aoi, token)
            if scene2:
                change_stats = core["compute_change_stats"](scene1, scene2)

    step("Computing composites…", 75)
    rgb_true  = core["true_color_rgb"](scene1.bands)
    rgb_false = core["false_color_rgb"](scene1.bands)

    llm_response = ""
    if not skip_llm:
        step(f"Querying {ollama_model}…", 88)
        effective_task = "change" if change_stats else task
        llm_response = core["query_ollama"](
            scene1.stats,
            change_stats=change_stats,
            model=ollama_model,
            host=ollama_host,
            api_key=ollama_api_key or None,
            task=effective_task,
        )

    progress.progress(100)
    status.empty()
    progress.empty()

except Exception as e:
    progress.empty(); status.empty()
    st.error(f"Pipeline error: {e}")
    st.exception(e)
    st.stop()


# ── Results header ────────────────────────────────────────────────────────────
cloud_badge_cls = "b-green" if scene_cloud1 < 10 else "b-amber" if scene_cloud1 < 25 else "b-red"

st.markdown(f"""
<div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;flex-wrap:wrap">
  <span style="font-family:'IBM Plex Mono',monospace;font-size:1.05rem;font-weight:600">{aoi_name}</span>
  <span class="badge b-blue">{scene_date1}</span>
  <span class="badge {cloud_badge_cls}">{scene_cloud1:.1f}% cloud</span>
  <span class="badge b-amber">{len(scene1.bands)} bands read</span>
  <span class="badge b-green">{scene1.stats.get('coverage_pct',0)}% valid px</span>
</div>
""", unsafe_allow_html=True)


# ── Metric cards ──────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Key Metrics</div>', unsafe_allow_html=True)

idx_stats = scene1.stats.get("indices", {})
lc        = scene1.stats.get("land_cover_proxy", {})

col1, col2, col3, col4, col5 = st.columns(5)
cards = [
    (col1, "NDVI",      f"{idx_stats.get('NDVI',{}).get('mean','—')}", "vegetation vigour"),
    (col2, "NDWI",      f"{idx_stats.get('NDWI',{}).get('mean','—')}", "open water"),
    (col3, "NDBI",      f"{idx_stats.get('NDBI',{}).get('mean','—')}", "built-up index"),
    (col4, "Dense veg", f"{lc.get('dense_veg_pct','—')}%", "NDVI > 0.5"),
    (col5, "Water",     f"{scene1.stats.get('water_pct','—')}%", "NDWI > 0"),
]
for col, label, val, sub in cards:
    col.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">{label}</div>
      <div class="metric-value">{val}</div>
      <div class="metric-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)


# ── Imagery row ───────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Imagery</div>', unsafe_allow_html=True)

ncols = 3 if scene2 else 2
img_cols = st.columns(ncols)

with img_cols[0]:
    st.caption(f"True Color (RGB) — {scene_date1}")
    if rgb_true is not None:
        st.image(rgb_true, use_container_width=True)
    else:
        st.info("B04/B03/B02 not all available")

with img_cols[1]:
    st.caption(f"False Color NIR (NIR/R/G) — {scene_date1}")
    if rgb_false is not None:
        st.image(rgb_false, use_container_width=True)
    else:
        st.info("B08/B04/B03 not all available")

if scene2 and ncols == 3:
    rgb_true2 = core["true_color_rgb"](scene2.bands)
    with img_cols[2]:
        st.caption(f"True Color — {scene2.date}")
        if rgb_true2 is not None:
            st.image(rgb_true2, use_container_width=True)
        else:
            st.info("True color unavailable")


# ── Index visualization ───────────────────────────────────────────────────────
st.markdown('<div class="section-label">Spectral Indices</div>', unsafe_allow_html=True)

try:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    available_indices = list(scene1.indices.keys())
    n_idx = len(available_indices)

    if n_idx > 0:
        fig, axes = plt.subplots(1, min(n_idx, 4), figsize=(4 * min(n_idx, 4), 3.5),
                                  facecolor="#060a10", constrained_layout=True)
        if n_idx == 1:
            axes = [axes]

        cmaps = {"NDVI": "RdYlGn", "NDWI": "Blues", "NDBI": "RdPu",
                 "EVI": "YlGn", "SAVI": "YlGn", "MNDWI": "Blues"}

        for ax, idx_name in zip(axes, available_indices[:4]):
            arr = scene1.indices[idx_name]
            masked = np.where(scene1.mask, arr, np.nan)
            cmap_name = cmaps.get(idx_name, "RdYlGn")
            im = ax.imshow(masked, cmap=cmap_name, vmin=-1, vmax=1)
            ax.set_title(idx_name, color="#38bdf8", fontsize=9,
                          fontfamily="monospace")
            ax.axis("off")
            plt.colorbar(im, ax=ax, shrink=0.75, pad=0.02)

        fig.patch.set_facecolor("#060a10")
        plt.savefig("/tmp/s2_indices.png", dpi=130, bbox_inches="tight",
                    facecolor="#060a10")
        st.image("/tmp/s2_indices.png", use_container_width=True)

except ImportError:
    st.info("Install matplotlib for index maps")


# ── Index stats table ─────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Index Statistics</div>', unsafe_allow_html=True)

if idx_stats:
    rows_html = ""
    for name, s in idx_stats.items():
        mean_val = s.get("mean", 0)
        color = INDEX_COLORS.get(name, "#38bdf8")
        # normalize mean [-1,1] to bar width [0,100%]
        bar_pct = int((mean_val + 1) / 2 * 100)
        # change delta if available
        delta_html = ""
        if change_stats:
            ic = change_stats.get("index_changes", {}).get(name, {})
            delta = ic.get("delta", None)
            if delta is not None:
                cls = "delta-pos" if delta >= 0 else "delta-neg"
                sign = "+" if delta >= 0 else ""
                delta_html = f'<span class="{cls}">{sign}{delta:.4f}</span>'

        rows_html += f"""
        <div class="index-row">
          <div class="index-name">{name}</div>
          <div class="index-bar-wrap">
            <div class="index-bar" style="width:{bar_pct}%;background:{color};opacity:0.8"></div>
          </div>
          <div class="index-val">{mean_val:.4f}</div>
          <div style="width:70px;text-align:right">{delta_html}</div>
        </div>"""

    st.markdown(rows_html, unsafe_allow_html=True)
    if change_stats:
        st.caption(f"Δ = {scene2.date} minus {scene1.date}")


# ── Land cover proxy ──────────────────────────────────────────────────────────
if lc:
    st.markdown('<div class="section-label">Land Cover Proxy (from NDVI thresholds)</div>',
                unsafe_allow_html=True)
    lc_cols = st.columns(4)
    lc_items = [
        ("Dense veg", lc.get("dense_veg_pct", 0), "#4ade80"),
        ("Sparse veg", lc.get("sparse_veg_pct", 0), "#86efac"),
        ("Barren/built", lc.get("barren_pct", 0), "#fbbf24"),
        ("Water/snow", lc.get("water_snow_pct", 0), "#38bdf8"),
    ]
    for col, (label, pct, color) in zip(lc_cols, lc_items):
        col.markdown(f"""
        <div class="metric-card" style="border-top-color:{color}">
          <div class="metric-label">{label}</div>
          <div class="metric-value" style="color:{color};font-size:1.1rem">{pct}%</div>
        </div>""", unsafe_allow_html=True)


# ── Change stats ──────────────────────────────────────────────────────────────
if change_stats:
    st.markdown(f'<div class="section-label">Change Detection — {scene1.date} → {scene2.date}</div>',
                unsafe_allow_html=True)
    ic = change_stats.get("index_changes", {})
    ch_cols = st.columns(min(len(ic), 4))
    for col, (name, vals) in zip(ch_cols, list(ic.items())[:4]):
        delta = vals.get("delta", 0)
        color = "#4ade80" if delta >= 0 else "#f87171"
        sign  = "+" if delta >= 0 else ""
        col.markdown(f"""
        <div class="metric-card" style="border-top-color:{color}">
          <div class="metric-label">{name} change</div>
          <div class="metric-value" style="color:{color};font-size:1.1rem">{sign}{delta:.4f}</div>
          <div class="metric-sub">{vals['mean_t1']:.3f} → {vals['mean_t2']:.3f}</div>
        </div>""", unsafe_allow_html=True)


# ── LLM ──────────────────────────────────────────────────────────────────────
if not skip_llm and llm_response:
    st.markdown('<div class="section-label">Ollama Interpretation</div>', unsafe_allow_html=True)
    endpoint_badge = "cloud" if ollama_api_key else "local"
    badge_cls = "b-green" if ollama_api_key else "b-blue"
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">
      <span style="font-size:1rem">🤖</span>
      <span class="mono" style="color:#475569">{ollama_model} · {task_label.lower()}</span>
      <span class="badge {badge_cls}">{endpoint_badge}</span>
    </div>
    <div class="llm-box">{llm_response}</div>
    """, unsafe_allow_html=True)
elif skip_llm:
    st.info("Ollama skipped.")


# ── Export ────────────────────────────────────────────────────────────────────
with st.expander("📋 Raw Statistics (JSON)"):
    export = {"scene1_stats": scene1.stats}
    if change_stats:
        export["change_stats"] = change_stats
    if llm_response:
        export["llm_response"] = llm_response
    st.json(export)
    st.download_button(
        "⬇ Download JSON",
        data=json.dumps(export, indent=2),
        file_name=f"s2_{aoi_name.replace(' ','_')}_{scene_date1}.json",
        mime="application/json",
    )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="font-size:0.65rem;color:#334155;text-align:center">
  Sentinel-2 L2A · Microsoft Planetary Computer ·
  <code>planetarycomputer.microsoft.com</code>
</div>
""", unsafe_allow_html=True)
