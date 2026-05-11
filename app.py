"""
Sentinel-2 Explorer v2 — Planetary Computer · Ollama Cloud
"""
import json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import streamlit as st

st.set_page_config(page_title="Sentinel-2 Explorer", page_icon="🌍",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500&display=swap');
:root{--bg:#060a10;--surf:#0d1420;--surf2:#111a2a;--bdr:#162032;
      --sky:#38bdf8;--grn:#4ade80;--amb:#fbbf24;--red:#f87171;--pur:#a78bfa;
      --txt:#cbd5e1;--mut:#475569;}
html,body,[class*="css"]{font-family:'IBM Plex Sans',sans-serif;background:var(--bg);color:var(--txt);}
[data-testid="stSidebar"]{background:var(--surf)!important;border-right:1px solid var(--bdr);}
[data-testid="stSidebar"] .stMarkdown p{font-size:.75rem;color:var(--mut);}
.hd{border-left:3px solid var(--sky);padding:8px 0 8px 16px;margin-bottom:16px;}
.hd-main{font-family:'IBM Plex Mono',monospace;font-size:1.45rem;font-weight:600;color:#f1f5f9;letter-spacing:-.02em;}
.hd-sub{font-size:.78rem;color:var(--mut);margin-top:2px;}
.mono{font-family:'IBM Plex Mono',monospace;font-size:.74rem;}
.sl{font-family:'IBM Plex Mono',monospace;font-size:.62rem;text-transform:uppercase;
    letter-spacing:.12em;color:var(--mut);border-bottom:1px solid var(--bdr);
    padding-bottom:4px;margin:16px 0 10px;}
.mc{background:var(--surf);border:1px solid var(--bdr);border-top:2px solid var(--sky);
    border-radius:6px;padding:11px 13px;}
.ml{font-family:'IBM Plex Mono',monospace;font-size:.6rem;text-transform:uppercase;
    letter-spacing:.1em;color:var(--mut);}
.mv{font-family:'IBM Plex Mono',monospace;font-size:1.25rem;color:var(--sky);margin-top:1px;}
.ms{font-size:.66rem;color:var(--mut);}
.badge{display:inline-block;padding:2px 8px;border-radius:20px;
       font-family:'IBM Plex Mono',monospace;font-size:.6rem;font-weight:600;letter-spacing:.04em;}
.bb{background:rgba(56,189,248,.1);color:var(--sky);border:1px solid rgba(56,189,248,.25);}
.bg{background:rgba(74,222,128,.1);color:var(--grn);border:1px solid rgba(74,222,128,.25);}
.ba{background:rgba(251,191,36,.1);color:var(--amb);border:1px solid rgba(251,191,36,.25);}
.br{background:rgba(248,113,113,.1);color:var(--red);border:1px solid rgba(248,113,113,.25);}
.bp{background:rgba(167,139,250,.1);color:var(--pur);border:1px solid rgba(167,139,250,.25);}
.llm{background:var(--surf);border:1px solid var(--bdr);border-left:3px solid var(--grn);
     border-radius:6px;padding:18px 22px;font-size:.87rem;line-height:1.8;white-space:pre-wrap;}
.llm h2{font-family:'IBM Plex Mono',monospace;font-size:.78rem;text-transform:uppercase;
         letter-spacing:.1em;color:var(--sky);margin:14px 0 4px;}
.scene-row{display:flex;align-items:center;gap:10px;padding:5px 8px;
           border-radius:4px;border:1px solid var(--bdr);margin-bottom:4px;background:var(--surf2);}
.scene-date{font-family:'IBM Plex Mono',monospace;font-size:.72rem;color:var(--txt);width:90px;}
.scene-cloud{font-family:'IBM Plex Mono',monospace;font-size:.7rem;width:60px;}
.stButton>button{background:var(--sky);color:#060a10;border:none;border-radius:5px;
  font-family:'IBM Plex Mono',monospace;font-weight:600;font-size:.78rem;
  padding:10px 18px;width:100%;letter-spacing:.03em;transition:opacity .15s;}
.stButton>button:hover{opacity:.85;}
div[data-testid="stNumberInput"] label,div[data-testid="stTextInput"] label,
div[data-testid="stSelectbox"] label,div[data-testid="stCheckbox"] label,
div[data-testid="stSlider"] label{font-size:.66rem!important;text-transform:uppercase;
  letter-spacing:.08em;color:var(--mut)!important;font-family:'IBM Plex Mono',monospace!important;}
.stTabs [data-baseweb="tab-list"]{gap:4px;background:var(--surf);padding:4px;border-radius:6px;}
.stTabs [data-baseweb="tab"]{font-family:'IBM Plex Mono',monospace;font-size:.72rem;
  background:transparent;color:var(--mut);border:none;padding:6px 14px;border-radius:4px;}
.stTabs [aria-selected="true"]{background:var(--bdr)!important;color:var(--sky)!important;}
</style>""", unsafe_allow_html=True)

# ── core import ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _core():
    from sentinel_core import (AOI, search_scenes, best_scene, extract_scene,
        get_sas_token, compute_change_stats, change_map, query_ollama,
        true_color_rgb, false_color_rgb, swir_composite, index_colormap,
        scene_list_summary)
    return locals()

PRESETS = {
    "Custom":                     None,
    "Catonsville, MD":            (-76.755,39.240,-76.680,39.290),
    "Baltimore, MD":              (-76.720,39.250,-76.550,39.380),
    "Washington DC":              (-77.120,38.800,-76.910,38.990),
    "New York City":              (-74.050,40.680,-73.920,40.800),
    "Chesapeake Bay":             (-76.500,38.700,-76.200,39.000),
    "Atlanta, GA":                (-84.500,33.650,-84.300,33.850),
    "Houston, TX":                (-95.500,29.650,-95.200,29.850),
    "Skaftafellsjökull, Iceland": (-17.050,63.980,-16.850,64.080),
    "Strait of Hormuz":           ( 56.050,26.200, 57.050,26.800),
}

IDX_META = {
    "NDVI": ("RdYlGn","Vegetation vigour","#4ade80"),
    "EVI":  ("YlGn",  "Enhanced vegetation","#86efac"),
    "SAVI": ("YlGn",  "Soil-adjusted veg","#6ee7b7"),
    "NDWI": ("Blues_r","Open water (McFeeters)","#38bdf8"),
    "MNDWI":("Blues_r","Modified water (urban)","#7dd3fc"),
    "NDBI": ("RdPu",  "Built-up/impervious","#f87171"),
    "BSI":  ("YlOrBr","Bare soil index","#fbbf24"),
    "NDSI": ("PuBu",  "Snow & ice","#bae6fd"),
    "NBR":  ("RdYlGn","Normalized Burn Ratio","#a78bfa"),
    "BAI":  ("hot",   "Burned Area Index","#fb923c"),
}

# ── header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hd">
  <div class="hd-main">🌍 Sentinel-2 Explorer</div>
  <div class="hd-sub">Microsoft Planetary Computer · S2 L2A · 10 spectral indices · Ollama Cloud · No GEE</div>
</div>
<div style="margin-bottom:16px">
  <span class="badge bb">S2 L2A</span>&nbsp;
  <span class="badge bg">NDVI·NDWI·NDBI·EVI·SAVI·MNDWI·BSI·NDSI·NBR·BAI</span>&nbsp;
  <span class="badge ba">PC STAC</span>&nbsp;
  <span class="badge bp">Ollama Cloud</span>
</div>""", unsafe_allow_html=True)

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sl">Area of Interest</div>', unsafe_allow_html=True)
    preset = st.selectbox("Preset", list(PRESETS.keys()), index=1)
    if preset == "Custom":
        c1,c2 = st.columns(2)
        lon_min=c1.number_input("Lon min",value=-76.755,format="%.4f")
        lat_min=c2.number_input("Lat min",value=39.240, format="%.4f")
        lon_max=c1.number_input("Lon max",value=-76.680,format="%.4f")
        lat_max=c2.number_input("Lat max",value=39.290, format="%.4f")
        aoi_name=st.text_input("Name","Custom AOI")
    else:
        lon_min,lat_min,lon_max,lat_max = PRESETS[preset]
        aoi_name = preset

    st.markdown('<div class="sl">Primary Date Range</div>', unsafe_allow_html=True)
    import datetime as _dt
    d1s = st.date_input("Start", value=_dt.date(2023,6,1))
    d1e = st.date_input("End",   value=_dt.date(2023,9,1))
    date_range1 = f"{d1s}/{d1e}"
    max_cloud = st.slider("Max cloud %", 0, 60, 20)

    st.markdown('<div class="sl">Change Detection</div>', unsafe_allow_html=True)
    enable_change = st.checkbox("Compare a second period")
    date_range2 = None
    if enable_change:
        d2s = st.date_input("Compare start", value=_dt.date(2019,6,1))
        d2e = st.date_input("Compare end",   value=_dt.date(2019,9,1))
        date_range2 = f"{d2s}/{d2e}"
        chg_index = st.selectbox("Change index map", list(IDX_META.keys()), index=0)

    st.markdown('<div class="sl">Ollama</div>', unsafe_allow_html=True)
    _sk = ""
    try: _sk = st.secrets.get("OLLAMA_API_KEY","")
    except Exception: pass
    use_cloud = st.checkbox("Ollama Cloud", value=bool(_sk))
    if use_cloud:
        ollama_host    = st.text_input("Host",    value="https://ollama.com")
        ollama_api_key = st.text_input("API key", value=_sk, type="password")
    else:
        ollama_host    = st.text_input("Host", value="http://localhost:11434")
        ollama_api_key = ""
    ollama_model = st.text_input("Model", value="gpt-oss:20b-cloud",
                                  help="Cloud: gpt-oss:20b-cloud  Local: llama3.2")
    task_map = {"Landscape interpretation":"interpret",
                "Change detection":"change","Technical report":"report"}
    task_label = st.selectbox("Task", list(task_map.keys()),
                               index=1 if enable_change else 0)
    task = task_map[task_label]
    skip_llm = st.checkbox("Skip Ollama")
    st.markdown("---")
    run_btn = st.button("▶  Analyze", use_container_width=True)
    st.markdown('<p style="font-size:.58rem;color:#1e3a5f">Data: Microsoft Planetary Computer<br>sentinel-2-l2a · Public read · No auth</p>',
                unsafe_allow_html=True)

# ── idle ──────────────────────────────────────────────────────────────────────
if not run_btn:
    st.markdown("""
    <div style="background:#0d1420;border:1px dashed #162032;border-radius:10px;
                padding:52px 32px;text-align:center;margin-top:24px">
      <div style="font-size:3rem;margin-bottom:14px">🛰️</div>
      <div style="font-family:'IBM Plex Mono',monospace;font-size:1rem;color:#f1f5f9;margin-bottom:10px">
        Configure your AOI and hit Analyze
      </div>
      <div style="font-size:.82rem;color:#475569;max-width:520px;margin:0 auto;line-height:1.7">
        Pulls Sentinel-2 L2A from Microsoft Planetary Computer via STAC,
        computes 10 spectral indices, renders composites, and sends structured
        statistics to Ollama Cloud for expert interpretation.
      </div>
      <div style="margin-top:22px;display:flex;gap:8px;justify-content:center;flex-wrap:wrap">
        <span class="badge bb">True color · False color · SWIR</span>
        <span class="badge bg">10 spectral indices</span>
        <span class="badge ba">Pixel distributions</span>
        <span class="badge bp">Per-pixel change maps</span>
        <span class="badge br">LLM interpretation</span>
      </div>
    </div>""", unsafe_allow_html=True)
    st.stop()

# ── run pipeline ──────────────────────────────────────────────────────────────
core = _core()
AOI  = core["AOI"]
aoi  = AOI(lon_min,lat_min,lon_max,lat_max,aoi_name)

status   = st.empty()
progress = st.progress(0)
def step(msg,pct):
    status.markdown(f'<div class="mono" style="color:#38bdf8">⟳ {msg}</div>',
                    unsafe_allow_html=True)
    progress.progress(pct)

try:
    step("Fetching SAS token…",5)
    token = core["get_sas_token"]()

    step(f"Searching scenes: {date_range1} · cloud<{max_cloud}%…",15)
    items1 = core["search_scenes"](aoi,date_range1,max_cloud=max_cloud)
    if not items1:
        progress.empty(); status.empty()
        st.error(f"No scenes found. Try wider dates or higher cloud %.")
        st.stop()

    item1 = core["best_scene"](items1)
    sc1_date  = item1["properties"]["datetime"][:10]
    sc1_cloud = item1["properties"].get("eo:cloud_cover",0)

    step(f"Reading bands: {sc1_date} ({sc1_cloud:.1f}% cloud)…",30)
    scene1 = core["extract_scene"](item1,aoi,token)
    if scene1 is None:
        progress.empty(); status.empty()
        st.error("Band extraction failed — try different dates or larger AOI.")
        st.stop()

    scene2=None; change_stats=None; cmap_delta=None
    if enable_change and date_range2:
        step(f"Searching comparison scenes: {date_range2}…",48)
        items2 = core["search_scenes"](aoi,date_range2,max_cloud=max_cloud)
        if items2:
            step(f"Reading comparison bands…",58)
            scene2 = core["extract_scene"](core["best_scene"](items2),aoi,token)
            if scene2:
                change_stats = core["compute_change_stats"](scene1,scene2)
                cmap_delta   = core["change_map"](scene1,scene2,chg_index if enable_change else "NDVI")

    step("Building composites…",72)
    rgb_tc   = core["true_color_rgb"](scene1.bands)
    rgb_fc   = core["false_color_rgb"](scene1.bands)
    rgb_swir = core["swir_composite"](scene1.bands)

    llm_response=""
    if not skip_llm:
        step(f"Querying {ollama_model}…",88)
        llm_response = core["query_ollama"](
            scene1.stats, change_stats=change_stats,
            model=ollama_model, host=ollama_host,
            api_key=ollama_api_key or None,
            task="change" if change_stats else task)

    progress.progress(100); status.empty(); progress.empty()

except Exception as e:
    progress.empty(); status.empty()
    st.error(f"Pipeline error: {e}")
    st.exception(e); st.stop()

# ── results header ────────────────────────────────────────────────────────────
cb = "bg" if sc1_cloud<10 else "ba" if sc1_cloud<25 else "br"
ctx = scene1.aoi.context()
st.markdown(f"""
<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;flex-wrap:wrap">
  <span style="font-family:'IBM Plex Mono',monospace;font-size:1.05rem;font-weight:600">{aoi_name}</span>
  <span class="badge bb">{sc1_date}</span>
  <span class="badge {cb}">{sc1_cloud:.1f}% cloud</span>
  <span class="badge ba">{len(scene1.bands)} bands</span>
  <span class="badge bg">{scene1.stats.get('coverage_pct',0)}% valid</span>
  <span class="badge bp">{scene1.aoi.area_km2()} km²</span>
</div>""", unsafe_allow_html=True)
if ctx:
    st.markdown(f'<div style="font-size:.75rem;color:#475569;margin-bottom:12px;font-style:italic">{ctx}</div>',
                unsafe_allow_html=True)

# ── key metrics row ───────────────────────────────────────────────────────────
idx_stats = scene1.stats.get("indices",{})
lc        = scene1.stats.get("land_cover",{})

def mc(label,val,sub,color="var(--sky)"):
    return f"""<div class="mc" style="border-top-color:{color}">
    <div class="ml">{label}</div>
    <div class="mv" style="color:{color}">{val}</div>
    <div class="ms">{sub}</div></div>"""

cols = st.columns(6)
metrics = [
    ("NDVI",      f"{idx_stats.get('NDVI',{}).get('mean','—')}",  "vegetation vigour","#4ade80"),
    ("NDWI",      f"{idx_stats.get('NDWI',{}).get('mean','—')}",  "open water","#38bdf8"),
    ("NDBI",      f"{idx_stats.get('NDBI',{}).get('mean','—')}",  "built-up","#f87171"),
    ("NDSI",      f"{scene1.stats.get('snow_ice_pct','—')}%",     "snow/ice px","#bae6fd"),
    ("Dense veg", f"{lc.get('dense_veg_pct','—')}%",             "NDVI>0.5","#86efac"),
    ("Water",     f"{scene1.stats.get('water_pct','—')}%",        "NDWI>0","#7dd3fc"),
]
for col,(label,val,sub,color) in zip(cols,metrics):
    col.markdown(mc(label,val,sub,color), unsafe_allow_html=True)

# ── tabs ──────────────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import matplotlib.cm as mcm
import matplotlib.colors as mcolors

tab_labels = ["🖼 Imagery","📊 Indices","📈 Distributions","🔀 Change","🤖 Analysis","📋 Export"]
if not enable_change or not scene2:
    tab_labels.pop(3)

tabs = st.tabs(tab_labels)
tab_idx = 0

# ── TAB: Imagery ─────────────────────────────────────────────────────────────
tab_idx += 1
with tabs[tab_idx - 1]:
    composites = [
        (rgb_tc,   f"True Color (B4/B3/B2) — {sc1_date}"),
        (rgb_fc,   f"False Color NIR (B8/B4/B3) — {sc1_date}"),
        (rgb_swir, f"SWIR Composite (B11/B8/B4) — {sc1_date}"),
    ]
    if scene2:
        rgb_tc2 = core["true_color_rgb"](scene2.bands)
        composites.append((rgb_tc2, f"True Color — {scene2.date}"))

    n_comp = sum(1 for img,_ in composites if img is not None)
    img_cols = st.columns(min(n_comp,3))
    ci = 0
    for img,caption in composites:
        if img is None: continue
        with img_cols[ci % 3]:
            st.caption(caption)
            st.image(img, use_container_width=True)
        ci += 1

    # Scene catalog
    catalog = core["scene_list_summary"](items1)
    if catalog:
        st.markdown('<div class="sl" style="margin-top:20px">Available Scenes This Period</div>',
                    unsafe_allow_html=True)
        rows = ""
        for s in catalog:
            cloud_color = "#4ade80" if s["cloud"]<10 else "#fbbf24" if s["cloud"]<25 else "#f87171"
            star = " ★" if s["date"]==sc1_date else ""
            rows += f"""<div class="scene-row">
              <div class="scene-date">{s['date']}{star}</div>
              <div class="scene-cloud" style="color:{cloud_color}">{s['cloud']}% ☁</div>
              <div style="font-family:'IBM Plex Mono',monospace;font-size:.65rem;color:#334155">{s['id'][:40]}</div>
            </div>"""
        st.markdown(rows, unsafe_allow_html=True)

# ── TAB: Indices ─────────────────────────────────────────────────────────────
tab_idx += 1
with tabs[tab_idx - 1]:
    available = [k for k in IDX_META if k in scene1.indices]
    n = len(available)
    if n:
        ncols_idx = min(n, 4)
        nrows_idx = (n + ncols_idx - 1) // ncols_idx
        fig, axes = plt.subplots(nrows_idx, ncols_idx,
                                  figsize=(5*ncols_idx, 4.2*nrows_idx),
                                  facecolor="#060a10", constrained_layout=True)
        axes_flat = np.array(axes).flatten() if n > 1 else [axes]
        for ax, name in zip(axes_flat, available):
            cmap_name, desc, _ = IDX_META[name]
            arr    = scene1.indices[name]
            masked = np.where(scene1.mask, arr, np.nan)
            vmin, vmax = (-1,1) if name!="BAI" else (0,1)
            im = ax.imshow(masked, cmap=cmap_name, vmin=vmin, vmax=vmax)
            ax.set_title(f"{name}", color="#38bdf8", fontsize=10, fontfamily="monospace",fontweight="bold")
            ax.set_xlabel(desc, color="#475569", fontsize=7)
            ax.axis("off")
            plt.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
        for ax in axes_flat[n:]:
            ax.set_visible(False)
        fig.patch.set_facecolor("#060a10")
        plt.savefig("/tmp/s2_idx.png", dpi=140, bbox_inches="tight", facecolor="#060a10")
        st.image("/tmp/s2_idx.png", use_container_width=True)

    # Index stats table
    st.markdown('<div class="sl" style="margin-top:16px">Index Statistics</div>', unsafe_allow_html=True)
    rows_html = ""
    for name in available:
        s  = idx_stats.get(name, {})
        _,_,color = IDX_META[name]
        mean_val = s.get("mean", 0)
        bar_pct  = int((mean_val+1)/2*100)
        d_html   = ""
        if change_stats:
            ic  = change_stats.get("index_changes",{}).get(name,{})
            delta = ic.get("delta")
            if delta is not None:
                cls  = "bg" if delta>=0 else "br"
                sign = "+" if delta>=0 else ""
                d_html = f'<span class="badge {cls}" style="font-size:.58rem">{sign}{delta:.4f}</span>'
        rows_html += f"""<div style="display:flex;align-items:center;padding:6px 0;
          border-bottom:1px solid var(--bdr);gap:10px">
          <div style="font-family:'IBM Plex Mono',monospace;font-size:.74rem;color:{color};width:58px">{name}</div>
          <div style="flex:1;background:var(--bdr);border-radius:2px;height:6px">
            <div style="width:{bar_pct}%;height:6px;border-radius:2px;background:{color};opacity:.8"></div></div>
          <div style="font-family:'IBM Plex Mono',monospace;font-size:.72rem;width:56px;text-align:right">{mean_val:.4f}</div>
          <div style="width:80px;text-align:right">{d_html}</div>
        </div>"""
    st.markdown(rows_html, unsafe_allow_html=True)
    if change_stats:
        st.caption(f"Δ = {scene2.date} minus {sc1_date}")

# ── TAB: Distributions ───────────────────────────────────────────────────────
tab_idx += 1
with tabs[tab_idx - 1]:
    hists = scene1.stats.get("histograms",{})
    bins  = np.linspace(-1,1,40)
    avail_h = [k for k in IDX_META if k in hists]
    if avail_h:
        ncols_h = min(len(avail_h),3)
        nrows_h = (len(avail_h)+ncols_h-1)//ncols_h
        fig2, axes2 = plt.subplots(nrows_h, ncols_h,
                                    figsize=(5.5*ncols_h, 3.5*nrows_h),
                                    facecolor="#060a10", constrained_layout=True)
        axes2_flat = np.array(axes2).flatten() if len(avail_h)>1 else [axes2]
        for ax, name in zip(axes2_flat, avail_h):
            hist = hists[name]
            _,_,color = IDX_META[name]
            ax.bar(bins, hist, width=0.048, color=color, alpha=0.75, edgecolor="none")
            mean_v = idx_stats.get(name,{}).get("mean",0)
            p10    = idx_stats.get(name,{}).get("p10",0)
            p90    = idx_stats.get(name,{}).get("p90",0)
            ax.axvline(mean_v, color="white",lw=1.5,ls="--",label=f"mean {mean_v:.3f}")
            ax.axvspan(p10, p90, alpha=0.12, color=color, label="P10–P90")
            ax.set_facecolor("#0d1420")
            ax.set_title(name, color=color, fontsize=10,fontfamily="monospace",fontweight="bold")
            ax.tick_params(colors="#475569",labelsize=7)
            ax.spines[:].set_color("#162032")
            ax.legend(fontsize=6,labelcolor="white",fancybox=False,framealpha=.2)
            if scene2 and name in scene2.stats.get("histograms",{}):
                hist2 = scene2.stats["histograms"][name]
                ax.bar(bins, hist2, width=0.048, color="#ffffff", alpha=0.2, edgecolor="none", label=scene2.date)
        for ax in axes2_flat[len(avail_h):]:
            ax.set_visible(False)
        fig2.patch.set_facecolor("#060a10")
        plt.savefig("/tmp/s2_hist.png", dpi=140, bbox_inches="tight", facecolor="#060a10")
        st.image("/tmp/s2_hist.png", use_container_width=True)
        if scene2:
            st.caption(f"White bars = {scene2.date} overlay")

    # Land cover donut
    if lc:
        st.markdown('<div class="sl" style="margin-top:12px">Land Cover Proxy</div>', unsafe_allow_html=True)
        lc_cols = st.columns(4)
        lc_items = [("Dense veg",lc.get("dense_veg_pct",0),"#4ade80"),
                    ("Sparse veg",lc.get("sparse_veg_pct",0),"#86efac"),
                    ("Barren/built",lc.get("barren_pct",0),"#fbbf24"),
                    ("Water/snow",lc.get("water_snow_pct",0),"#38bdf8")]
        for col,(label,pct,color) in zip(lc_cols,lc_items):
            col.markdown(f'<div class="mc" style="border-top-color:{color}">'
                         f'<div class="ml">{label}</div>'
                         f'<div class="mv" style="color:{color};font-size:1.1rem">{pct}%</div></div>',
                         unsafe_allow_html=True)

# ── TAB: Change (conditional) ─────────────────────────────────────────────────
if enable_change and scene2 and change_stats:
    tab_idx += 1
    with tabs[tab_idx - 1]:
        ic = change_stats.get("index_changes",{})

        # Change metric cards
        ch_c = st.columns(min(len(ic),5))
        for col,(name,vals) in zip(ch_c, list(ic.items())[:5]):
            delta = vals.get("delta",0)
            color = "#4ade80" if delta>=0 else "#f87171"
            sign  = "+" if delta>=0 else ""
            dpct  = vals.get("delta_pct",0)
            col.markdown(f'<div class="mc" style="border-top-color:{color}">'
                         f'<div class="ml">{name}</div>'
                         f'<div class="mv" style="color:{color};font-size:1.1rem">{sign}{delta:.4f}</div>'
                         f'<div class="ms">{vals["mean_t1"]:.3f}→{vals["mean_t2"]:.3f} ({sign}{dpct:.1f}%)</div>'
                         f'</div>', unsafe_allow_html=True)

        # Per-pixel change map
        st.markdown(f'<div class="sl" style="margin-top:16px">Per-Pixel {chg_index} Change Map — {sc1_date} → {scene2.date}</div>',
                    unsafe_allow_html=True)
        if cmap_delta is not None:
            fig3, ax3 = plt.subplots(1, 2, figsize=(12, 4.5), facecolor="#060a10")
            # T1 vs T2 side-by-side
            rgb_tc2 = core["true_color_rgb"](scene2.bands)
            for ax,img,lbl in [(ax3[0],rgb_tc,f"T1: {sc1_date}"),(ax3[1],rgb_tc2,f"T2: {scene2.date}")]:
                if img is not None:
                    ax.imshow(img); ax.axis("off")
                    ax.set_title(lbl,color="#38bdf8",fontsize=9,fontfamily="monospace")
            plt.tight_layout(pad=.5)
            fig3.patch.set_facecolor("#060a10")
            plt.savefig("/tmp/s2_t1t2.png",dpi=130,bbox_inches="tight",facecolor="#060a10")
            st.image("/tmp/s2_t1t2.png", use_container_width=True)

            fig4, ax4 = plt.subplots(figsize=(8,4.5), facecolor="#060a10")
            lim = max(abs(np.nanpercentile(cmap_delta,2)), abs(np.nanpercentile(cmap_delta,98)), 0.01)
            im4 = ax4.imshow(cmap_delta, cmap="RdYlGn", vmin=-lim, vmax=lim)
            ax4.set_title(f"Δ{chg_index}  (green=increase  red=decrease)",
                           color="#38bdf8",fontsize=10,fontfamily="monospace")
            ax4.axis("off")
            plt.colorbar(im4, ax=ax4, label=f"Δ{chg_index}", shrink=0.8)
            fig4.patch.set_facecolor("#060a10")
            plt.savefig("/tmp/s2_chgmap.png",dpi=140,bbox_inches="tight",facecolor="#060a10")
            st.image("/tmp/s2_chgmap.png", use_container_width=True)

        # Land cover change table
        lcc = change_stats.get("land_cover_change",{})
        if lcc:
            st.markdown('<div class="sl" style="margin-top:14px">Land Cover Change</div>',
                        unsafe_allow_html=True)
            lcc_cols = st.columns(len(lcc))
            colors = {"dense_veg_pct":"#4ade80","sparse_veg_pct":"#86efac",
                      "barren_pct":"#fbbf24","water_snow_pct":"#38bdf8"}
            for col,(k,v) in zip(lcc_cols,lcc.items()):
                color = colors.get(k,"#38bdf8")
                delta = v.get("delta",0)
                sign  = "+" if delta>=0 else ""
                dcol  = "#4ade80" if delta>=0 else "#f87171"
                col.markdown(f'<div class="mc" style="border-top-color:{color}">'
                             f'<div class="ml">{k.replace("_pct","").replace("_"," ")}</div>'
                             f'<div class="mv" style="color:{color};font-size:1rem">{v["t1"]}→{v["t2"]}%</div>'
                             f'<div class="ms" style="color:{dcol}">{sign}{delta}pp</div>'
                             f'</div>', unsafe_allow_html=True)

# ── TAB: Analysis ─────────────────────────────────────────────────────────────
tab_idx += 1
with tabs[tab_idx - 1]:
    if skip_llm:
        st.info("Ollama skipped — uncheck 'Skip Ollama' in sidebar to enable.")
    elif llm_response:
        ep  = "cloud" if ollama_api_key else "local"
        ebc = "bg" if ollama_api_key else "bb"
        st.markdown(f"""<div style="display:flex;align-items:center;gap:8px;margin-bottom:10px">
          <span>🤖</span>
          <span class="mono" style="color:#475569">{ollama_model} · {task_label.lower()}</span>
          <span class="badge {ebc}">{ep}</span></div>""", unsafe_allow_html=True)
        # Render markdown headers inside llm-box
        formatted = llm_response.replace("## ","<h2>").replace("\n","<br>")
        # Actually just use st.markdown for proper rendering
        st.markdown(llm_response)
    else:
        st.warning("No response from Ollama. Check host, API key, and model name.")

# ── TAB: Export ───────────────────────────────────────────────────────────────
tab_idx += 1
with tabs[tab_idx - 1]:
    export = {"scene1_stats": {k:v for k,v in scene1.stats.items() if k!="histograms"}}
    if change_stats:
        export["change_stats"] = change_stats
    if llm_response:
        export["llm_response"] = llm_response

    st.download_button("⬇ Download JSON summary",
                       data=json.dumps(export,indent=2),
                       file_name=f"s2_{aoi_name.replace(' ','_')}_{sc1_date}.json",
                       mime="application/json")

    try:
        from PIL import Image as PILImage
        import io
        if rgb_tc is not None:
            buf = io.BytesIO()
            PILImage.fromarray(rgb_tc).save(buf,format="PNG")
            st.download_button("⬇ Download True Color PNG", data=buf.getvalue(),
                               file_name=f"s2_tc_{aoi_name.replace(' ','_')}_{sc1_date}.png",
                               mime="image/png")
    except ImportError:
        pass

    with st.expander("Full raw statistics (JSON)"):
        st.json(scene1.stats)

# ── footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div style="font-size:.62rem;color:#1e3a5f;text-align:center">'
            'Sentinel-2 L2A · Microsoft Planetary Computer · '
            'planetarycomputer.microsoft.com</div>',
            unsafe_allow_html=True)
