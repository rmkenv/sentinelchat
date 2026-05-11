# 🌍 Sentinel-2 Explorer

**Sentinel-2 L2A · Microsoft Planetary Computer · Ollama Cloud · No GEE**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/sentinel-explorer/blob/main/sentinel_colab.ipynb)

Pulls Sentinel-2 surface reflectance imagery directly from Microsoft Planetary Computer STAC, computes six spectral indices, and interprets results via Ollama Cloud. No Google Earth Engine. No authentication required for data access.

---

## Indices Computed

| Index | Formula | Measures |
|-------|---------|---------|
| NDVI  | (NIR−Red)/(NIR+Red) | Vegetation vigour |
| NDWI  | (Green−NIR)/(Green+NIR) | Open water |
| NDBI  | (SWIR1−NIR)/(SWIR1+NIR) | Built-up / impervious |
| EVI   | 2.5×(NIR−Red)/(NIR+6×Red−7.5×Blue+1) | Enhanced vegetation |
| SAVI  | 1.5×(NIR−Red)/(NIR+Red+0.5) | Soil-adjusted vegetation |
| MNDWI | (Green−SWIR1)/(Green+SWIR1) | Modified water index |

---

## Quickstart

```bash
git clone https://github.com/YOUR_USERNAME/sentinel-explorer
cd sentinel-explorer

# System deps (macOS)
brew install gdal proj

# System deps (Ubuntu)
sudo apt-get install libgdal-dev gdal-bin libproj-dev

pip install -r requirements.txt

# Ollama (optional — for LLM interpretation)
ollama pull llama3.2 && ollama serve

streamlit run app.py
```

## Streamlit Community Cloud

1. Push to GitHub
2. [share.streamlit.io](https://share.streamlit.io) → New app → `app.py`
3. Secrets dashboard → add `OLLAMA_API_KEY = "your-key"`
4. Deploy — `packages.txt` handles GDAL automatically

## Files

```
app.py               Streamlit UI
sentinel_core.py     Pipeline: STAC search, band read, indices, Ollama
requirements.txt     Python deps
packages.txt         System deps for Streamlit Cloud
.streamlit/
  config.toml        Dark theme
  secrets.toml.example  API key template
```

## Data

- **Source**: Microsoft Planetary Computer
- **Collection**: `sentinel-2-l2a` (Sentinel-2 Level-2A surface reflectance)
- **Access**: Public read via SAS token (auto-fetched, no account needed)
- **Resolution**: 10m (B02/B03/B04/B08), 20m (B11/B12/SCL)
- **Coverage**: Global, 2017–present
