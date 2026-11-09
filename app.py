"""
Sentinel-1 LULC prototype.

Input : latitude, longitude, start month
Output: 12 monthly land cover maps from the Sentinel-1-only U-Net, class shares per
        month, month-to-month change and total change (percentage points).

Run:    streamlit run app.py
"""
from __future__ import annotations

import io
import os
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import streamlit as st

import config as C
import s1_data as S
from lulc_model import load_checkpoint, predict, temporal_majority, area_statistics, valid_fraction

st.set_page_config(page_title="Sentinel-1 Land Cover", page_icon="🛰️", layout="wide")

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
CMAP = ListedColormap(C.CLASS_COLORS)
ROOT = Path(__file__).parent


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Connecting to Earth Engine…")
def connect_ee(project: str):
    S.init_ee(project)
    return True


@st.cache_resource(show_spinner="Loading model…")
def get_model(path: str):
    return load_checkpoint(path, "cpu")


@st.cache_data(show_spinner=False, max_entries=256)
def fetch_month(lat, lon, size_px, year, month):
    grid = S.make_grid(lat, lon, size_px)
    return S.fetch_s1_month(grid, year, month)


@st.cache_data(show_spinner=False, max_entries=32)
def fetch_reference(lat, lon, size_px):
    return S.fetch_worldcover(S.make_grid(lat, lon, size_px))


@st.cache_data(show_spinner=False, max_entries=128, ttl=24 * 3600)
def geocode(query: str, limit: int = 5) -> list[dict]:
    """Place name -> candidate locations via OpenStreetMap Nominatim (free, no API key).
    Usage policy: identify the app in the User-Agent, at most 1 request per second, cache results."""
    import requests
    r = requests.get(
        "https://nominatim.openstreetmap.org/search",
        params={"q": query, "format": "jsonv2", "limit": limit},
        headers={"User-Agent": "s1-lulc-prototype/1.0 (student project)"},
        timeout=15,
    )
    r.raise_for_status()
    return [{"name": h["display_name"], "lat": float(h["lat"]), "lon": float(h["lon"])} for h in r.json()]


def apply_place(place: dict):
    """Put a geocoding result into the latitude/longitude inputs."""
    st.session_state["lat"] = round(place["lat"], 4)
    st.session_state["lon"] = round(place["lon"], 4)
    st.session_state["place_name"] = place["name"]
    st.session_state["place_coords"] = (st.session_state["lat"], st.session_state["lon"])


def on_choice_change():
    idx = st.session_state["geo_choice"]
    apply_place(st.session_state["geo_results"][idx])


def last_complete_month(today: date | None = None):
    today = today or date.today()
    return S.add_months(today.year, today.month, -1)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
def legend_handles():
    return [plt.Rectangle((0, 0), 1, 1, color=c) for c in C.CLASS_COLORS]


def maps_figure(maps, labels, valid):
    fig, axes = plt.subplots(3, 4, figsize=(16, 12.6))
    for ax, m, lab, v in zip(axes.ravel(), maps, labels, valid):
        ax.imshow(np.ma.masked_equal(m, C.IGNORE_INDEX), cmap=CMAP, vmin=0, vmax=C.NUM_CLASSES - 1,
                  interpolation="nearest")
        ax.set_title(lab + ("" if v > 0.95 else f"  (data {100 * v:.0f}%)"), fontsize=13)
        ax.axis("off")
    fig.legend(legend_handles(), C.CLASS_NAMES, loc="lower center", ncol=C.NUM_CLASSES, fontsize=12)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def single_map_figure(m, title):
    fig, ax = plt.subplots(figsize=(5, 5.4))
    ax.imshow(np.ma.masked_equal(m, C.IGNORE_INDEX), cmap=CMAP, vmin=0, vmax=C.NUM_CLASSES - 1,
              interpolation="nearest")
    ax.set_title(title); ax.axis("off")
    fig.legend(legend_handles(), C.CLASS_NAMES, loc="lower center", ncol=3, fontsize=8)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    return fig


def share_figure(pct):
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for name, col in zip(C.CLASS_NAMES, C.CLASS_COLORS):
        ax.plot(pct.index, pct[name], "-o", color=col, lw=2, ms=4, label=name)
    ax.set_ylabel("% of area"); ax.grid(alpha=.3)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.setp(ax.get_xticklabels(), rotation=30)
    fig.tight_layout()
    return fig


def fig_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    return buf.getvalue()


def signed(df):
    return df.style.format("{:+.2f}").background_gradient(cmap="RdBu_r", vmin=-10, vmax=10)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    ee_project = st.text_input("Earth Engine project ID", os.environ.get("EE_PROJECT", ""),
                               help="Google Cloud project registered for Earth Engine. "
                                    "Can also be set via the EE_PROJECT environment variable.")
    model_path = st.text_input("Model weights", str(ROOT / "models" / "best_model.pth"))
    size_km = st.select_slider("Area size (km)", options=[2.56, 5.12, 7.68], value=5.12)
    smooth = st.checkbox("3-month majority smoothing", value=False,
                         help="Reduces month-to-month flicker caused by seasonal radar effects "
                              "(crop growth, soil moisture). Off = raw model output.")
    show_ref = st.checkbox("Show WorldCover 2021 for comparison", value=True)

size_px = int(round(size_km * 1000 / C.PIXEL_SIZE_M))

# ---------------------------------------------------------------------------
# Main inputs
# ---------------------------------------------------------------------------
st.title("🛰️ Sentinel-1 Land Cover over 12 Months")
st.caption("Radar-only U-Net (Sentinel-1 VV + VH). Land cover classes learnt from ESA WorldCover 2021.")

ly, lm = last_complete_month()
max_start_y, max_start_m = S.add_months(ly, lm, -11)

# Defaults: central Mannheim
st.session_state.setdefault("lat", 49.4875)
st.session_state.setdefault("lon", 8.4660)
st.session_state.setdefault("place_name", "Mannheim, Baden-Württemberg, Deutschland")
st.session_state.setdefault("geo_results", [])
st.session_state.setdefault("place_coords", (49.4875, 8.4660))

# --- Search by place name (fills latitude and longitude) ---
with st.form("place_search", clear_on_submit=False, border=False):
    s1, s2 = st.columns([5, 1])
    query = s1.text_input("Search a place", placeholder="e.g. Heidelberg, Lake Constance, Paris La Défense",
                          label_visibility="collapsed", key="place_query")
    searched = s2.form_submit_button("🔍 Search")
if searched and query.strip():
    try:
        results = geocode(query.strip())
    except Exception as e:
        results = None
        st.error(f"Place search failed ({e.__class__.__name__}). Enter latitude and longitude instead.")
    if results == []:
        st.warning(f"No place found for “{query.strip()}”. Try adding the country, e.g. “Mannheim, Germany”.")
    elif results:
        st.session_state["geo_results"] = results
        st.session_state["geo_choice"] = 0
        apply_place(results[0])
geo_results = st.session_state["geo_results"]
if len(geo_results) > 1:
    st.selectbox("Did you mean", range(len(geo_results)), key="geo_choice",
                 format_func=lambda i: geo_results[i]["name"], on_change=on_choice_change)

c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
lat = c1.number_input("Latitude", -60.0, 75.0, step=0.01, format="%.4f", key="lat")
lon = c2.number_input("Longitude", -180.0, 180.0, step=0.01, format="%.4f", key="lon")
years = list(range(2015, max_start_y + 1))
start_year = c3.selectbox("Start year", years, index=years.index(min(2023, max_start_y)))
start_month = c4.selectbox("Start month", list(range(1, 13)), format_func=lambda m: MONTH_NAMES[m - 1])

months = S.month_range(start_year, start_month, 12)
month_labels = [f"{MONTH_NAMES[m - 1]} {y}" for y, m in months]
end_y, end_m = months[-1]
too_late = (end_y, end_m) > (ly, lm)
st.write(f"Period: **{month_labels[0]} → {month_labels[-1]}**, area {size_km} km × {size_km} km "
         f"({size_px} × {size_px} pixels at 10 m)")
if too_late:
    st.error(f"The 12-month window must end by {MONTH_NAMES[lm - 1]} {ly}. "
             f"Latest possible start: {MONTH_NAMES[max_start_m - 1]} {max_start_y}.")

# Show the place name only while the coordinates still match the searched place
place = st.session_state["place_name"] if st.session_state["place_coords"] == (lat, lon) else None
if place:
    st.caption(f"📍 {place}")

grid = S.make_grid(lat, lon, size_px)
with st.expander("Location preview", expanded=False):
    try:
        import folium
        from streamlit_folium import st_folium
        s, w, n, e = grid.latlon_bounds()
        fmap = folium.Map(location=[lat, lon], zoom_start=12, tiles="OpenStreetMap")
        folium.Rectangle([[s, w], [n, e]], color="red", weight=2, fill=False).add_to(fmap)
        st_folium(fmap, height=350, width=None, returned_objects=[])
    except ImportError:
        st.info("Install folium and streamlit-folium for a map preview.")

run = st.button("Run analysis", type="primary", disabled=too_late)

# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------
if run:
    if not ee_project:
        st.error("Enter your Earth Engine project ID in the sidebar."); st.stop()
    if not Path(model_path).exists():
        st.error(f"Model weights not found at `{model_path}`. Train with notebooks/02_train.ipynb "
                 "or download best_model.pth into models/ (see README)."); st.stop()
    connect_ee(ee_project)
    model, ckpt = get_model(model_path)

    progress = st.progress(0.0, text="Starting…")
    s1_stack, scenes, maps = [], [], []
    for i, ((y, m), lab) in enumerate(zip(months, month_labels)):
        progress.progress(i / 12, text=f"Downloading Sentinel-1 for {lab}…")
        arr, n = fetch_month(lat, lon, size_px, y, m)
        progress.progress((i + 0.5) / 12, text=f"Classifying {lab}…")
        maps.append(predict(model, arr, ckpt["mean"], ckpt["std"]))
        s1_stack.append(arr); scenes.append(n)
    progress.progress(1.0, text="Done")
    progress.empty()

    ref = fetch_reference(lat, lon, size_px) if show_ref else None
    st.session_state["result"] = dict(
        maps=np.stack(maps), s1=np.stack(s1_stack), scenes=scenes, labels=month_labels,
        ref=ref, lat=lat, lon=lon, size_km=size_km, place=place,
    )

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
res = st.session_state.get("result")
if res is None:
    st.info("Choose a location and start month, then press **Run analysis**.")
    st.stop()

maps = temporal_majority(res["maps"], 3) if smooth else res["maps"]
labels = res["labels"]
valid = valid_fraction(maps)
pct, delta, total = area_statistics(maps, labels)

bad = [f"{l} ({n} scenes, {100 * v:.0f}% coverage)" for l, n, v in zip(labels, res["scenes"], valid) if v < 0.9]
if bad:
    st.warning("Incomplete radar coverage, percentages use valid pixels only: " + "; ".join(bad))

where = f"{res['lat']:.4f}, {res['lon']:.4f}"
if res.get("place"):
    where = f"{res['place'].split(',')[0]} ({where})"
st.subheader(f"Results for {where}" + ("  ·  smoothed" if smooth else ""))

k1, k2, k3 = st.columns(3)
biggest = total["Change (pp)"].abs().idxmax()
k1.metric("Dominant class (first month)", pct.iloc[0].idxmax(), f"{pct.iloc[0].max():.1f}%", delta_color="off")
k2.metric("Dominant class (last month)", pct.iloc[-1].idxmax(), f"{pct.iloc[-1].max():.1f}%", delta_color="off")
k3.metric("Largest total change", biggest, f"{total.loc[biggest, 'Change (pp)']:+.2f} pp")

tab_maps, tab_stats, tab_radar = st.tabs(["🗺️ Monthly maps", "📊 Statistics", "📡 Radar input"])

with tab_maps:
    fig = maps_figure(maps, labels, valid)
    st.pyplot(fig)
    st.download_button("Download maps (PNG)", fig_png(fig), "lulc_maps.png", "image/png")
    if res["ref"] is not None:
        c1, c2 = st.columns(2)
        c1.pyplot(single_map_figure(res["ref"], "ESA WorldCover 2021 (reference)"))
        c2.pyplot(single_map_figure(maps[-1], f"Model · {labels[-1]}"))

with tab_stats:
    st.markdown("#### Share of area per class (%)")
    st.dataframe(pct.style.format("{:.2f}"))
    st.pyplot(share_figure(pct))

    st.markdown("#### Month-to-month change (percentage points)")
    st.dataframe(signed(delta))

    st.markdown(f"#### Total change {labels[0]} → {labels[-1]}")
    st.dataframe(total.style.format({total.columns[0]: "{:.2f}", total.columns[1]: "{:.2f}",
                                     "Change (pp)": "{:+.2f}"}))

    csv = pd.concat({"share_pct": pct, "mom_change_pp": delta}, names=["table", "period"])
    st.download_button("Download statistics (CSV)",
                       csv.to_csv().encode() + b"\n" + total.to_csv().encode(),
                       "lulc_statistics.csv", "text/csv")
    st.caption("Percentage points (pp) = difference between two percentages. "
               "Note: part of the month-to-month change in a radar-only model reflects seasonal "
               "backscatter changes (vegetation growth, soil moisture, harvest), not real land "
               "cover change. Enable smoothing in the sidebar to reduce it.")

with tab_radar:
    st.caption("Sentinel-1 monthly median VV backscatter (dB). Bright = rough or built surfaces, dark = water or smooth surfaces.")
    fig, axes = plt.subplots(2, 6, figsize=(18, 6.5))
    for ax, a, lab, n in zip(axes.ravel(), res["s1"], labels, res["scenes"]):
        ax.imshow(a[0], cmap="gray", vmin=-25, vmax=0)
        ax.set_title(f"{lab}\n{n} scenes", fontsize=10); ax.axis("off")
    fig.tight_layout()
    st.pyplot(fig)