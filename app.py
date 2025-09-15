from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
import plotly.express as px
import plotly.graph_objects as go
import geopandas as gpd  # required for land cover reader

from data_config import (
    YEAR_MIN, YEAR_MAX, RES, CENTER, LANDCOVER_DIR, GIT_BASE_URL
)
from data_loaders import (
    get_cells_df, hex_outline_gdf, h3_polygon_lonlat, discover_landcover, to_git_url,
    load_biodiv_metrics, load_birds_for_hex, load_invasive_sensitive,
    load_land_use, load_environmental_risks, load_aqueduct_center,
    load_eco_for_hex
)

st.set_page_config(page_title="HelMos Biodiversity", page_icon="✨", layout="wide")

# ---------- Sidebar ----------
cells_df = get_cells_df()
cell_ids = cells_df["h3_index"].tolist()
pos_map = dict(zip(cells_df["h3_index"], cells_df["position"]))

with st.sidebar:
    st.title("✨ HelMos Biodiversity")
    view = st.radio(
        "View",
        ["Overview", "Biodiversity", "Ecosystem State", "Pressures", "Risks", "Land Cover"],
        index=0
    )
    year_range = st.slider("Time range", YEAR_MIN, YEAR_MAX, (2012, 2022), step=1)
    ymin, ymax = year_range

    hex_choice = st.selectbox(
        "Grid (for per-hex tables)",
        [f"{pos_map[h]} | {h}" for h in cell_ids],
        index=0
    )
    sel_h3 = hex_choice.split("|")[1].strip()
    sel_pos = pos_map[sel_h3]

    if view == "Land Cover":
        lc_year = st.number_input("Land-cover year", value=int(ymax), min_value=YEAR_MIN, max_value=YEAR_MAX, step=1)
        lc_opacity = st.slider("Land-cover opacity", 0, 100, 70, 5)

# ---------- H3 base map ----------
HEX_OUTLINES = hex_outline_gdf(cell_ids)
bounds = HEX_OUTLINES.total_bounds  # minx, miny, maxx, maxy
center_lat = (bounds[1] + bounds[3]) / 2
center_lon = (bounds[0] + bounds[2]) / 2

def h3_outline_layer():
    data = []
    for _, row in HEX_OUTLINES.iterrows():
        coords = list(row.geometry.exterior.coords)
        data.append({"polygon": [[x, y] for (x, y) in coords], "label": row["h3_index"]})
    return pdk.Layer(
        "PolygonLayer", data=data, get_polygon="polygon",
        get_fill_color=[0,0,0,0], get_line_color=[0,0,0,180], line_width_min_pixels=1.5,
        stroked=True, filled=False, pickable=True
    )

def map_view_state(zoom=9):
    return pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=zoom)

# ----------------- LAND COVER VIEW -----------------
def show_land_cover():
    st.subheader(f"🗺️ Land Cover — {lc_year}")
    df = discover_landcover()
    if df.empty:
        st.info(f"No GeoJSON files found in {LANDCOVER_DIR}")
        return

    subset = df[df["year"] == int(lc_year)]
    if subset.empty:
        st.info(f"No land-cover files for year {lc_year} in {LANDCOVER_DIR}")
        return

    layers = [h3_outline_layer()]
    alpha = int(round(255 * (lc_opacity/100.0)))

    for _, r in subset.iterrows():
        gdf = gpd.read_file(r["file"])
        if gdf.empty or gdf.geometry.is_empty.all():
            continue
        if "class" not in gdf.columns and "Class" in gdf.columns:
            gdf.rename(columns={"Class":"class"}, inplace=True)

        # Color per class (placeholder distinct colors; swap to GLC/DW palettes if desired)
        def color_for_row(row):
            cls = row.get("class", 0)
            seed = int(cls) if pd.notna(cls) else 0
            rng = np.random.default_rng(seed)
            rC, gC, bC = rng.integers(50, 221, size=3).tolist()
            return [int(rC), int(gC), int(bC), alpha]

        recs = []
        gg = gdf.explode(index_parts=False, ignore_index=True)
        for _, gr in gg.iterrows():
            geom = gr.geometry
            if geom is None or geom.is_empty or geom.geom_type != "Polygon":
                continue
            coords = list(geom.exterior.coords)
            recs.append({
                "polygon": [[x, y] for (x, y) in coords],
                "fill_color": color_for_row(gr),
                "stroke_color": [30,30,30,120],
                "label": f"{r['position']} | {r['tag']} | class {gr.get('class')} | {r['year']}"
            })

        if recs:
            layers.append(pdk.Layer(
                "PolygonLayer", data=recs, get_polygon="polygon",
                get_fill_color="fill_color", get_line_color="stroke_color",
                line_width_min_pixels=0.5, stroked=True, filled=True, pickable=True
            ))

    deck = pdk.Deck(layers=layers, initial_view_state=map_view_state(), map_style="light", tooltip={"text":"{label}"})
    st.pydeck_chart(deck)

    with st.expander("Source files for this year"):
        for _, r in subset.iterrows():
            url = to_git_url(Path(r["file"]))
            st.write(f"{Path(r['file']).name}" + (f" — [open]({url})" if url else ""))

# ----------------- BIODIVERSITY VIEW -----------------
def show_biodiversity():
    st.subheader("Biodiversity")

    met = load_biodiv_metrics()
    if met.empty:
        st.info("No Biodiversity_Metrics_By_H3.csv found.")
    else:
        st.plotly_chart(px.bar(met, x="position", y=["alpha","msa","shannon","pielou"]), use_container_width=True)

    birds, fpath = load_birds_for_hex(sel_h3, sel_pos)
    st.markdown(f"**Birds in {sel_pos}** " + (f" — [{Path(fpath).name}]({to_git_url(fpath)})" if fpath and to_git_url(fpath) else ""))
    st.dataframe(birds, use_container_width=True, hide_index=True)

    inv, sens, sfile = load_invasive_sensitive(sel_h3, sel_pos)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Invasive species** " + (f" — [{Path(sfile).name}]({to_git_url(sfile)})" if sfile and to_git_url(sfile) else ""))
        st.dataframe(inv.head(200), use_container_width=True, hide_index=True)
    with c2:
        st.markdown("**Sensitive species (VU/EN/CR)**")
        st.dataframe(sens.head(200), use_container_width=True, hide_index=True)

# ----------------- ECOSYSTEM STATE VIEW -----------------
def show_eco():
    st.subheader("Ecosystem State")

    env, fpath = load_environmental_risks(sel_pos)
    st.markdown("**Protected ecosystems / environmental risks** " + (f"— [{Path(fpath).name}]({to_git_url(fpath)})" if fpath and to_git_url(fpath) else ""))
    if env.empty:
        st.info("No environmental_risks file for this position.")
    else:
        st.dataframe(env, use_container_width=True, hide_index=True)

    eco = load_eco_for_hex(sel_h3, sel_pos)
    ffi_df, ffi_path = eco["avg_ffi"]
    if ffi_df is not None and not ffi_df.empty:
        ycol = "year" if "year" in ffi_df.columns else ffi_df.columns[0]
        vcol = "avg_ffi" if "avg_ffi" in ffi_df.columns else ( "fragmentation" if "fragmentation" in ffi_df.columns else ffi_df.columns[1] )
        if np.issubdtype(ffi_df[ycol].dtype, np.number):
            mask = (ffi_df[ycol] >= ymin) & (ffi_df[ycol] <= ymax)
            plot_df = ffi_df.loc[mask]
        else:
            plot_df = ffi_df
        st.markdown("**Fragmentation (avg FFI)** " + (f"— [{Path(ffi_path).name}]({to_git_url(ffi_path)})" if to_git_url(ffi_path) else ""))
        st.plotly_chart(px.line(plot_df, x=ycol, y=vcol), use_container_width=True)

    row = st.columns(3)
    hi_df, hi_path = eco["high_integrity"]
    with row[0]:
        st.markdown("**High integrity (2022)** " + (f"— [{Path(hi_path).name}]({to_git_url(hi_path)})" if to_git_url(hi_path) else ""))
        st.dataframe((hi_df.head(200) if hi_df is not None else pd.DataFrame()), use_container_width=True, hide_index=True)
    cor_df, cor_path = eco["corridors"]
    with row[1]:
        st.markdown("**Corridors (2022)** " + (f"— [{Path(cor_path).name}]({to_git_url(cor_path)})" if to_git_url(cor_path) else ""))
        st.dataframe((cor_df.head(200) if cor_df is not None else pd.DataFrame()), use_container_width=True, hide_index=True)
    rd_df, rd_path = eco["rapid_decline"]
    with row[2]:
        st.markdown("**Rapid decline (2005–2022)** " + (f"— [{Path(rd_path).name}]({to_git_url(rd_path)})" if to_git_url(rd_path) else ""))
        st.dataframe((rd_df.head(200) if rd_df is not None else pd.DataFrame()), use_container_width=True, hide_index=True)

# ----------------- PRESSURES VIEW -----------------
def show_pressures():
    st.subheader("Pressures")

    lu, lup = load_land_use(sel_h3, sel_pos)
    st.markdown("**Activities (impact)** " + (f"— [{Path(lup).name}]({to_git_url(lup)})" if lup and to_git_url(lup) else ""))
    if lu.empty:
        st.info("No land-use file for this hex.")
    else:
        st.dataframe(lu, use_container_width=True, hide_index=True)

    st.caption("Climate drivers: plug your environmental time-series here if desired.")

# ----------------- RISKS VIEW -----------------
def show_risks():
    st.subheader("Risks")

    center_h3 = cells_df.loc[cells_df["position"]=="CENTER","h3_index"].iloc[0]
    aq, aqp = load_aqueduct_center(center_h3)
    if aq.empty:
        st.info("No Aqueduct V4 CSV for the central grid.")
    else:
        st.markdown("**Water risks (Aqueduct V4)** " + (f"— [{Path(aqp).name}]({to_git_url(aqp)})" if to_git_url(aqp) else ""))
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=aq["score"], theta=aq["dimension"], fill="toself",
            text=aq["label"], hovertemplate="%{theta}<br>%{text}<extra></extra>"
        ))
        fig.update_layout(margin=dict(l=20,r=20,t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)

    inv, sens, sfile = load_invasive_sensitive(sel_h3, sel_pos)
    st.markdown("**Sensitive species (VU/EN/CR)** " + (f"— [{Path(sfile).name}]({to_git_url(sfile)})" if sfile and to_git_url(sfile) else ""))
    st.dataframe(sens.head(200), use_container_width=True, hide_index=True)

# ----------------- OVERVIEW VIEW -----------------
def show_overview():
    st.subheader("Overview")
    met = load_biodiv_metrics()
    if met.empty:
        st.info("No Biodiversity_Metrics_By_H3.csv found.")
        return
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Avg. Alpha (richness)", round(met["alpha"].mean(), 2))
    with c2: st.metric("Avg. MSA", f"{met['msa'].mean():.2f}")
    with c3: st.metric("Gamma richness", int(met["gamma"].iloc[0]) if "gamma" in met.columns else "—")
    with c4: st.metric("β (Whittaker)", f"{met['beta_whittaker'].iloc[0]:.2f}" if "beta_whittaker" in met.columns else "—")
    st.plotly_chart(px.bar(met, x="position", y=["alpha","msa","shannon","pielou"]), use_container_width=True)

# ---------- Router ----------
if view == "Land Cover":
    show_land_cover()
elif view == "Biodiversity":
    show_biodiversity()
elif view == "Ecosystem State":
    show_eco()
elif view == "Pressures":
    show_pressures()
elif view == "Risks":
    show_risks()
else:
    show_overview()
