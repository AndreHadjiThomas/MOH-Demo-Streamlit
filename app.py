from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
import plotly.express as px
import plotly.graph_objects as go
import geopandas as gpd  # needed for land cover

from data_config import (
    YEAR_MIN, YEAR_MAX, GIT_BASE_URL
)
from data_loaders import (
    get_cells_df, hex_outline_gdf, discover_landcover, to_git_url,
    load_biodiv_metrics, load_birds_for_hex, load_invasive_sensitive,
    load_land_use, load_environmental_risks, load_aqueduct_center,
    load_eco_for_hex, load_all_avg_ffi
)

st.set_page_config(page_title="HelMos Biodiversity", page_icon="✨", layout="wide")

# ---------------- Sidebar (tabs + range like React) ----------------
cells_df = get_cells_df()
cell_ids = cells_df["h3_index"].tolist()
pos_map = dict(zip(cells_df["h3_index"], cells_df["position"]))

with st.sidebar:
    st.title("✨ HelMos Biodiversity")
    view = st.radio(
        "Views",
        ["overview", "biodiv", "eco", "pressure", "risk", "landcover"],
        format_func=lambda v: {
            "overview": "Overview",
            "biodiv": "Biodiversity State",
            "eco": "Ecosystem State",
            "pressure": "Pressures",
            "risk": "Risks",
            "landcover": "Land Cover",
        }[v],
        index=0,
    )
    ymin, ymax = st.slider("Time range", YEAR_MIN, YEAR_MAX, (2012, 2022), 1)

    hex_choice = st.selectbox(
        "Grid (for per-hex tables)",
        [f"{pos_map[h]} | {h}" for h in cell_ids],
        index=0
    )
    sel_h3 = hex_choice.split("|")[1].strip()
    sel_pos = pos_map[sel_h3]

    if view == "landcover":
        lc_year = st.number_input("Land-cover year", value=int(ymax), min_value=YEAR_MIN, max_value=YEAR_MAX, step=1)
        lc_opacity = st.slider("LC opacity", 0, 100, 70, 5)

# ---------------- H3 map base ----------------
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
        get_fill_color=[0, 0, 0, 0], get_line_color=[51, 65, 85, 200], line_width_min_pixels=1.5,
        stroked=True, filled=False, pickable=True
    )

def deck_with_layers(layers, tooltip="{label}"):
    return pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=9),
        map_style="light",
        tooltip={"text": tooltip}
    )

def file_link_caption(path: Path | None, label: str):
    if not path:
        return label
    url = to_git_url(path)
    return f"{label} — [{path.name}]({url})" if url else f"{label} — {path.name}"

# ---------------- LAND COVER ----------------
def page_landcover():
    st.markdown("### Region Map")
    df = discover_landcover()
    layers = [h3_outline_layer()]
    if df.empty:
        st.info("No GeoJSONs found in main directory.")
    else:
        subset = df[df["year"] == int(lc_year)]
        if subset.empty:
            st.info(f"No land-cover files for year {lc_year}.")
        else:
            alpha = int(round(255 * (lc_opacity / 100.0)))
            for _, r in subset.iterrows():
                gdf = gpd.read_file(r["file"])
                if gdf.empty or gdf.geometry.is_empty.all():
                    continue
                if "class" not in gdf.columns and "Class" in gdf.columns:
                    gdf.rename(columns={"Class": "class"}, inplace=True)
                gg = gdf.explode(index_parts=False, ignore_index=True)
                recs = []
                for _, gr in gg.iterrows():
                    geom = gr.geometry
                    if geom is None or geom.is_empty or geom.geom_type != "Polygon":
                        continue
                    coords = list(geom.exterior.coords)
                    cls = gr.get("class", 0)
                    # deterministic pseudo-palette by class
                    seed = int(cls) if pd.notna(cls) else 0
                    rng = np.random.default_rng(seed)
                    rC, gC, bC = rng.integers(60, 221, size=3).tolist()
                    recs.append({
                        "polygon": [[x, y] for (x, y) in coords],
                        "fill_color": [int(rC), int(gC), int(bC), alpha],
                        "stroke_color": [30, 30, 30, 120],
                        "label": f"{r['position']} | {r['tag']} | class {cls} | {r['year']}",
                    })
                if recs:
                    layers.append(pdk.Layer(
                        "PolygonLayer", data=recs, get_polygon="polygon",
                        get_fill_color="fill_color", get_line_color="stroke_color",
                        line_width_min_pixels=0.5, stroked=True, filled=True, pickable=True
                    ))
    st.pydeck_chart(deck_with_layers(layers))
    with st.expander("Source land-cover files"):
        if df.empty:
            st.write("—")
        else:
            for _, r in df[df["year"] == int(lc_year)].iterrows():
                st.markdown(f"- {file_link_caption(Path(r['file']), r['position'])}")

# ---------------- BIODIVERSITY ----------------
def page_biodiv():
    # Map shaded by alpha richness
    st.markdown("### Species Map")
    met = load_biodiv_metrics()
    layers = [h3_outline_layer()]
    if not met.empty:
        v = met.set_index("h3_index")["alpha"].reindex(cell_ids).fillna(0).values
        vmin, vmax = (float(np.min(v)), float(np.max(v))) if len(v) else (0.0, 1.0)
        def to_color(x):
            t = 0 if vmax == vmin else (x - vmin) / (vmax - vmin)
            r = int(round(255 * t)); g = int(round(180 * (1 - t))); b = int(round(120 + 80 * (1 - t)))
            return [r, g, b, 240]
        polys = []
        for i, h in enumerate(cell_ids):
            color = to_color(float(v[i]))
            coords = list(HEX_OUTLINES.iloc[i].geometry.exterior.coords)
            polys.append({"polygon": [[x, y] for (x, y) in coords], "fill_color": color, "label": f"{pos_map[h]} | alpha={v[i]:.0f}"})
        layers.append(pdk.Layer(
            "PolygonLayer", data=polys, get_polygon="polygon",
            get_fill_color="fill_color", get_line_color=[51, 65, 85, 220],
            line_width_min_pixels=1, stroked=True, filled=True, pickable=True
        ))
    st.pydeck_chart(deck_with_layers(layers))

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(file_link_caption(Path("Biodiversity_Metrics_By_H3.csv"), "**Mean Species Abundance (MSA)**"))
        if met.empty:
            st.info("Missing Biodiversity_Metrics_By_H3.csv")
        else:
            st.plotly_chart(px.bar(met, x="position", y="msa"), use_container_width=True)
    with col2:
        st.markdown("**Diversity (α, H′, E across grids)**")
        if met.empty:
            st.info("Missing Biodiversity_Metrics_By_H3.csv")
        else:
            trip = met[["position", "alpha", "shannon", "pielou"]].copy()
            fig = go.Figure()
            for k in ["alpha", "shannon", "pielou"]:
                fig.add_bar(x=trip["position"], y=trip[k], name=k.capitalize())
            fig.update_layout(barmode="group", margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

    # Birds table
    birds, f_birds = load_birds_for_hex(sel_h3, sel_pos)
    st.markdown(file_link_caption(f_birds, "**Birds in the Region**"))
    st.dataframe(birds, use_container_width=True, hide_index=True)

# ---------------- ECOSYSTEM STATE ----------------
def page_eco():
    st.markdown("### Region Map")
    met = load_biodiv_metrics()
    layers = [h3_outline_layer()]
    if not met.empty:
        v = met.set_index("h3_index")["alpha"].reindex(cell_ids).fillna(0).values
        vmin, vmax = (float(np.min(v)), float(np.max(v))) if len(v) else (0.0, 1.0)
        def to_color(x):
            t = 0 if vmax == vmin else (x - vmin) / (vmax - vmin)
            r = int(round(255 * t)); g = int(round(180 * (1 - t))); b = int(round(120 + 80 * (1 - t)))
            return [r, g, b, 240]
        polys = []
        for i, h in enumerate(cell_ids):
            color = to_color(float(v[i]))
            coords = list(HEX_OUTLINES.iloc[i].geometry.exterior.coords)
            polys.append({"polygon": [[x, y] for (x, y) in coords], "fill_color": color, "label": f"{pos_map[h]} | alpha={v[i]:.0f}"})
        layers.append(pdk.Layer(
            "PolygonLayer", data=polys, get_polygon="polygon",
            get_fill_color="fill_color", get_line_color=[51, 65, 85, 220],
            line_width_min_pixels=1, stroked=True, filled=True, pickable=True
        ))
    st.pydeck_chart(deck_with_layers(layers))

    # Environmental risks — H3 first, then position
    env, f_env = load_environmental_risks(sel_h3, sel_pos)
    label_env = "**Protected ecosystems / environmental risks**"
    st.markdown((f"{label_env} — [{Path(f_env).name}]({to_git_url(f_env)})") if f_env and to_git_url(f_env) else label_env)
    if env.empty:
        st.info("No environmental_risks file found for this grid.")
    else:
        st.dataframe(env, use_container_width=True, hide_index=True)

    # FFI (ALL grids)
    st.markdown("**Fragmentation (avg FFI) — all grids**")
    ffi_all, ffi_paths = load_all_avg_ffi()
    if ffi_all.empty:
        st.info("No avg_ffi_<h3>_<pos>.csv files found.")
    else:
        mask = (ffi_all["year"] >= ymin) & (ffi_all["year"] <= ymax)
        plot_df = ffi_all.loc[mask].copy()
        plot_df["grid"] = plot_df["position"]
        fig = px.line(plot_df, x="year", y="value", color="grid", markers=True)
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h"))
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("FFI source files"):
            for p in sorted(set(ffi_paths), key=lambda x: x.name):
                url = to_git_url(p)
                st.markdown(f"- {p.name}" + (f" — [open]({url})" if url else ""))

    # Keep per-hex eco tables to mirror React
    eco = load_eco_for_hex(sel_h3, sel_pos)
    row = st.columns(3)
    hi_df, hi_path = eco["high_integrity"]
    with row[0]:
        lbl = "**High integrity (2022)**"
        st.markdown((f"{lbl} — [{Path(hi_path).name}]({to_git_url(hi_path)})") if hi_path and to_git_url(hi_path) else lbl)
        st.dataframe((hi_df.head(200) if hi_df is not None else pd.DataFrame()), use_container_width=True, hide_index=True)
    cor_df, cor_path = eco["corridors"]
    with row[1]:
        lbl = "**Corridors (2022)**"
        st.markdown((f"{lbl} — [{Path(cor_path).name}]({to_git_url(cor_path)})") if cor_path and to_git_url(cor_path) else lbl)
        st.dataframe((cor_df.head(200) if cor_df is not None else pd.DataFrame()), use_container_width=True, hide_index=True)
    rd_df, rd_path = eco["rapid_decline"]
    with row[2]:
        lbl = "**Rapid decline (2005–2022)**"
        st.markdown((f"{lbl} — [{Path(rd_path).name}]({to_git_url(rd_path)})") if rd_path and to_git_url(rd_path) else lbl)
        st.dataframe((rd_df.head(200) if rd_df is not None else pd.DataFrame()), use_container_width=True, hide_index=True)

# ---------------- PRESSURES ----------------
def page_pressure():
    st.markdown("### Region Map")
    # Shade by number of activities files rows per hex (simple proxy)
    layers = [h3_outline_layer()]
    scores = {}
    for h in cell_ids:
        pos = pos_map[h]
        df, _ = load_land_use(h, pos)
        if not df.empty:
            scores[h] = float(len(df))
    if scores:
        vals = np.array([scores.get(h, 0.0) for h in cell_ids])
        vmin, vmax = float(vals.min()), float(vals.max()) if len(vals) > 0 else (0.0, 1.0)
        def to_color(x):
            t = 0 if vmax == vmin else (x - vmin) / (vmax - vmin)
            r = int(round(255 * t)); g = int(round(180 * (1 - t))); b = int(round(120 + 80 * (1 - t)))
            return [r, g, b, 240]
        polys = []
        for i, h in enumerate(cell_ids):
            color = to_color(float(scores.get(h, 0.0)))
            coords = list(HEX_OUTLINES.iloc[i].geometry.exterior.coords)
            polys.append({"polygon": [[x, y] for (x, y) in coords], "fill_color": color, "label": f"{pos_map[h]} | activities={scores.get(h, 0):.0f}"})
        layers.append(pdk.Layer(
            "PolygonLayer", data=polys, get_polygon="polygon",
            get_fill_color="fill_color", get_line_color=[51, 65, 85, 220],
            line_width_min_pixels=1, stroked=True, filled=True, pickable=True
        ))
    st.pydeck_chart(deck_with_layers(layers))

    act, f_act = load_land_use(sel_h3, sel_pos)
    st.markdown(file_link_caption(f_act, "**Activities (impact)**"))
    if act.empty:
        st.info("No Land_use_{h3}_{pos}.csv for selected grid.")
    else:
        st.dataframe(act, use_container_width=True, hide_index=True)

    st.caption("Climate drivers — plug in your time series here if desired.")

# ---------------- RISKS ----------------
def page_risk():
    st.markdown("### Region Map")
    center_h3 = cells_df.loc[cells_df["position"] == "CENTER", "h3_index"].iloc[0]
    layers = [h3_outline_layer()]
    for i, h in enumerate(cell_ids):
        if h != center_h3:
            continue
        coords = list(HEX_OUTLINES.iloc[i].geometry.exterior.coords)
        layers.append(pdk.Layer(
            "PolygonLayer", data=[{"polygon": [[x, y] for (x, y) in coords], "label": "CENTER"}],
            get_polygon="polygon", get_fill_color=[80, 160, 255, 220],
            get_line_color=[51, 65, 85, 220], line_width_min_pixels=1, stroked=True, filled=True, pickable=True
        ))
    st.pydeck_chart(deck_with_layers(layers))

    aq, f_aq = load_aqueduct_center(center_h3)
    st.markdown(file_link_caption(f_aq, "**Water risks (radar)**"))
    if aq.empty:
        st.info("No aqueduct_v4_{CENTER}.csv found.")
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=aq["score"], theta=aq["dimension"], fill="toself",
            text=aq["label"], hovertemplate="%{theta}<br>%{text}<extra></extra>"
        ))
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    inv, sens, f_sp = load_invasive_sensitive(sel_h3, sel_pos)
    st.markdown(file_link_caption(f_sp, "**Sensitive species**"))
    st.dataframe(sens.head(200), use_container_width=True, hide_index=True)

# ---------------- OVERVIEW ----------------
def page_overview():
    st.markdown("### Region Map")
    met = load_biodiv_metrics()
    layers = [h3_outline_layer()]
    if not met.empty:
        v = met.set_index("h3_index")["alpha"].reindex(cell_ids).fillna(0).values
        vmin, vmax = (float(np.min(v)), float(np.max(v))) if len(v) else (0.0, 1.0)
        def to_color(x):
            t = 0 if vmax == vmin else (x - vmin) / (vmax - vmin)
            r = int(round(255 * t)); g = int(round(180 * (1 - t))); b = int(round(120 + 80 * (1 - t)))
            return [r, g, b, 240]
        polys = []
        for i, h in enumerate(cell_ids):
            color = to_color(float(v[i]))
            coords = list(HEX_OUTLINES.iloc[i].geometry.exterior.coords)
            polys.append({"polygon": [[x, y] for (x, y) in coords], "fill_color": color, "label": f"{pos_map[h]} | alpha={v[i]:.0f}"})
        layers.append(pdk.Layer(
            "PolygonLayer", data=polys, get_polygon="polygon",
            get_fill_color="fill_color", get_line_color=[51, 65, 85, 220],
            line_width_min_pixels=1, stroked=True, filled=True, pickable=True
        ))
    st.pydeck_chart(deck_with_layers(layers))

    # KPIs
    if met.empty:
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Avg. richness", "—")
        with c2: st.metric("Avg. abundance", "—")
        with c3: st.metric("Protected cover", "—")
        with c4: st.metric("Water risk score", "—")
    else:
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Avg. richness", f"{met['alpha'].mean():.2f}")
        with c2: st.metric("Avg. abundance (H′)", f"{met['shannon'].mean():.2f}")
        with c3: st.metric("Protected cover", "—")
        with c4: st.metric("Water risk score", "—")

    colA, colB = st.columns(2)
    with colA:
        if met.empty:
            st.info("Biodiversity metrics file missing.")
        else:
            fig = go.Figure()
            for _, row in met.iterrows():
                fig.add_trace(go.Scatterpolar(
                    r=[row["alpha"], row["shannon"]],
                    theta=["richness (α)", "diversity (H′)"],
                    name=row["position"],
                    fill="toself"
                ))
            fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h"))
            st.plotly_chart(fig, use_container_width=True)

    with colB:
        center_h3 = cells_df.loc[cells_df["position"] == "CENTER", "h3_index"].iloc[0]
        aq, f_aq = load_aqueduct_center(center_h3)
        st.markdown(file_link_caption(f_aq, "**Water Risks (dimensions)**"))
        if aq.empty:
            st.info("No Aqueduct V4 CSV found.")
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=aq["score"], theta=aq["dimension"], fill="toself",
                text=aq["label"], hovertemplate="%{theta}<br>%{text}<extra></extra>"
            ))
            fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

# ---------------- Router ----------------
if view == "landcover":
    page_landcover()
elif view == "biodiv":
    page_biodiv()
elif view == "eco":
    page_eco()
elif view == "pressure":
    page_pressure()
elif view == "risk":
    page_risk()
else:
    page_overview()


