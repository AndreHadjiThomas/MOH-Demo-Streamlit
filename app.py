from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
import plotly.express as px
import plotly.graph_objects as go
import geopandas as gpd  # for land cover & KML

from data_config import (
    YEAR_MIN, YEAR_MAX, GIT_BASE_URL
)
from data_loaders import (
    get_cells_df, hex_outline_gdf, discover_landcover, to_git_url,
    load_biodiv_metrics, load_birds_for_hex, load_invasive_sensitive,
    load_land_use, load_environmental_risks, load_env_timeseries, compute_climate_intensity,
    load_aqueduct_center, load_eco_for_hex, load_all_avg_ffi, load_wind_kml
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
        "Your Grid",
        [f"{pos_map[h]} | {h}" for h in cell_ids],
        index=0
    )
    sel_h3 = hex_choice.split("|")[1].strip()
    sel_pos = pos_map[sel_h3]

    # Land cover overlay options (applies to all maps)
    lc_year = st.number_input("Land-cover overlay year", value=int(ymax), min_value=YEAR_MIN, max_value=YEAR_MAX, step=1)
    lc_opacity = st.slider("Land-cover overlay opacity", 0, 100, 50, 5)

    # FFI filters
    ffi_positions = st.multiselect("FFI grids (multi-select)", options=[pos_map[h] for h in cell_ids], default=[pos_map[h] for h in cell_ids])
    ffi_threshold = st.slider("FFI threshold (fade lines below)", 0.0, 1.0, 0.5, 0.01)

# ---------------- Map helpers ----------------
HEX_OUTLINES = hex_outline_gdf(cell_ids)
bounds = HEX_OUTLINES.total_bounds  # minx, miny, maxx, maxy
center_lat = (bounds[1] + bounds[3]) / 2
center_lon = (bounds[0] + bounds[2]) / 2

def h3_outline_layer():
    # Label "Your Grid" for selected hex
    data = []
    for _, row in HEX_OUTLINES.iterrows():
        coords = list(row.geometry.exterior.coords)
        label = "Your Grid" if row["h3_index"] == sel_h3 else row["h3_index"]
        data.append({"polygon": [[x, y] for (x, y) in coords], "label": label})
    return pdk.Layer(
        "PolygonLayer", data=data, get_polygon="polygon",
        get_fill_color=[0, 0, 0, 0], get_line_color=[51, 65, 85, 200], line_width_min_pixels=1.5,
        stroked=True, filled=False, pickable=True
    )

def landcover_layers(year: int, opacity_pct: int):
    df = discover_landcover()
    if df.empty:
        return []
    subset = df[df["year"] == int(year)]
    if subset.empty:
        return []
    alpha = int(round(255 * (opacity_pct / 100.0)))
    layers = []
    for _, r in subset.iterrows():
        try:
            gdf = gpd.read_file(r["file"])
        except Exception:
            continue
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
    return layers

def windfarm_layer():
    gdf, pth = load_wind_kml()
    if gdf.empty:
        return None
    recs = []
    gg = gdf.explode(index_parts=False, ignore_index=True)
    for _, gr in gg.iterrows():
        geom = gr.geometry
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type == "Polygon":
            polys = [geom]
        elif geom.geom_type == "MultiPolygon":
            polys = list(geom.geoms)
        else:
            continue
        for poly in polys:
            coords = list(poly.exterior.coords)
            recs.append({
                "polygon": [[x, y] for (x, y) in coords],
                "label": "Windfarm polygon"
            })
    return pdk.Layer(
        "PolygonLayer", data=recs, get_polygon="polygon",
        get_fill_color=[255, 140, 0, 90], get_line_color=[255, 140, 0, 220],
        line_width_min_pixels=2, stroked=True, filled=True, pickable=True
    )

def deck_with_layers(layers, tooltip="{label}"):
    full_layers = [h3_outline_layer()]
    full_layers.extend(landcover_layers(lc_year, lc_opacity))
    wf = windfarm_layer()
    if wf:
        full_layers.append(wf)
    full_layers.extend(layers)
    return pdk.Deck(
        layers=full_layers,
        initial_view_state=pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=9),
        map_style="light",
        tooltip={"text": tooltip}
    )

def color_ramp_fn(vals):
    v = np.array(vals)
    vmin, vmax = (float(np.min(v)), float(np.max(v))) if len(v) else (0.0, 1.0)
    def to_color(x):
        t = 0 if vmax == vmin else (x - vmin) / (vmax - vmin)
        r = int(round(255 * t)); g = int(round(180 * (1 - t))); b = int(round(120 + 80 * (1 - t)))
        return [r, g, b, 240]
    return to_color

# ---------------- LAND COVER PAGE ----------------
def page_landcover():
    st.markdown("### Region Map (with Land Cover overlay)")
    st.pydeck_chart(deck_with_layers([]))

# ---------------- BIODIVERSITY ----------------
def page_biodiv():
    st.markdown("### Species Map")
    met = load_biodiv_metrics()
    layers = []
    if not met.empty:
        vals = met.set_index("h3_index")["alpha"].reindex(cell_ids).fillna(0).values
        to_color = color_ramp_fn(vals)
        polys = []
        for i, h in enumerate(cell_ids):
            color = to_color(float(vals[i]))
            coords = list(HEX_OUTLINES.iloc[i].geometry.exterior.coords)
            label = "Your Grid" if h == sel_h3 else f"{pos_map[h]} | α={vals[i]:.0f}"
            polys.append({"polygon": [[x, y] for (x, y) in coords], "fill_color": color, "label": label})
        layers.append(pdk.Layer(
            "PolygonLayer", data=polys, get_polygon="polygon",
            get_fill_color="fill_color", get_line_color=[51, 65, 85, 220],
            line_width_min_pixels=1, stroked=True, filled=True, pickable=True
        ))
    st.pydeck_chart(deck_with_layers(layers))

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Mean Species Abundance (MSA)** — [Biodiversity_Metrics_By_H3.csv]({})".format(to_git_url(Path("Biodiversity_Metrics_By_H3.csv")) if to_git_url(Path("Biodiversity_Metrics_By_H3.csv")) else ""))
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

    birds, f_birds = load_birds_for_hex(sel_h3, sel_pos)
    st.markdown(("**Birds in Your Grid** — [{}]({})".format(Path(f_birds).name, to_git_url(f_birds))) if f_birds and to_git_url(f_birds) else "**Birds in Your Grid**")
    st.dataframe(birds, use_container_width=True, hide_index=True)

# ---------------- ECOSYSTEM STATE ----------------
def _format_table(df: pd.DataFrame, rename_map: dict | None = None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    t = df.copy()
    if rename_map:
        cols = {c: rename_map.get(c, c) for c in t.columns}
        t.rename(columns=cols, inplace=True)
    return t

def page_eco():
    st.markdown("### Region Map")
    met = load_biodiv_metrics()
    layers = []
    if not met.empty:
        vals = met.set_index("h3_index")["alpha"].reindex(cell_ids).fillna(0).values
        to_color = color_ramp_fn(vals)
        polys = []
        for i, h in enumerate(cell_ids):
            color = to_color(float(vals[i]))
            coords = list(HEX_OUTLINES.iloc[i].geometry.exterior.coords)
            label = "Your Grid" if h == sel_h3 else f"{pos_map[h]} | α={vals[i]:.0f}"
            polys.append({"polygon": [[x, y] for (x, y) in coords], "fill_color": color, "label": label})
        layers.append(pdk.Layer(
            "PolygonLayer", data=polys, get_polygon="polygon",
            get_fill_color="fill_color", get_line_color=[51, 65, 85, 220],
            line_width_min_pixels=1, stroked=True, filled=True, pickable=True
        ))
    st.pydeck_chart(deck_with_layers(layers))

    # Protected ecosystems / environmental risks (tabular)
    env, f_env = load_environmental_risks(sel_h3, sel_pos)
    st.markdown(("**Protected ecosystems / environmental risks — Your Grid** — [{}]({})".format(Path(f_env).name, to_git_url(f_env))) if f_env and to_git_url(f_env) else "**Protected ecosystems / environmental risks — Your Grid**")
    if env.empty:
        st.info("No environmental_risks file found for Your Grid.")
    else:
        # Try to standardize: keep name/area/cover columns if present
        rename_map = {
            "Name": "Name", "name": "Name",
            "Area": "Area", "area": "Area", "Area_km2": "Area", "area_km2": "Area",
            "% Cover": "% Cover", "PercentCover": "% Cover", "cover_pct": "% Cover"
        }
        st.dataframe(_format_table(env, rename_map), use_container_width=True, hide_index=True)

    # FFI (ALL grids) with filters + threshold
    st.markdown("**Fragmentation (avg FFI) — all grids**")
    ffi_all, _ = load_all_avg_ffi()
    if ffi_all.empty:
        st.info("No avg_ffi_<h3>_<pos>.csv files found.")
    else:
        mask = (ffi_all["year"] >= ymin) & (ffi_all["year"] <= ymax) & (ffi_all["position"].isin(ffi_positions))
        plot_df = ffi_all.loc[mask].copy()
        if plot_df.empty:
            st.info("No data for current filters.")
        else:
            # Build style via separate traces so we can fade below threshold
            fig = go.Figure()
            for pos in sorted(plot_df["position"].unique()):
                d = plot_df[plot_df["position"] == pos]
                opacity = 1.0 if (d["value"].max() >= ffi_threshold) else 0.25
                fig.add_trace(go.Scatter(x=d["year"], y=d["value"], mode="lines+markers", name=pos, opacity=opacity))
            fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h"))
            st.plotly_chart(fig, use_container_width=True)

    # Corridors & High integrity & Rapid decline — tidy tables for Your Grid
    eco = load_eco_for_hex(sel_h3, sel_pos)
    c1, c2, c3 = st.columns(3)
    hi_df, hi_path = eco["high_integrity"]
    with c1:
        st.markdown(("**High integrity (2022) — Your Grid** — [{}]({})".format(Path(hi_path).name, to_git_url(hi_path))) if hi_path and to_git_url(hi_path) else "**High integrity (2022) — Your Grid**")
        st.dataframe(_format_table(hi_df), use_container_width=True, hide_index=True)
    cor_df, cor_path = eco["corridors"]
    with c2:
        st.markdown(("**Corridors (2022) — Your Grid** — [{}]({})".format(Path(cor_path).name, to_git_url(cor_path))) if cor_path and to_git_url(cor_path) else "**Corridors (2022) — Your Grid**")
        st.dataframe(_format_table(cor_df), use_container_width=True, hide_index=True)
    rd_df, rd_path = eco["rapid_decline"]
    with c3:
        st.markdown(("**Rapid decline (2005–2022) — Your Grid** — [{}]({})".format(Path(rd_path).name, to_git_url(rd_path))) if rd_path and to_git_url(rd_path) else "**Rapid decline (2005–2022) — Your Grid**")
        st.dataframe(_format_table(rd_df), use_container_width=True, hide_index=True)

# ---------------- PRESSURES ----------------
def page_pressure():
    st.markdown("### Region Map")
    # Shade by number of activities per hex
    layers = []
    scores = {}
    for h in cell_ids:
        pos = pos_map[h]
        df, _ = load_land_use(h, pos)
        if not df.empty:
            scores[h] = float(len(df))
    if scores:
        vals = np.array([scores.get(h, 0.0) for h in cell_ids])
        to_color = color_ramp_fn(vals)
        polys = []
        for i, h in enumerate(cell_ids):
            color = to_color(float(scores.get(h, 0.0)))
            coords = list(HEX_OUTLINES.iloc[i].geometry.exterior.coords)
            label = "Your Grid" if h == sel_h3 else f"{pos_map[h]} | activities={scores.get(h, 0):.0f}"
            polys.append({"polygon": [[x, y] for (x, y) in coords], "fill_color": color, "label": label})
        layers.append(pdk.Layer(
            "PolygonLayer", data=polys, get_polygon="polygon",
            get_fill_color="fill_color", get_line_color=[51, 65, 85, 220],
            line_width_min_pixels=1, stroked=True, filled=True, pickable=True
        ))
    st.pydeck_chart(deck_with_layers(layers))

    # Activities (impact + TagValue)
    act, f_act = load_land_use(sel_h3, sel_pos)
    st.markdown(("**Activities (impact) — Your Grid** — [{}]({})".format(Path(f_act).name, to_git_url(f_act))) if f_act and to_git_url(f_act) else "**Activities (impact) — Your Grid**")
    if act.empty:
        st.info("No Land_use_{h3}_{pos}.csv for Your Grid.")
    else:
        st.dataframe(act, use_container_width=True, hide_index=True)

    # Invasive species (Pressures)
    inv, sens, sfile = load_invasive_sensitive(sel_h3, sel_pos)
    st.markdown(("**Invasive species — Your Grid** — [{}]({})".format(Path(sfile).name, to_git_url(sfile))) if sfile and to_git_url(sfile) else "**Invasive species — Your Grid**")
    st.dataframe(inv.head(200), use_container_width=True, hide_index=True)

    # Climate drivers & intensity metric
    envts, f_envts = load_env_timeseries(sel_pos)
    st.markdown(("**Climate drivers — Your Grid** — [{}]({})".format(Path(f_envts).name, to_git_url(f_envts))) if f_envts and to_git_url(f_envts) else "**Climate drivers — Your Grid**")
    if envts.empty:
        st.info("No environmental_data_<POSITION>_.csv found for Your Grid.")
    else:
        # plot all numeric driver columns vs year
        num_cols = [c for c in envts.columns if c != "year" and pd.api.types.is_numeric_dtype(envts[c])]
        if num_cols:
            plot_df = envts[(envts["year"] >= ymin) & (envts["year"] <= ymax)]
            fig = go.Figure()
            for c in num_cols:
                fig.add_trace(go.Scatter(x=plot_df["year"], y=plot_df[c], mode="lines", name=c))
            fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h"))
            st.plotly_chart(fig, use_container_width=True)
        # KPI: climate intensity
        ci = compute_climate_intensity(envts, ymin, ymax)
        st.metric("Climate intensity (z-score blend)", f"{ci:.2f}" if pd.notna(ci) else "—")

# ---------------- RISKS ----------------
def page_risk():
    st.markdown("### Region Map")
    # Full map (not only center)
    layers = []
    # Simple fill to show all grids equally (risk lines are in radar below)
    polys = []
    for i, h in enumerate(cell_ids):
        coords = list(HEX_OUTLINES.iloc[i].geometry.exterior.coords)
        label = "Your Grid" if h == sel_h3 else pos_map[h]
        polys.append({"polygon": [[x, y] for (x, y) in coords], "fill_color": [80,160,255,80], "label": label})
    layers.append(pdk.Layer(
        "PolygonLayer", data=polys, get_polygon="polygon",
        get_fill_color="fill_color", get_line_color=[51, 65, 85, 220],
        line_width_min_pixels=1, stroked=True, filled=True, pickable=True
    ))
    st.pydeck_chart(deck_with_layers(layers))

    # Water risk radar (nicely named dimensions)
    center_h3 = cells_df.loc[cells_df["position"] == "CENTER", "h3_index"].iloc[0]
    aq, f_aq = load_aqueduct_center(center_h3)
    st.markdown(("**Water risks (radar) — Center** — [{}]({})".format(Path(f_aq).name, to_git_url(f_aq))) if f_aq and to_git_url(f_aq) else "**Water risks (radar) — Center**")
    if aq.empty:
        st.info("No Aqueduct V4 CSV found for the central grid.")
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=aq["score"], theta=aq["dimension"], fill="toself",
            text=aq["label"], hovertemplate="%{theta}<br>%{text}<extra></extra>"
        ))
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    # Sensitive species checklist (Your Grid)
    inv, sens, sfile = load_invasive_sensitive(sel_h3, sel_pos)
    st.markdown(("**Sensitive species (VU/EN/CR) — Your Grid** — [{}]({})".format(Path(sfile).name, to_git_url(sfile))) if sfile and to_git_url(sfile) else "**Sensitive species (VU/EN/CR) — Your Grid**")
    st.dataframe(sens.head(500), use_container_width=True, hide_index=True)

# ---------------- OVERVIEW ----------------
def page_overview():
    st.markdown("### Region Map")
    met = load_biodiv_metrics()
    layers = []
    if not met.empty:
        vals = met.set_index("h3_index")["alpha"].reindex(cell_ids).fillna(0).values
        to_color = color_ramp_fn(vals)
        polys = []
        for i, h in enumerate(cell_ids):
            color = to_color(float(vals[i]))
            coords = list(HEX_OUTLINES.iloc[i].geometry.exterior.coords)
            label = "Your Grid" if h == sel_h3 else f"{pos_map[h]} | α={vals[i]:.0f}"
            polys.append({"polygon": [[x, y] for (x, y) in coords], "fill_color": color, "label": label})
        layers.append(pdk.Layer(
            "PolygonLayer", data=polys, get_polygon="polygon",
            get_fill_color="fill_color", get_line_color=[51, 65, 85, 220],
            line_width_min_pixels=1, stroked=True, filled=True, pickable=True
        ))
    st.pydeck_chart(deck_with_layers(layers))

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    if met.empty:
        with c1: st.metric("Avg. richness", "—")
        with c2: st.metric("Avg. abundance (H′)", "—")
        with c3: st.metric("Protected cover (km²)", "—")
        with c4: st.metric("Water risk score", "—")
    else:
        with c1: st.metric("Avg. richness", f"{met['alpha'].mean():.2f}")
        with c2: st.metric("Avg. abundance (H′)", f"{met['shannon'].mean():.2f}")
        # Protected cover KPI: sum area from env_risks tables across all grids if possible
        total_protected_km2 = 0.0
        for h in cell_ids:
            pos = pos_map[h]
            env, _ = load_environmental_risks(h, pos)
            if env is not None and not env.empty:
                # Try columns that look like area in km2
                area_col = None
                for cand in ["Area", "area", "Area_km2", "area_km2", "Area (km2)", "Area_km²"]:
                    if cand in env.columns:
                        area_col = cand; break
                if area_col is None:
                    # try to parse from strings like "45.7 km²"
                    for c in env.columns:
                        if env[c].astype(str).str.contains("km").any():
                            try:
                                vals = env[c].astype(str).str.extract(r"([\d\.]+)").astype(float)
                                total_protected_km2 += float(vals.fillna(0).sum())
                                area_col = c
                                break
                            except Exception:
                                pass
                if area_col:
                    try:
                        total_protected_km2 += pd.to_numeric(env[area_col], errors="coerce").fillna(0).sum()
                    except Exception:
                        pass
        with c3: st.metric("Protected cover (km²)", f"{total_protected_km2:.1f}")
        with c4: st.metric("Water risk score", "—")  # stays placeholder (no single scalar in inputs)

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
        st.markdown(("**Water Risks (dimensions)** — [{}]({})".format(Path(f_aq).name, to_git_url(f_aq))) if f_aq and to_git_url(f_aq) else "**Water Risks (dimensions)**")
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
