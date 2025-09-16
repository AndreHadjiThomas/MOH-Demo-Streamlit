from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
import plotly.express as px
import plotly.graph_objects as go
import geopandas as gpd

from data_config import YEAR_MIN, YEAR_MAX
from data_loaders import (
    get_cells_df, hex_outline_gdf, discover_landcover, to_git_url,
    load_biodiv_metrics, load_birds_for_hex, load_invasive_sensitive,
    load_land_use, load_environmental_risks, load_env_timeseries, compute_climate_intensity,
    load_aqueduct_center, load_eco_for_hex, load_all_avg_ffi, load_wind_kml,
    aggregate_activities_all_grids, get_activity_row, PRESSURE_COLS
)

st.set_page_config(page_title="HelMos Biodiversity", page_icon="✨", layout="wide")

# ---------------- Sidebar ----------------
cells_df = get_cells_df()
cell_ids = cells_df["h3_index"].tolist()
pos_map = dict(zip(cells_df["h3_index"], cells_df["position"]))

with st.sidebar:
    st.title("✨ HelMos Biodiversity")
    view = st.radio(
        "Views",
        ["overview", "biodiv", "eco", "pressure", "risk"],
        format_func=lambda v: {
            "overview": "Overview",
            "biodiv": "Biodiversity State",
            "eco": "Ecosystem State",
            "pressure": "Activities / Pressures",
            "risk": "Risks",
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

    # Land cover overlay controls
    lc_year = st.number_input("Land-cover overlay year", value=int(ymax), min_value=YEAR_MIN, max_value=YEAR_MAX, step=1)
    lc_opacity = st.slider("Land-cover overlay opacity", 0, 100, 50, 5)

    # FFI filters
    ffi_positions = st.multiselect("FFI grids", options=[pos_map[h] for h in cell_ids], default=[pos_map[h] for h in cell_ids])
    ffi_threshold = st.slider("FFI threshold (fade lines below)", 0.0, 1.0, 0.5, 0.01)

# ---------------- Map helpers ----------------
HEX_OUTLINES = hex_outline_gdf(cell_ids)
bounds = HEX_OUTLINES.total_bounds
center_lat = (bounds[1] + bounds[3]) / 2
center_lon = (bounds[0] + bounds[2]) / 2

def h3_outline_layer():
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
        polys = [geom] if geom.geom_type == "Polygon" else (list(geom.geoms) if geom.geom_type == "MultiPolygon" else [])
        for poly in polys:
            coords = list(poly.exterior.coords)
            recs.append({"polygon": [[x, y] for (x, y) in coords], "label": "Windfarm polygon"})
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

# ---------- Small helpers for tables ----------
def pick_columns_type_distance(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["Type", "Distance"])
    cols = {c.lower(): c for c in df.columns}
    type_col = (cols.get("class") or cols.get("landcover") or cols.get("ecosystem") or
                cols.get("type") or cols.get("tagvalue") or next(iter(df.columns)))
    dist_col = (cols.get("distance") or cols.get("distance_m") or cols.get("distance_km") or
                cols.get("dist_m") or cols.get("dist_km"))
    out = pd.DataFrame()
    out["Type"] = df[type_col].astype(str)
    out["Distance"] = df[dist_col] if dist_col else ""
    return out

def pick_name_distance(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["Name", "Distance"])
    cols = {c.lower(): c for c in df.columns}
    name_col = (cols.get("name") or cols.get("site") or cols.get("protected_area") or
                cols.get("pa_name") or next(iter(df.columns)))
    dist_col = (cols.get("distance") or cols.get("distance_m") or cols.get("distance_km") or
                cols.get("dist_m") or cols.get("dist_km"))
    out = pd.DataFrame()
    out["Name"] = df[name_col].astype(str)
    out["Distance"] = df[dist_col] if dist_col else ""
    return out

# ---------------- OVERVIEW ----------------
def page_overview():
    st.markdown("### Region Map (with Land Cover overlay)")
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
        # Protected cover KPI: sum area from environmental_risks tables across all grids
        total_protected_km2 = 0.0
        for h in cell_ids:
            pos = pos_map[h]
            env, _ = load_environmental_risks(h, pos)
            if env is None or env.empty:
                continue
            area_col = None
            for cand in ["Area", "area", "Area_km2", "area_km2", "Area (km2)", "Area_km²"]:
                if cand in env.columns:
                    area_col = cand; break
            if area_col:
                try:
                    total_protected_km2 += pd.to_numeric(env[area_col], errors="coerce").fillna(0).sum()
                    continue
                except Exception:
                    pass
            for c in env.columns:
                if env[c].astype(str).str.contains("km").any():
                    try:
                        vals = env[c].astype(str).str.extract(r"([\d\.]+)").astype(float)
                        total_protected_km2 += float(vals.fillna(0).sum())
                        break
                    except Exception:
                        pass
        with c3: st.metric("Protected cover (km²)", f"{total_protected_km2:.1f}")
        with c4: st.metric("Water risk score", "—")

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
        st.markdown("**Mean Species Abundance (MSA)**")
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

    birds, _ = load_birds_for_hex(sel_h3, sel_pos)
    st.markdown("**Birds — Your Grid**")
    st.dataframe(birds, use_container_width=True, hide_index=True)

# ---------------- ECOSYSTEM STATE ----------------
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

    # Protected ecosystems across ALL grids
    st.markdown("**Protected ecosystems — All Grids (Name & Distance)**")
    prot_rows = []
    for h in cell_ids:
        pos = pos_map[h]
        env, _ = load_environmental_risks(h, pos)
        if env is None or env.empty:
            continue
        nd = pick_name_distance(env)
        if not nd.empty:
            nd["Grid"] = pos
            prot_rows.append(nd[["Grid", "Name", "Distance"]])
    if prot_rows:
        all_prot = pd.concat(prot_rows, ignore_index=True)
        st.dataframe(all_prot, use_container_width=True, hide_index=True)
    else:
        st.info("No environmental_risks files found to list protected ecosystems.")

    # Eco details (Your Grid): Type + Distance
    eco = load_eco_for_hex(sel_h3, sel_pos)
    c1, c2, c3 = st.columns(3)
    hi_df, _ = eco["high_integrity"]
    with c1:
        st.markdown("**High integrity (2022) — Your Grid**")
        st.dataframe(pick_columns_type_distance(hi_df), use_container_width=True, hide_index=True)
    cor_df, _ = eco["corridors"]
    with c2:
        st.markdown("**Corridors (2022) — Your Grid**")
        st.dataframe(pick_columns_type_distance(cor_df), use_container_width=True, hide_index=True)
    rd_df, _ = eco["rapid_decline"]
    with c3:
        st.markdown("**Rapid decline (2005–2022) — Your Grid**")
        st.dataframe(pick_columns_type_distance(rd_df), use_container_width=True, hide_index=True)

    # FFI (ALL grids) with filters + softer fade
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
            fig = go.Figure()
            for pos in sorted(plot_df["position"].unique()):
                d = plot_df[plot_df["position"] == pos]
                opacity = 1.0 if (d["value"].max() >= ffi_threshold) else 0.7
                fig.add_trace(go.Scatter(x=d["year"], y=d["value"], mode="lines+markers", name=pos, opacity=opacity))
            fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h"))
            st.plotly_chart(fig, use_container_width=True)

# ---------------- ACTIVITIES / PRESSURES ----------------
def page_pressure():
    st.markdown("### Activities")

    all_acts = aggregate_activities_all_grids(cells_df)

    if all_acts.empty:
        st.error("No activities found anywhere. I couldn’t read Land_use files or they had no usable columns.")
        st.stop()

    # If pressures are all NaN, we’re in fallback mode
    have_any_pressure = any(
        (p in all_acts.columns) and all_acts[p].notna().any()
        for p in PRESSURE_COLS
    )
    if not have_any_pressure:
        with st.expander("Why am I seeing only a bare activities list?"):
            st.write(
                "- The Activity→Pressure Excel mapping could not be used or didn’t match any TagValue.\n"
                "- I’m showing a fallback table: grouped TagValue across all grids with total area.\n"
                "- To enable the pressure widget, ensure your Excel is readable and names align (synonyms help)."
            )

    show_cols = ["Activity","TotalArea_m2","Sectors","Businesses"]
    st.dataframe(all_acts[show_cols], use_container_width=True, hide_index=True)

    pick = st.selectbox(
        "Pick an activity from the table to view its pressure profile",
        options=all_acts["Activity"].tolist()
    )
    row = get_activity_row(all_acts, pick)
    if row is None:
        st.info("Selected activity not found.")
        return

    press_df = pd.DataFrame([
        {"Pressure": p, "Value": (float(row[p]) if (p in row and pd.notna(row[p])) else np.nan)}
        for p in PRESSURE_COLS
    ])
    # If all NaN → we’re in fallback; show a note
    if press_df["Value"].isna().all():
        st.warning("No pressure values for this activity (fallback mode). Check the Excel mapping or add a synonym.")

    def val_to_rgba(v: float):
        if pd.isna(v):
            return "rgba(180,180,180,0.6)"
        t = float(np.clip(v, 0, 1))
        r = int(round(255 * t)); g = int(round(180 * (1 - t))); b = int(round(120 + 80 * (1 - t)))
        return f"rgba({r},{g},{b},0.9)"

    c1, c2 = st.columns([1, 1])
    with c1:
        bar = go.Figure()
        bar.add_trace(go.Bar(
            x=press_df["Value"],
            y=press_df["Pressure"],
            orientation="h",
            marker=dict(color=[val_to_rgba(v) for v in press_df["Value"]]),
            hovertemplate="%{y}: %{x:.2f}<extra></extra>"
        ))
        bar.update_layout(
            xaxis=dict(range=[0,1]),
            margin=dict(l=10,r=10,t=10,b=10),
            height=520,
            title=f"Pressure profile — {row['Activity']}"
        )
        st.plotly_chart(bar, use_container_width=True)

    with c2:
        st.markdown("**Details**")
        md = pd.DataFrame({
            "Field": ["Activity", "Total area (m²)", "Sectors", "Businesses"],
            "Value": [row["Activity"], f"{row['TotalArea_m2']:.0f}", row.get("Sectors",""), row.get("Businesses","")]
        })
        st.dataframe(md, hide_index=True, use_container_width=True)
        st.markdown("**Raw pressure values**")
        st.dataframe(press_df, hide_index=True, use_container_width=True)


    def val_to_rgba(v: float):
        if pd.isna(v):
            return "rgba(180,180,180,0.6)"
        t = float(np.clip(v, 0, 1))
        r = int(round(255 * t)); g = int(round(180 * (1 - t))); b = int(round(120 + 80 * (1 - t)))
        return f"rgba({r},{g},{b},0.9)"

    c1, c2 = st.columns([1, 1])
    with c1:
        bar = go.Figure()
        bar.add_trace(go.Bar(
            x=press_df["Value"],
            y=press_df["Pressure"],
            orientation="h",
            marker=dict(color=[val_to_rgba(v) for v in press_df["Value"]]),
            hovertemplate="%{y}: %{x:.2f}<extra></extra>"
        ))
        bar.update_layout(
            xaxis=dict(range=[0,1]),
            margin=dict(l=10,r=10,t=10,b=10),
            height=520,
            title=f"Pressure profile — {row['Activity']}"
        )
        st.plotly_chart(bar, use_container_width=True)

    with c2:
        st.markdown("**Details**")
        md = pd.DataFrame({
            "Field": ["Activity", "Total area (m²)", "Sectors", "Businesses"],
            "Value": [row["Activity"], f"{row['TotalArea_m2']:.0f}", row["Sectors"], row["Businesses"]]
        })
        st.dataframe(md, hide_index=True, use_container_width=True)
        st.markdown("**Raw pressure values**")
        st.dataframe(press_df, hide_index=True, use_container_width=True)

# ---------------- RISKS ----------------
def page_risk():
    st.markdown("### Region Map")
    layers = []
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

    center_h3 = cells_df.loc[cells_df["position"] == "CENTER", "h3_index"].iloc[0]
    aq, _ = load_aqueduct_center(center_h3)
    st.markdown("**Water risks — Center (name & value)**")
    if aq.empty:
        st.info("No Aqueduct V4 CSV found for the central grid.")
    else:
        tbl = aq[["dimension", "label"]].rename(columns={"dimension": "Risk", "label": "Value"})
        st.dataframe(tbl, use_container_width=True, hide_index=True)

    _, sens, _ = load_invasive_sensitive(sel_h3, sel_pos)
    st.markdown("**Sensitive species (VU/EN/CR) — Your Grid**")
    st.dataframe(sens.head(500), use_container_width=True, hide_index=True)

# ---------------- Router ----------------
if view == "biodiv":
    page_biodiv()
elif view == "eco":
    page_eco()
elif view == "pressure":
    page_pressure()
elif view == "risk":
    page_risk()
else:
    page_overview()
