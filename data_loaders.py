from __future__ import annotations
import re
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import h3

from data_config import (
    DATA_ROOT, LANDCOVER_DIR, BIODIVERSITY_METRICS, SPECIES_DIR, LAND_USE_DIR,
    ENV_RISK_DIR, AQUEDUCT_DIR, ECO_FILES, RES, CENTER, GIT_BASE_URL
)

# ---------- H3 helpers ----------
def get_cells_df() -> pd.DataFrame:
    """Return DataFrame with ['position','h3_index'] for center+6 neighbors."""
    csv_path = DATA_ROOT / "h3_res7_center_plus_neighbors_minimal.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df["h3_index"] = df["h3_index"].astype(str)
        return df[["position","h3_index"]].copy()

    center = h3.latlng_to_cell(CENTER["lat"], CENTER["lon"], RES)
    neigh = list(h3.grid_ring(center, 1))
    rows = [{"position":"CENTER","h3_index":center}]
    for i, h in enumerate(neigh, start=1):
        rows.append({"position":f"N{i}", "h3_index":h})
    return pd.DataFrame(rows)

def h3_polygon_lonlat(h: str):
    ring = h3.cell_to_boundary(h, True)  # [(lat,lon),...]
    return [[lon, lat] for (lat, lon) in ring] + [[ring[0][1], ring[0][0]]]

def hex_outline_gdf(h3_indices: List[str]) -> gpd.GeoDataFrame:
    recs = []
    for hidx in h3_indices:
        ring = h3.cell_to_boundary(hidx)
        poly = Polygon([(lon, lat) for (lat, lon) in ring])
        recs.append({"h3_index":hidx, "geometry":poly})
    return gpd.GeoDataFrame(recs, crs="EPSG:4326")

# ---------- Land-cover discovery ----------
import os
LC_PAT = re.compile(r"^(?P<h3>[0-9a-f]+)_(?P<pos>[A-Za-z0-9_]+)_(?P<year>\d{4})_(?P<tag>glc_fcs30d|landcover)\.geojson$", re.I)

def discover_landcover() -> pd.DataFrame:
    if not LANDCOVER_DIR.exists():
        return pd.DataFrame(columns=["file","h3_index","position","year","tag"])
    parsed = []
    for f in LANDCOVER_DIR.glob("*.geojson"):
        m = LC_PAT.match(f.name)
        if m:
            parsed.append(dict(
                file=f,
                h3_index=m.group("h3"),
                position=m.group("pos"),
                year=int(m.group("year")),
                tag=m.group("tag").lower(),
            ))
    return pd.DataFrame(parsed)

# ---------- Git URL helper ----------
def to_git_url(local_path: Path) -> Optional[str]:
    if not GIT_BASE_URL:
        return None
    base = GIT_BASE_URL.rstrip("/")
    try:
        rel = local_path.relative_to(Path("."))
    except Exception:
        rel = local_path
    return f"{base}/{rel.as_posix()}"

# ---------- Biodiversity ----------
def load_biodiv_metrics() -> pd.DataFrame:
    p = Path(BIODIVERSITY_METRICS)
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    df.rename(columns={
        "H3_Index":"h3_index","Position":"position",
        "Alpha_Richness":"alpha","Shannon_H":"shannon",
        "Pielou_E":"pielou","MSA":"msa",
        "Gamma_Richness":"gamma","Beta_Whittaker":"beta_whittaker",
    }, inplace=True)
    df["h3_index"] = df["h3_index"].astype(str)
    return df

def load_birds_for_hex(h3_index: str, position: str):
    f = SPECIES_DIR / f"species_pool_{h3_index}_{position}.csv"
    if not f.exists():
        return pd.DataFrame(columns=["Species","Status","Count"]), None
    df = pd.read_csv(f)
    # Heuristics
    name_col = "CanonicalName" if "CanonicalName" in df.columns else df.columns[0]
    # Identify birds
    birds = pd.DataFrame()
    for cand in ["Group","group","Class","class","Order","order"]:
        if cand in df.columns:
            birds = df[df[cand].astype(str).str.contains("bird|aves", case=False, na=False)]
            if not birds.empty: break
    if birds.empty:
        birds = df
    out = birds.rename(columns={name_col:"Species"})
    if "Status" not in out.columns: out["Status"] = ""
    if "GBIF_Total_Observations" in out.columns and "Count" not in out.columns:
        out["Count"] = out["GBIF_Total_Observations"]
    if "Count" not in out.columns: out["Count"] = 1
    return out[["Species","Status","Count"]].sort_values("Count", ascending=False).head(200), f

def load_invasive_sensitive(h3_index: str, position: str):
    f = SPECIES_DIR / f"species_pool_{h3_index}_{position}.csv"
    inv = pd.DataFrame(columns=["Species"])
    sens = pd.DataFrame(columns=["Species","Status"])
    if not f.exists():
        return inv, sens, None
    df = pd.read_csv(f)
    name_col = "CanonicalName" if "CanonicalName" in df.columns else df.columns[0]
    # Invasive flags
    for cand in ["Invasive","Is_Invasive","is_invasive"]:
        if cand in df.columns:
            inv = df[df[cand]==1][name_col].to_frame("Species")
            break
    # Sensitive categories (VU/EN/CR)
    for cand in ["IUCN","iucn_status","iucnRedListCategory","Status"]:
        if cand in df.columns:
            sens = df[df[cand].astype(str).str.upper().isin(["VU","EN","CR"])][[name_col, cand]]
            sens.columns = ["Species","Status"]
            break
    return inv, sens, f

# ---------- Activities ----------
def load_land_use(h3_index: str, position: str):
    f = LAND_USE_DIR / f"Land_use_{h3_index}_{position}.csv"
    if not f.exists():
        return pd.DataFrame(columns=["Activity","Impact"]), None
    df = pd.read_csv(f)
    # Normalize columns
    low = {c.lower(): c for c in df.columns}
    a = low.get("activity") or low.get("land_use") or list(df.columns)[0]
    imp = low.get("impact") or low.get("intensity")
    if a != "Activity": df.rename(columns={a:"Activity"}, inplace=True)
    if imp and imp != "Impact": df.rename(columns={imp:"Impact"}, inplace=True)
    if "Impact" not in df.columns: df["Impact"] = ""
    return df[["Activity","Impact"]], f

# ---------- Environmental risks ----------
def load_environmental_risks(position: str):
    f = ENV_RISK_DIR / f"environmental_risks_{position}.csv"
    if not f.exists():
        return pd.DataFrame(), None
    return pd.read_csv(f), f

# ---------- Aqueduct ----------
def parse_label_to_score(label: str) -> float:
    if not isinstance(label, str):
        return np.nan
    lbl = label.strip()
    table = {
        "Medium (0.4-0.6)": 0.5,
        "Low - Medium (0.25-0.50)": 0.375,
        "High (6 in 1,000 to 1 in 100)": (6/1000 + 1/100)/2,
        "Medium - High (7 in 100,000 to 3 in 10,000)": (7/100000 + 3/10000)/2,
    }
    if lbl in table:
        return float(table[lbl])
    import re
    m = re.search(r"(\d*\.?\d+)\s*[-–]\s*(\d*\.?\d+)", lbl)
    if m:
        return (float(m.group(1)) + float(m.group(2))) / 2.0
    m2 = re.search(r"(\d*\.?\d+)\s*in\s*(\d[\d,]*)\s*to\s*(\d*\.?\d+)\s*in\s*(\d[\d,]*)", lbl, re.I)
    if m2:
        a, b, c, d = float(m2.group(1)), float(m2.group(2).replace(",","")), float(m2.group(3)), float(m2.group(4).replace(",",""))
        return (a/b + c/d) / 2.0
    m3 = re.search(r"(\d*\.?\d+)", lbl)
    if m3:
        return float(m3.group(1))
    return np.nan

def load_aqueduct_center(center_h3: str):
    f = AQUEDUCT_DIR / f"aqueduct_v4_{center_h3}.csv"
    if not f.exists():
        return pd.DataFrame(), None
    df = pd.read_csv(f)
    keep = [
        "water_stress_value",
        "water_depletion_label",
        "riverine_flood_risk_label",
        "drought_risk_label",
        "unimproved_no_drinking_water_label",
    ]
    present = [c for c in keep if c in df.columns]
    sub = df[present].iloc[[0]].copy()
    rows = []
    for col in present:
        lbl = str(sub.iloc[0][col])
        score = parse_label_to_score(lbl)
        rows.append({"dimension": col, "score": score, "label": lbl})
    return pd.DataFrame(rows), f

# ---------- Eco-Integrity ----------
def _eco_file(pattern_dir: Path, stem: str):
    cands = list(pattern_dir.glob(f"{stem}*.csv"))
    return cands[0] if cands else None

def load_eco_for_hex(h3_index: str, position: str):
    stem_hi = f"high_integrity_{h3_index}_{position}_2022"
    stem_rd = f"rapid_decline_{h3_index}_{position}_2005_2022"
    stem_cor = f"corridors_{h3_index}_{position}_2022"
    stem_ffi = f"avg_ffi_{h3_index}_{position}"

    hi = _eco_file(ECO_FILES["high_integrity"], stem_hi)
    rd = _eco_file(ECO_FILES["rapid_decline"], stem_rd)
    cor = _eco_file(ECO_FILES["corridors"], stem_cor)
    ffi = _eco_file(ECO_FILES["avg_ffi"], stem_ffi)

    def read_csv(p: Path|None):
        if p and p.exists():
            try: return pd.read_csv(p)
            except Exception: return None
        return None

    return dict(
        high_integrity=(read_csv(hi), hi),
        rapid_decline=(read_csv(rd), rd),
        corridors=(read_csv(cor), cor),
        avg_ffi=(read_csv(ffi), ffi),
    )
