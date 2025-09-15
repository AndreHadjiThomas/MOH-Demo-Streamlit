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
    ROOT, GIT_BASE_URL, RES, CENTER, H3_LIST_FILE,
    BIODIVERSITY_METRICS, SPECIES_FILE, LAND_USE_FILE,
    ENV_RISK_FILE_BY_H3, ENV_RISK_FILE, AQUEDUCT_FILE,
    LANDCOVER_GLOB, ECO_FILE
)

# ---------------- H3 helpers (version-agnostic) ----------------
def _cell_boundary_latlon(h: str):
    try:
        return h3.cell_to_boundary(h)               # new API
    except TypeError:
        return h3.cell_to_boundary(h, True)         # older API

def get_cells_df() -> pd.DataFrame:
    if H3_LIST_FILE.exists():
        df = pd.read_csv(H3_LIST_FILE)
        df["h3_index"] = df["h3_index"].astype(str)
        return df[["position","h3_index"]].copy()
    center = h3.latlng_to_cell(CENTER["lat"], CENTER["lon"], RES)
    neigh = list(h3.grid_ring(center, 1))
    rows = [{"position":"CENTER","h3_index":center}]
    for i, hh in enumerate(neigh, start=1):
        rows.append({"position":f"N{i}","h3_index":hh})
    return pd.DataFrame(rows)

def hex_outline_gdf(h3_indices: List[str]) -> gpd.GeoDataFrame:
    recs = []
    for hidx in h3_indices:
        ring = _cell_boundary_latlon(hidx)
        poly = Polygon([(lon, lat) for (lat, lon) in ring])
        recs.append({"h3_index": hidx, "geometry": poly})
    return gpd.GeoDataFrame(recs, crs="EPSG:4326")

# ---------------- Git URL helper ----------------
def to_git_url(local_path: Path) -> Optional[str]:
    if not GIT_BASE_URL:
        return None
    base = GIT_BASE_URL.rstrip("/")
    try:
        rel = local_path.relative_to(Path("."))
    except Exception:
        rel = local_path
    return f"{base}/{rel.as_posix()}"

# ---------------- Land cover discovery ----------------
LC_PAT = re.compile(r"^(?P<h3>[0-9a-f]+)_(?P<pos>[A-Za-z0-9_]+)_(?P<year>\d{4})_(?P<tag>glc_fcs30d|landcover)\.geojson$", re.I)

def discover_landcover() -> pd.DataFrame:
    parsed = []
    for f in ROOT.glob(LANDCOVER_GLOB):
        m = LC_PAT.match(f.name)
        if m:
            parsed.append(dict(
                file=f,
                h3_index=m.group("h3"),
                position=m.group("pos"),
                year=int(m.group("year")),
                tag=m.group("tag").lower(),
            ))
    return pd.DataFrame(parsed, columns=["file","h3_index","position","year","tag"])

# ---------------- Biodiversity ----------------
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
    f = SPECIES_FILE(h3_index, position)
    if not f.exists():
        return pd.DataFrame(columns=["Species","Status","Count"]), None
    df = pd.read_csv(f)
    name_col = "CanonicalName" if "CanonicalName" in df.columns else df.columns[0]
    birds = pd.DataFrame()
    for cand in ["Group","group","Class","class","Order","order"]:
        if cand in df.columns:
            birds = df[df[cand].astype(str).str.contains("bird|aves", case=False, na=False)]
            if not birds.empty: break
    if birds.empty: birds = df
    out = birds.rename(columns={name_col:"Species"})
    if "Status" not in out.columns: out["Status"] = ""
    if "GBIF_Total_Observations" in out.columns and "Count" not in out.columns:
        out["Count"] = out["GBIF_Total_Observations"]
    if "Count" not in out.columns: out["Count"] = 1
    return out[["Species","Status","Count"]].sort_values("Count", ascending=False).head(200), f

def load_invasive_sensitive(h3_index: str, position: str):
    f = SPECIES_FILE(h3_index, position)
    inv = pd.DataFrame(columns=["Species"])
    sens = pd.DataFrame(columns=["Species","Status"])
    if not f.exists():
        return inv, sens, None
    df = pd.read_csv(f)
    name_col = "CanonicalName" if "CanonicalName" in df.columns else df.columns[0]
    for cand in ["Invasive","Is_Invasive","is_invasive"]:
        if cand in df.columns:
            inv = df[df[cand]==1][name_col].to_frame("Species")
            break
    for cand in ["IUCN","iucn_status","iucnRedListCategory","Status"]:
        if cand in df.columns:
            sens = df[df[cand].astype(str).str.upper().isin(["VU","EN","CR"])][[name_col, cand]]
            sens.columns = ["Species","Status"]
            break
    return inv, sens, f

# ---------------- Activities ----------------
def load_land_use(h3_index: str, position: str):
    f = LAND_USE_FILE(h3_index, position)
    if not f.exists():
        return pd.DataFrame(columns=["Activity","Impact"]), None
    df = pd.read_csv(f)
    low = {c.lower():c for c in df.columns}
    a = low.get("activity") or low.get("land_use") or list(df.columns)[0]
    imp = low.get("impact") or low.get("intensity")
    if a != "Activity": df.rename(columns={a:"Activity"}, inplace=True)
    if imp and imp != "Impact": df.rename(columns={imp:"Impact"}, inplace=True)
    if "Impact" not in df.columns: df["Impact"] = ""
    return df[["Activity","Impact"]], f

# ---------------- Environmental risks (h3-first) ----------------
def load_environmental_risks(h3_index: str, position: str) -> Tuple[pd.DataFrame, Optional[Path]]:
    fh3 = ENV_RISK_FILE_BY_H3(h3_index)
    if fh3.exists():
        return pd.read_csv(fh3), fh3
    fpos = ENV_RISK_FILE(position)
    if fpos.exists():
        return pd.read_csv(fpos), fpos
    return pd.DataFrame(), None

# ---------------- Aqueduct ----------------
def parse_label_to_score(label: str) -> float:
    if not isinstance(label, str): return np.nan
    lbl = label.strip()
    table = {
        "Medium (0.4-0.6)": 0.5,
        "Low - Medium (0.25-0.50)": 0.375,
        "High (6 in 1,000 to 1 in 100)": (6/1000 + 1/100)/2,
        "Medium - High (7 in 100,000 to 3 in 10,000)": (7/100000 + 3/10000)/2,
    }
    if lbl in table: return float(table[lbl])
    m = re.search(r"(\d*\.?\d+)\s*[-–]\s*(\d*\.?\d+)", lbl)
    if m: return (float(m.group(1)) + float(m.group(2))) / 2.0
    m2 = re.search(r"(\d*\.?\d+)\s*in\s*(\d[\d,]*)\s*to\s*(\d*\.?\d+)\s*in\s*(\d[\d,]*)", lbl, re.I)
    if m2:
        a, b, c, d = float(m2.group(1)), float(m2.group(2).replace(",","")), float(m2.group(3)), float(m2.group(4).replace(",",""))
        return (a/b + c/d) / 2.0
    m3 = re.search(r"(\d*\.?\d+)", lbl)
    if m3: return float(m3.group(1))
    return np.nan

def load_aqueduct_center(center_h3: str):
    f = AQUEDUCT_FILE(center_h3)
    if not f.exists(): return pd.DataFrame(), None
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

# ---------------- Eco-Integrity: existing per-hex loader ----------------
def load_eco_for_hex(h3_index: str, position: str):
    stems = dict(
        high_integrity=f"high_integrity_{h3_index}_{position}_2022",
        rapid_decline=f"rapid_decline_{h3_index}_{position}_2005_2022",
        corridors=f"corridors_{h3_index}_{position}_2022",
        avg_ffi=f"avg_ffi_{h3_index}_{position}",
    )
    res = {}
    for key, stem in stems.items():
        p = ECO_FILE(stem)
        if p.exists():
            try:
                res[key] = (pd.read_csv(p), p)
            except Exception:
                res[key] = (None, p)
        else:
            res[key] = (None, None)
    return res

# ---------------- ✅ NEW: FFI across ALL hexes ----------------
FFI_PAT = re.compile(r"^avg_ffi_(?P<h3>[0-9a-f]+)_(?P<pos>[A-Za-z0-9]+)\.csv$", re.I)

def load_all_avg_ffi() -> Tuple[pd.DataFrame, List[Path]]:
    """
    Reads every file like: avg_ffi_<h3>_<pos>.csv
    Tries to detect (year, value) columns robustly.
    Returns a long DF: [h3_index, position, year, value] and list of file Paths.
    """
    rows, paths = [], []
    for f in ROOT.glob("avg_ffi_*_*.csv"):
        m = FFI_PAT.match(f.name)
        if not m: continue
        h3i, pos = m.group("h3"), m.group("pos")
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        # pick year/value columns
        ycol = "year" if "year" in df.columns else None
        if ycol is None:
            # guess a column that looks like years
            for c in df.columns:
                if pd.api.types.is_numeric_dtype(df[c]) and df[c].between(1900, 2100, inclusive="left").sum() >= max(3, len(df)//4):
                    ycol = c; break
        if ycol is None:  # fallback: first column
            ycol = df.columns[0]
        vcol = None
        for c in ["avg_ffi","fragmentation","ffi","value","val"]:
            if c in df.columns: vcol = c; break
        if vcol is None:
            # pick the first numeric column different from ycol
            for c in df.columns:
                if c != ycol and pd.api.types.is_numeric_dtype(df[c]):
                    vcol = c; break
        if vcol is None:
            continue
        sub = df[[ycol, vcol]].copy()
        sub.columns = ["year","value"]
        sub["h3_index"] = str(h3i)
        sub["position"] = str(pos)
        rows.append(sub)
        paths.append(f)
    if not rows:
        return pd.DataFrame(columns=["h3_index","position","year","value"]), []
    out = pd.concat(rows, ignore_index=True)
    # ensure numeric year for filtering
    out["year"] = pd.to_numeric(out["year"], errors="coerce")
    out = out.dropna(subset=["year"])
    out["year"] = out["year"].astype(int)
    return out[["h3_index","position","year","value"]], paths


