from __future__ import annotations
import re
import unicodedata
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
    ENV_RISK_FILE_BY_H3, ENV_RISK_FILE, ENV_TIMESERIES_FILE, POP_FILE,
    AQUEDUCT_FILE, LANDCOVER_GLOB, ECO_FILE, WIND_KML, ACTIVITY_PRESSURE_FILE
)

# Optional fuzzy matcher
try:
    from rapidfuzz import process, fuzz
    _HAS_FUZZ = True
except Exception:
    _HAS_FUZZ = False

# ---------- H3 helpers ----------
def _cell_boundary_latlon(h: str):
    try:
        return h3.cell_to_boundary(h)  # newer API
    except TypeError:
        return h3.cell_to_boundary(h, True)  # older API

def get_cells_df() -> pd.DataFrame:
    if H3_LIST_FILE.exists():
        df = pd.read_csv(H3_LIST_FILE)
        df["h3_index"] = df["h3_index"].astype(str)
        return df[["position", "h3_index"]].copy()
    center = h3.latlng_to_cell(CENTER["lat"], CENTER["lon"], RES)
    neigh = list(h3.grid_ring(center, 1))
    rows = [{"position": "CENTER", "h3_index": center}]
    for i, hh in enumerate(neigh, start=1):
        rows.append({"position": f"N{i}", "h3_index": hh})
    return pd.DataFrame(rows)

def hex_outline_gdf(h3_indices: List[str]) -> gpd.GeoDataFrame:
    recs = []
    for hidx in h3_indices:
        ring = _cell_boundary_latlon(hidx)
        poly = Polygon([(lon, lat) for (lat, lon) in ring])
        recs.append({"h3_index": hidx, "geometry": poly})
    return gpd.GeoDataFrame(recs, crs="EPSG:4326")

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

# ---------- Land cover discovery ----------
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
    return pd.DataFrame(parsed, columns=["file", "h3_index", "position", "year", "tag"])

# ---------- KML windfarm ----------
def load_wind_kml() -> Tuple[gpd.GeoDataFrame, Optional[Path]]:
    p = WIND_KML
    if not p.exists():
        return gpd.GeoDataFrame(geometry=[]), None
    try:
        gdf = gpd.read_file(p)
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
        elif gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(4326)
        gdf = gdf[gdf.geometry.notna()].explode(index_parts=False, ignore_index=True)
        gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
        return gdf, p
    except Exception:
        return gpd.GeoDataFrame(geometry=[]), p

# ---------- Biodiversity ----------
def load_biodiv_metrics() -> pd.DataFrame:
    p = Path(BIODIVERSITY_METRICS)
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    df.rename(columns={
        "H3_Index": "h3_index", "Position": "position",
        "Alpha_Richness": "alpha", "Shannon_H": "shannon",
        "Pielou_E": "pielou", "MSA": "msa",
        "Gamma_Richness": "gamma", "Beta_Whittaker": "beta_whittaker",
    }, inplace=True)
    df["h3_index"] = df["h3_index"].astype(str)
    return df

def load_birds_for_hex(h3_index: str, position: str):
    f = SPECIES_FILE(h3_index, position)
    if not f.exists():
        return pd.DataFrame(columns=["Species", "Status", "Count"]), None
    df = pd.read_csv(f)
    name_col = "CanonicalName" if "CanonicalName" in df.columns else df.columns[0]
    birds = pd.DataFrame()
    for cand in ["Group", "group", "Class", "class", "Order", "order"]:
        if cand in df.columns:
            birds = df[df[cand].astype(str).str.contains("bird|aves", case=False, na=False)]
            if not birds.empty:
                break
    if birds.empty:
        birds = df
    out = birds.rename(columns={name_col: "Species"})
    if "Status" not in out.columns:
        out["Status"] = ""
    if "GBIF_Total_Observations" in out.columns and "Count" not in out.columns:
        out["Count"] = out["GBIF_Total_Observations"]
    if "Count" not in out.columns:
        out["Count"] = 1
    return out[["Species", "Status", "Count"]].sort_values("Count", ascending=False).head(200), f

def load_invasive_sensitive(h3_index: str, position: str):
    f = SPECIES_FILE(h3_index, position)
    inv = pd.DataFrame(columns=["Species"])
    sens = pd.DataFrame(columns=["Species", "Status"])
    if not f.exists():
        return inv, sens, None
    df = pd.read_csv(f)
    name_col = "CanonicalName" if "CanonicalName" in df.columns else df.columns[0]
    for cand in ["Invasive", "Is_Invasive", "is_invasive"]:
        if cand in df.columns:
            inv = df[df[cand] == 1][name_col].to_frame("Species")
            break
    for cand in ["IUCN", "iucn_status", "iucnRedListCategory", "Status"]:
        if cand in df.columns:
            sens = df[df[cand].astype(str).str.upper().isin(["VU", "EN", "CR"])][[name_col, cand]]
            sens.columns = ["Species", "Status"]
            break
    return inv.drop_duplicates(), sens.drop_duplicates(), f

# ---------- Activities (minimal) ----------
def load_land_use(h3_index: str, position: str):
    f = LAND_USE_FILE(h3_index, position)
    if not f.exists():
        return pd.DataFrame(columns=["TagValue", "AreaEnd_m2"]), None
    df = pd.read_csv(f)
    low = {c.lower(): c for c in df.columns}
    tagv = low.get("tagvalue") or low.get("tag_value") or low.get("class") or low.get("landcover") or low.get("land_cover")
    area = low.get("areaend_m2") or low.get("area_m2") or low.get("area")
    out = pd.DataFrame()
    if tagv:
        out["TagValue"] = df[tagv].astype(str)
    else:
        cand = next((c for c in df.columns if df[c].dtype == object), df.columns[0])
        out["TagValue"] = df[cand].astype(str)
    out["AreaEnd_m2"] = pd.to_numeric(df[area], errors="coerce") if area else 0
    return out, f

# ---------- Environmental risks ----------
def load_environmental_risks(h3_index: str, position: str) -> Tuple[pd.DataFrame, Optional[Path]]:
    fh3 = ENV_RISK_FILE_BY_H3(h3_index)
    if fh3.exists():
        return pd.read_csv(fh3), fh3
    fpos = ENV_RISK_FILE(position)
    if fpos.exists():
        return pd.read_csv(fpos), fpos
    return pd.DataFrame(), None

# ---------- Environmental/climate time-series ----------
def load_env_timeseries(position: str) -> Tuple[pd.DataFrame, Optional[Path]]:
    f = ENV_TIMESERIES_FILE(position)
    if not f.exists():
        return pd.DataFrame(), None
    df = pd.read_excel(f, engine="openpyxl") if f.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(f)
    ycol = "year" if "year" in df.columns else df.columns[0]
    df = df.rename(columns={ycol: "year"}).copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"]).copy()
    df["year"] = df["year"].astype(int)
    return df, f

def compute_climate_intensity(df: pd.DataFrame, year_min: int, year_max: int) -> float:
    if df is None or df.empty:
        return float("nan")
    sub = df[(df["year"] >= year_min) & (df["year"] <= year_max)].copy()
    if sub.empty:
        return float("nan")
    numeric_cols = [c for c in sub.columns if c != "year" and pd.api.types.is_numeric_dtype(sub[c])]
    if not numeric_cols:
        return float("nan")
    means = []
    for c in numeric_cols:
        s = pd.to_numeric(sub[c], errors="coerce").dropna()
        if s.empty:
            continue
        z = (s - s.mean()) / (s.std(ddof=0) + 1e-9)
        means.append(z.mean())
    if not means:
        return float("nan")
    return float(np.mean(means))

# ---------- Aqueduct ----------
def parse_label_to_score(label: str) -> float:
    if not isinstance(label, str):
        return np.nan
    lbl = label.strip()
    table = {
        "Medium (0.4-0.6)": 0.5,
        "Low - Medium (0.25-0.50)": 0.375,
        "High (6 in 1,000 to 1 in 100)": (6/1000 + 1/100) / 2,
        "Medium - High (7 in 100,000 to 3 in 10,000)": (7/100000 + 3/10000) / 2,
    }
    if lbl in table:
        return float(table[lbl])
    m = re.search(r"(\d*\.?\d+)\s*[-–]\s*(\d*\.?\d+)", lbl)
    if m:
        return (float(m.group(1)) + float(m.group(2))) / 2.0
    m2 = re.search(r"(\d*\.?\d+)\s*in\s*(\d[\d,]*)\s*to\s*(\d*\.?\d+)\s*in\s*(\d[\d,]*)", lbl, re.I)
    if m2:
        a, b, c, d = float(m2.group(1)), float(m2.group(2).replace(",", "")), float(m2.group(3)), float(m2.group(4).replace(",", ""))
        return (a / b + c / d) / 2.0
    m3 = re.search(r"(\d*\.?\d+)", lbl)
    if m3:
        return float(m3.group(1))
    return np.nan

def load_aqueduct_center(center_h3: str):
    f = AQUEDUCT_FILE(center_h3)
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
    nice = {
        "water_stress_value": "Water stress",
        "water_depletion_label": "Water depletion",
        "riverine_flood_risk_label": "Riverine flood risk",
        "drought_risk_label": "Drought risk",
        "unimproved_no_drinking_water_label": "Unimproved drinking water",
    }
    for col in present:
        lbl = str(sub.iloc[0][col])
        score = parse_label_to_score(lbl)
        rows.append({"dimension": nice.get(col, col), "score": score, "label": lbl})
    return pd.DataFrame(rows), f

# ---------- Eco-Integrity (per-hex) ----------
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

# ---------- FFI across ALL hexes ----------
FFI_PAT = re.compile(r"^avg_ffi_(?P<h3>[0-9a-f]+)_(?P<pos>[A-Za-z0-9]+)\.csv$", re.I)

def load_all_avg_ffi() -> Tuple[pd.DataFrame, List[Path]]:
    rows, paths = [], []
    for f in ROOT.glob("avg_ffi_*_*.csv"):
        m = FFI_PAT.match(f.name)
        if not m:
            continue
        h3i, pos = m.group("h3"), m.group("pos")
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        ycol = "year" if "year" in df.columns else df.columns[0]
        vcol = None
        for c in ["avg_ffi", "fragmentation", "ffi", "value", "val"]:
            if c in df.columns:
                vcol = c; break
        if vcol is None:
            for c in df.columns:
                if c != ycol and pd.api.types.is_numeric_dtype(df[c]):
                    vcol = c; break
        if vcol is None:
            continue
        sub = df[[ycol, vcol]].copy()
        sub.columns = ["year", "value"]
        sub["h3_index"] = str(h3i)
        sub["position"] = str(pos)
        rows.append(sub)
        paths.append(f)
    if not rows:
        return pd.DataFrame(columns=["h3_index", "position", "year", "value"]), []
    out = pd.concat(rows, ignore_index=True)
    out["year"] = pd.to_numeric(out["year"], errors="coerce")
    out = out.dropna(subset=["year"])
    out["year"] = out["year"].astype(int)
    return out[["h3_index", "position", "year", "value"]], paths

# ===== Activities → Pressures (Excel) =====

PRESSURE_COLS = [
    "Area of freshwater use",
    "Area of land use",
    "Area of seabed use",
    "Emissions of GHG",
    "Disturbances (noise, light)",
    "Emissions of non-GHG air pollutants",
    "Emissions of nutrient polluants to water and soil",
    "Emissions of toxic pollutants to water and soil",
    "Generation and release of solid waste",
    "Introduction of invasive species",
    "Other abiotic resource extraction",
    "Other biotic resource extraction",
    "Volume of water use",
]

# Manual synonyms (edit as needed)
SYNONYMS = {
    "quarry": "quarrying",
    "quarry extraction": "quarrying",
    "windfarm": "wind farm",
    "wind-farm": "wind farm",
    "agri": "agriculture",
    "farming": "agriculture",
    "tourist services": "tourism services",
}

def _normalize_str(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))  # strip accents
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s)  # drop punctuation
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _normalize_activity_name(s: str) -> str:
    base = _normalize_str(s)
    return SYNONYMS.get(base, base)

def _fuzzy_match_norm(query_norm: str, choices_norm: list[str], score_cutoff: int = 85) -> str | None:
    if not _HAS_FUZZ or not choices_norm or not query_norm:
        return None
    match = process.extractOne(
        query_norm,
        choices_norm,
        scorer=fuzz.token_sort_ratio,
        score_cutoff=score_cutoff,
    )
    return match[0] if match else None

def load_activity_pressure_matrix_with_businesses() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read Excel with one sheet per sector.
    Returns:
      matrix_df: ['Sector','Activity','Activity_norm'] + PRESSURE_COLS
      business_df: ['Sector','Activity','Activity_norm','Business']
    """
    xls = ACTIVITY_PRESSURE_FILE
    if not xls.exists():
        return (pd.DataFrame(columns=["Sector","Activity","Activity_norm"] + PRESSURE_COLS),
                pd.DataFrame(columns=["Sector","Activity","Activity_norm","Business"]))

    book = pd.read_excel(xls, sheet_name=None, engine="openpyxl")
    matrix_rows, business_rows = [], []

    for sector, df in book.items():
        if df is None or df.empty:
            continue
        df = df.copy()
        # find an activity-like column
        cols_lower = {c.lower(): c for c in df.columns}
        activity_col = None
        for k in ["activity", "name", "tagvalue", "class", "land use", "land_use"]:
            if k in cols_lower:
                activity_col = cols_lower[k]; break
        if activity_col is None:
            obj_cols = [c for c in df.columns if df[c].dtype == object]
            if not obj_cols:
                continue
            activity_col = obj_cols[0]

        present_pressures = [p for p in PRESSURE_COLS if p in df.columns]
        candidate_business_cols = [c for c in df.columns if c not in [activity_col] + PRESSURE_COLS]

        for p in present_pressures:
            df[p] = pd.to_numeric(df[p], errors="coerce")

        for _, row in df.iterrows():
            act_name = str(row[activity_col])
            act_norm = _normalize_activity_name(act_name)
            if not act_norm:
                continue
            rec = {"Sector": sector, "Activity": act_name, "Activity_norm": act_norm}
            for p in PRESSURE_COLS:
                rec[p] = row.get(p, np.nan)
            matrix_rows.append(rec)

            for bc in candidate_business_cols:
                val = row.get(bc)
                is_on = False
                if isinstance(val, (int, float)) and not pd.isna(val):
                    is_on = float(val) != 0.0
                elif isinstance(val, str):
                    is_on = val.strip() not in ("", "0", "nan", "NaN")
                elif isinstance(val, bool):
                    is_on = val
                if is_on:
                    business_rows.append({
                        "Sector": sector,
                        "Activity": act_name,
                        "Activity_norm": act_norm,
                        "Business": str(bc)
                    })

    matrix_df = pd.DataFrame(matrix_rows)
    business_df = pd.DataFrame(business_rows).drop_duplicates() if business_rows else pd.DataFrame(
        columns=["Sector","Activity","Activity_norm","Business"]
    )
    return matrix_df, business_df

def filter_land_use_to_activities(landuse_df: pd.DataFrame, matrix_df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only land-use rows whose TagValue matches an 'Activity' in the matrix.
    Matching order:
      1) exact normalized match
      2) fuzzy match (if rapidfuzz installed; cutoff=85)
      3) synonyms applied in normalization
    """
    if landuse_df is None or landuse_df.empty or matrix_df is None or matrix_df.empty:
        return pd.DataFrame(columns=["TagValue","AreaEnd_m2","Sector","Activity","Activity_norm"] + PRESSURE_COLS)

    df = landuse_df.copy()
    df["TagValue"] = df["TagValue"].astype(str)

    m = matrix_df.copy()
    if "Activity_norm" not in m.columns:
        m["Activity_norm"] = m["Activity"].map(_normalize_activity_name)
    valid_norms = m["Activity_norm"].dropna().unique().tolist()

    matched_norms = []
    for raw in df["TagValue"].astype(str).tolist():
        q = _normalize_activity_name(raw)
        if q in valid_norms:
            matched_norms.append(q)
        else:
            best = _fuzzy_match_norm(q, valid_norms, score_cutoff=85)
            matched_norms.append(best if best else "")

    df["Activity_norm"] = matched_norms
    df = df[df["Activity_norm"] != ""].copy()

    merged = df.merge(
        m[["Sector","Activity","Activity_norm"] + PRESSURE_COLS].drop_duplicates(),
        how="inner",
        on="Activity_norm"
    )
    keep = ["TagValue","AreaEnd_m2","Sector","Activity","Activity_norm"] + PRESSURE_COLS
    return merged[keep].copy()

def aggregate_activities_all_grids(cells_df: pd.DataFrame) -> pd.DataFrame:
    """
    Load land-use for all grids, keep only activities (via matrix),
    group by Activity across ALL grids (sum AreaEnd_m2), attach Sectors & Businesses,
    and keep one set of pressure values per Activity.
    """
    matrix_df, business_df = load_activity_pressure_matrix_with_businesses()
    if matrix_df.empty:
        return pd.DataFrame(columns=["Activity","TotalArea_m2","Sectors","Businesses"] + PRESSURE_COLS)

    rows = []
    for _, r in cells_df.iterrows():
        h3i, pos = r["h3_index"], r["position"]
        lu_df, _ = load_land_use(h3i, pos)
        if lu_df.empty:
            continue
        filt = filter_land_use_to_activities(lu_df, matrix_df)
        if not filt.empty:
            rows.append(filt)

    if not rows:
        return pd.DataFrame(columns=["Activity","TotalArea_m2","Sectors","Businesses"] + PRESSURE_COLS)

    all_lu = pd.concat(rows, ignore_index=True)

    agg_area = (all_lu.groupby(["Activity_norm","TagValue"])
                      .agg(TotalArea_m2=("AreaEnd_m2","sum"))
                      .reset_index())

    pressures = (matrix_df.sort_values("Sector")
                 .drop_duplicates(subset=["Activity_norm"])
                 [["Activity_norm","Activity"] + PRESSURE_COLS])

    out = agg_area.merge(pressures, how="left", on="Activity_norm")

    sectors = (all_lu.groupby("Activity_norm")["Sector"]
               .agg(lambda s: sorted(set(map(str, s))))
               .reset_index().rename(columns={"Sector":"Sectors"}))
    out = out.merge(sectors, how="left", on="Activity_norm")

    if not business_df.empty:
        biz = (business_df.groupby("Activity_norm")["Business"]
               .agg(lambda s: sorted(set(map(str, s))))
               .reset_index().rename(columns={"Business":"Businesses"}))
        out = out.merge(biz, how="left", on="Activity_norm")
    else:
        out["Businesses"] = [[] for _ in range(len(out))]

    out["Activity"] = out["TagValue"]
    out.drop(columns=["TagValue"], inplace=True)
    out["Sectors"] = out["Sectors"].apply(lambda x: ", ".join(x) if isinstance(x, list) else "")
    out["Businesses"] = out["Businesses"].apply(lambda x: ", ".join(x) if isinstance(x, list) else "")
    out = out[["Activity","TotalArea_m2","Sectors","Businesses","Activity_norm"] + PRESSURE_COLS]
    out = out.sort_values(["TotalArea_m2","Activity"], ascending=[False, True]).reset_index(drop=True)
    return out

def get_activity_row(all_activities_df: pd.DataFrame, activity_name: str) -> pd.Series | None:
    if all_activities_df is None or all_activities_df.empty:
        return None
    norm = _normalize_activity_name(activity_name)
    m = all_activities_df[all_activities_df["Activity_norm"] == norm]
    if len(m): return m.iloc[0]
    m = all_activities_df[all_activities_df["Activity"] == activity_name]
    return m.iloc[0] if len(m) else None
