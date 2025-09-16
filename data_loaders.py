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
from pathlib import Path as _Path
import pandas as _pd
import numpy as _np
import unicodedata as _unicodedata, re as _re
from pathlib import Path
from typing import Optional
from data_config import ACTIVITY_PRESSURE_FILE as _ACT_FILE
import unicodedata, re  # make sure these are imported

from data_config import (
    ROOT, GIT_BASE_URL, RES, CENTER, H3_LIST_FILE,
    BIODIVERSITY_METRICS, SPECIES_FILE, LAND_USE_FILE,
    ENV_RISK_FILE_BY_H3, ENV_RISK_FILE, ENV_TIMESERIES_FILE, POP_FILE,
    AQUEDUCT_FILE, LANDCOVER_GLOB, ECO_FILE, WIND_KML
)

# fuzzy matcher
try:
    from rapidfuzz import process as _rf_process, fuzz as _rf_fuzz
    _HAS_FUZZ = True
except Exception:
    _HAS_FUZZ = False


def _norm_txt(s: str) -> str:
    """Loose, accent/punct-insensitive normalizer for tag keys/values & activities."""
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

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
        # fallback name variants
        alt = list(ROOT.glob(f"Land_use_{h3_index}_{position}.*")) + \
              list(ROOT.glob(f"land_use_{h3_index}_{position}.*")) + \
              list(ROOT.glob(f"landuse_{h3_index}_{position}.*"))
        if alt:
            f = alt[0]
        else:
            return pd.DataFrame(columns=["TagKey","TagValue","AreaEnd_m2"]), None

    df = pd.read_csv(f)

    low = {c.lower(): c for c in df.columns}
    # Prefer explicit TagKey/TagValue if present
    tagkey_col = low.get("tagkey") or low.get("key")
    tagval_col = (low.get("tagvalue") or low.get("tag_value") or low.get("class") or
                  low.get("landcover") or low.get("land_cover") or low.get("activity") or
                  next((c for c in df.columns if df[c].dtype == object), df.columns[0]))

    # Area selection with unit normalization
    area_m2 = (low.get("areaend_m2") or low.get("area_m2") or low.get("area") or
               low.get("areaend_m") or low.get("aream2"))
    area_ha = low.get("areaend_ha") or low.get("area_ha")
    area_km2 = low.get("area_km2") or low.get("area_km²")

    out = pd.DataFrame()
    out["TagValue"] = df[tagval_col].astype(str)

    if tagkey_col:
        out["TagKey"] = df[tagkey_col].astype(str)
    else:
        # If no TagKey, try a heuristic: if a 'key' column exists or infer natural/landuse by common names
        out["TagKey"] = ""

    if area_m2:
        out["AreaEnd_m2"] = pd.to_numeric(df[area_m2], errors="coerce").fillna(0)
    elif area_ha:
        out["AreaEnd_m2"] = pd.to_numeric(df[area_ha], errors="coerce").fillna(0) * 10_000.0
    elif area_km2:
        out["AreaEnd_m2"] = pd.to_numeric(df[area_km2], errors="coerce").fillna(0) * 1_000_000.0
    else:
        guess = next((c for c in df.columns if "area" in c.lower() and pd.api.types.is_numeric_dtype(df[c])), None)
        out["AreaEnd_m2"] = pd.to_numeric(df[guess], errors="coerce").fillna(0) if guess else 0.0

    # -------- natural filter --------
    key_norm = out["TagKey"].map(_norm_txt)
    val_norm = out["TagValue"].map(_norm_txt)
    mask_natural = (key_norm == "natural") & (val_norm.isin(NATURAL_EXCLUDE))
    out = out.loc[~mask_natural].copy()

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
        "water_stress_label",
        "water_depletion_label",
        "riverine_flood_risk_label",
        "drought_risk_label",
        "unimproved_no_drinking_water_label",
    ]
    present = [c for c in keep if c in df.columns]
    sub = df[present].iloc[[0]].copy()
    rows = []
    nice = {
        "water_stress_label": "Water stress",
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

# 13 pressure dimensions (exact labels)
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


# -------------------------------------------------------
# Extended synonyms (OSM-ish) — activities + natural terms
# -------------------------------------------------------
SYNONYMS = {
    # --- Buildings / real estate ---
    "residential": "Residential buildings",
    "residential buildings": "Residential buildings",
    "housing": "Residential buildings",
    "commercial": "Commercial Buildings",
    "commercial buildings": "Commercial Buildings",
    "mall": "Commercial Buildings",
    "shopping": "Commercial Buildings",
    "infrastructure": "Infrastructure",
    "roads": "Infrastructure",
    "bridge": "Infrastructure",
    "cement": "Cement & Concrete Production",
    "concrete": "Cement & Concrete Production",
    "cement_plant": "Cement & Concrete Production",
    "real estate": "Real Estate Development & Management",
    "real estate development": "Real Estate Development & Management",
    "real estate management": "Real Estate Development & Management",
    # Land / buildings / commerce
    "residential": "Residential buildings",
    "residential buildings": "Residential buildings",
    "commercial": "Commercial Buildings",
    "commercial buildings": "Commercial Buildings",
    "square": "Infrastructure",        # or "Commercial Buildings" if that’s how you treat squares

    # Agriculture & related
    "farmland": "Crop Farming",
    "landuse=farmland": "Crop Farming",
    "farming": "Crop Farming",
    "agriculture": "Crop Farming",
    "meadow": "Livestock",       
    "farmyard": "Crop Farming",    

    # --- Energy ---
    "oil": "Oil & Gas",
    "gas": "Oil & Gas",
    "oil & gas": "Oil & Gas",
    "coal": "Coal Mining & Power",
    "coal mining": "Coal Mining & Power",
    "coal power": "Coal Mining & Power",
    "windfarm": "Wind Energy",
    "wind-farm": "Wind Energy",
    "wind energy": "Wind Energy",
    "solar": "Solar Energy",
    "solar farm": "Solar Energy",
    "hydro": "Hydro Energy",
    "hydropower": "Hydro Energy",
    "dam": "Hydro Energy",
    "biomass": "Biomass Energy",
    "biogas": "Biomass Energy",
    "water supply": "Water Supply",
    "electricity": "Electricity Grids",
    "electricity grid": "Electricity Grids",
    "grid": "Electricity Grids",

    # --- Finance ---
    "bank": "Banks",
    "banks": "Banks",
    "asset manager": "Asset Managers",
    "asset managers": "Asset Managers",
    "insurer": "Insurers",
    "insurance": "Insurers",

    # --- Food & agriculture ---
    "crop": "Crop Farming",
    "farming": "Crop Farming",
    "agriculture": "Crop Farming",
    "livestock": "Livestock",
    "cattle": "Livestock",
    "poultry": "Livestock",
    "aquaculture": "Aquaculture & Fisheries",
    "fisheries": "Aquaculture & Fisheries",
    "fish farm": "Aquaculture & Fisheries",
    "food processing": "Food Processing",
    "slaughterhouse": "Food Processing",
    "food packaging": "Food Packaging",
    "packaging": "Food Packaging",
    "supermarket": "Supermarkets",
    "supermarkets": "Supermarkets",
    "food delivery": "Food Delivery",
    "delivery": "Food Delivery",

    # --- Forestry ---
    "logging": "Logging & Timber Products",
    "timber": "Logging & Timber Products",
    "pulp": "Pulp & Paper Production",
    "paper": "Pulp & Paper Production",
    "forest": "Forest Management & Plantations",
    "forestry": "Forest Management & Plantations",
    "plantation": "Forest Management & Plantations",

    # --- Health / pharma ---
    "drug": "Drug Manufacturing",
    "pharma": "Drug Manufacturing",
    "medical devices": "Medical Devices",
    "hospital": "Hospitals & Clinics",
    "clinic": "Hospitals & Clinics",
    "veterinary": "Veterinary medicine",
    "diagnostic": "Diagnostics & laboratories",
    "laboratory": "Diagnostics & laboratories",
    "distribution": "Distribution & wholesale of medicines",
    "pharmacy": "Pharmacies & retail",
    "pharmacies": "Pharmacies & retail",
    "biopharma": "R&D in biopharma",

    # --- Industry / manufacturing ---
    "chemical": "Chemicals",
    "chemicals": "Chemicals",
    "petrochemical": "Petrochemicals",
    "petrochemicals": "Petrochemicals",
    "metals": "Metals",
    "metal": "Metals",
    "mining": "Mining",
    "textile": "Textiles",
    "apparel": "Apparel",
    "clothing": "Apparel",
    "automotive": "Automotive & Machinery",
    "machinery": "Automotive & Machinery",

    # --- ICT ---
    "software": "Software Development",
    "data": "Data Science",
    "cyber": "Cybersecurity",
    "cloud": "Cloud Computing",
    "ai": "Artificial Intelligence",
    "artificial intelligence": "Artificial Intelligence",
    "network": "Network Infrastructure",
    "wireless": "Wireless Communications",
    "telecom": "Telecom Engineering",
    "collaboration": "Collaboration Tools",

    # --- Transport ---
    "airline": "Airlines",
    "cruise": "Cruise Lines",
    "rail": "Rail Services",
    "car rental": "Car Rentals",
    "bus": "Bus Services",
    "private vehicle": "Private Vehicles",
    "micromobility": "Micromobility",
    "rideshare": "Ridesharing",
    "ridesharing": "Ridesharing",
    "car leasing": "Car Leasing",
    "aviation alternative": "Aviation Alternatives",
    "navigation": "Navigation Apps",
    "travel": "Travel Planning Tools",
    "fuel": "Fuel/Charging Networks",
    "charging": "Fuel/Charging Networks",
    "insurance app": "Insurance Apps",

    # --- Logistics ---
    "shipping": "Shipping & Ports",
    "port": "Shipping & Ports",
    "ports": "Shipping & Ports",
    "trucking": "Trucking & Freight",
    "freight": "Trucking & Freight",
    "warehousing": "Warehousing & Distribution",
    "distribution": "Warehousing & Distribution",

    # --- Health & wellness ---
    "gym": "Gyms & Fitness Clubs",
    "fitness": "Gyms & Fitness Clubs",
    "yoga": "Yoga & Pilates Studios",
    "pilates": "Yoga & Pilates Studios",
    "spa": "Spas, retreats, thermal baths",
    "retreat": "Spas, retreats, thermal baths",
    "thermal": "Spas, retreats, thermal baths",
    "holistic": "Holistic Health",
    "personal care": "Personal Care Products (Supplements)",
    "supplement": "Personal Care Products (Supplements)",
    "meditation": "Meditation & Mindfulness apps",
    "mindfulness": "Meditation & Mindfulness apps",
}


NATURAL_EXCLUDE = {
    "wood","nature_reserve","bare_rock","scrub",
    "plateau","heath","orchard","ridge","water","valley",
    "grass","cliff","pitch","wetland",
}

def _norm(s: str) -> str:
    if not isinstance(s, str): return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _find_activity_excel(default_path: Path) -> Path | None:
    if default_path and default_path.exists():
        return default_path
    # try common variants in repo root
    cands = list(Path(".").glob("*.xlsx"))
    pref = [p for p in cands if "biomet" in p.name.lower()]
    return (pref[0] if pref else (cands[0] if cands else None))
    
def _norm_str(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = _unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not _unicodedata.combining(ch))
    s = s.lower()
    s = _re.sub(r"[^\w\s]", " ", s)
    s = _re.sub(r"\s+", " ", s).strip()
    return s

def _normalize_activity_name(s: str) -> str:
    base = _norm_str(s)
    return SYNONYMS.get(base, base)

def _fuzzy_match_norm(q_norm: str, choices_norm: list[str], cutoff: int = 85) -> str | None:
    if not _HAS_FUZZ or not q_norm or not choices_norm:
        return None
    hit = _rf_process.extractOne(q_norm, choices_norm, scorer=_rf_fuzz.token_sort_ratio, score_cutoff=cutoff)
    return hit[0] if hit else None

# IMPORTANT: this replaces the earlier "load_activity_pressure_matrix_with_businesses"
def load_activity_pressure_matrix_with_businesses() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Excel layout handled here:
      - One sheet per sector.
      - COLUMN HEADERS are activity names (e.g., 'Residential buildings', 'Wind Energy', ...).
      - FIRST COLUMN contains the 13 pressure row labels.
      - The numeric cells are the intensity/value (0..1) for [pressure, activity].
    Returns:
      matrix_df: ['Sector','Activity','Activity_norm'] + PRESSURE_COLS
      business_df: empty placeholder (kept for shape compatibility)
    """
    from data_config import ACTIVITY_PRESSURE_FILE  # your configured Path
    xls_path = _find_activity_excel(ACTIVITY_PRESSURE_FILE)
    if xls_path is None or not Path(xls_path).exists():
        # No Excel: return empty (app will show areas but no pressures)
        return (pd.DataFrame(columns=["Sector","Activity","Activity_norm"] + PRESSURE_COLS),
                pd.DataFrame(columns=["Sector","Activity","Activity_norm","Business"]))

    book = pd.read_excel(xls_path, sheet_name=None, engine="openpyxl", header=0)

    rows = []
    for sector, df in (book or {}).items():
        if df is None or df.empty:
            continue

        # Identify the column that holds pressure labels.
        # Heuristic: choose the first column whose normalized values intersect PRESSURE_COLS a lot
        best_col = None
        best_hits = -1
        for c in df.columns:
            vals = pd.Series(df[c]).astype(str).map(_norm)
            hits = sum(v in {_norm(p) for p in PRESSURE_COLS} for v in vals)
            if hits > best_hits:
                best_hits = hits
                best_col = c

        if best_col is None or best_hits <= 0:
            continue  # sheet does not look like a pressure matrix

        # Build a normalized pressure-name column
        df = df.copy()
        df["_pressure_norm"] = df[best_col].astype(str).map(_norm)

        # Keep only the 13 known rows
        keep_norms = {_norm(p) for p in PRESSURE_COLS}
        df = df[df["_pressure_norm"].isin(keep_norms)]
        if df.empty:
            continue

        # Ensure we have exactly one row per pressure; if duplicates, take first
        df = df.drop_duplicates(subset=["_pressure_norm"], keep="first")

        # Reindex to the canonical order
        norm_to_label = {_norm(p): p for p in PRESSURE_COLS}
        df["PressureLabel"] = df["_pressure_norm"].map(norm_to_label)
        df = df.set_index("PressureLabel").reindex(PRESSURE_COLS)

        # Every other column is an activity column
        activity_cols = [c for c in df.columns if c not in [best_col, "_pressure_norm"]]
        for act_col in activity_cols:
            act_name = str(act_col).strip()
            # coerce numbers; leave NaN if empty
            vals = pd.to_numeric(df[act_col], errors="coerce")
            rec = {"Sector": sector, "Activity": act_name, "Activity_norm": _norm(act_name)}
            for p, v in zip(PRESSURE_COLS, vals.values.tolist()):
                rec[p] = v
            rows.append(rec)

    matrix_df = pd.DataFrame(rows)
    if not matrix_df.empty:
        # Clip to [0,1] if your Excel uses that scale; drop if wildly outside
        for p in PRESSURE_COLS:
            matrix_df[p] = pd.to_numeric(matrix_df[p], errors="coerce")
            matrix_df[p] = matrix_df[p].clip(lower=0, upper=1)

    # No per-activity "business" flags in wide layout; return empty placeholder
    business_df = pd.DataFrame(columns=["Sector","Activity","Activity_norm","Business"])
    return matrix_df, business_df


def filter_land_use_to_activities(landuse_df: _pd.DataFrame, matrix_df: _pd.DataFrame) -> _pd.DataFrame:
    """Match land-use TagValue rows to Excel activities (robust: normalize + synonyms + optional fuzzy)."""
    if landuse_df is None or landuse_df.empty or matrix_df is None or matrix_df.empty:
        return _pd.DataFrame(columns=["TagValue","AreaEnd_m2","Sector","Activity","Activity_norm"] + PRESSURE_COLS)

    df = landuse_df.copy()
    df["TagValue"] = df["TagValue"].astype(str)

    m = matrix_df.copy()
    if "Activity_norm" not in m.columns:
        m["Activity_norm"] = m["Activity"].map(_normalize_activity_name)
    valid = m["Activity_norm"].dropna().unique().tolist()

    matches = []
    for raw in df["TagValue"].astype(str):
        q = _normalize_activity_name(raw)
        if q in valid:
            matches.append(q)
        else:
            best = _fuzzy_match_norm(q, valid, cutoff=85)
            matches.append(best if best else "")
    df["Activity_norm"] = matches
    df = df[df["Activity_norm"] != ""].copy()

    merged = df.merge(
        m[["Sector","Activity","Activity_norm"] + PRESSURE_COLS].drop_duplicates(),
        how="inner", on="Activity_norm"
    )
    keep = ["TagValue","AreaEnd_m2","Sector","Activity","Activity_norm"] + PRESSURE_COLS
    return merged[keep].copy()

def aggregate_activities_all_grids(cells_df: pd.DataFrame) -> pd.DataFrame:
    mdf, _bdf = load_activity_pressure_matrix_with_businesses()

    parts = []
    for _, r in cells_df.iterrows():
        h3i, pos = r["h3_index"], r["position"]
        lu_df, _ = load_land_use(h3i, pos)
        if lu_df.empty:
            continue
        # keep only real activities via synonyms + Excel activities
        filt = filter_land_use_to_activities(lu_df, mdf) if not mdf.empty else pd.DataFrame()
        parts.append(filt if not filt.empty else lu_df.assign(Activity_norm=lu_df["TagValue"].map(_norm)))

    if not parts:
        return pd.DataFrame(columns=["Activity","TotalArea_m2","Activity_norm"] + PRESSURE_COLS)

    all_lu = pd.concat(parts, ignore_index=True)

    # Sum area by (normalized) activity
    grouped = (all_lu
               .assign(Activity=all_lu.get("Activity", all_lu["TagValue"]))
               .groupby(["Activity_norm","Activity"], as_index=False)
               .agg(TotalArea_m2=("AreaEnd_m2","sum")))

    # Attach pressures from Excel by Activity_norm
    out = grouped.merge(
        mdf.drop_duplicates(subset=["Activity_norm"])[["Activity_norm"] + PRESSURE_COLS],
        how="left", on="Activity_norm"
    )

    # Order by area
    out = out.sort_values(["TotalArea_m2","Activity"], ascending=[False, True]).reset_index(drop=True)
    return out


    # ---- FALLBACK (no matches) ----
    # Group raw land-use by TagValue across all grids so the page still shows something.
    # Pressures remain NaN; you can still pick an activity but the widget will be empty.
    lu_rows = []
    for _, r in cells_df.iterrows():
        h3i, pos = r["h3_index"], r["position"]
        lu_df, _ = load_land_use(h3i, pos)
        if lu_df.empty:
            continue
        lu_rows.append(lu_df[["TagValue","AreaEnd_m2"]])
    if not lu_rows:
        return pd.DataFrame(columns=["Activity","TotalArea_m2","Sectors","Businesses"] + PRESSURE_COLS)

    all_lu = pd.concat(lu_rows, ignore_index=True)
    agg = (all_lu.groupby("TagValue")
                 .agg(TotalArea_m2=("AreaEnd_m2","sum"))
                 .reset_index())
    agg["Activity"] = agg["TagValue"]
    agg["Sectors"] = ""
    agg["Businesses"] = ""
    agg["Activity_norm"] = agg["Activity"].apply(_normalize_activity_name)
    # add the pressure columns as NaN
    for p in PRESSURE_COLS:
        agg[p] = np.nan
    agg = agg[["Activity","TotalArea_m2","Sectors","Businesses","Activity_norm"] + PRESSURE_COLS]
    agg = agg.sort_values(["TotalArea_m2","Activity"], ascending=[False, True]).reset_index(drop=True)
    return agg


def get_activity_row(all_activities_df: _pd.DataFrame, activity_name: str) -> _pd.Series | None:
    """Find a single activity row by normalized name (fallback to exact Activity)."""
    if all_activities_df is None or all_activities_df.empty:
        return None
    norm = _normalize_activity_name(activity_name)
    m = all_activities_df[all_activities_df["Activity_norm"] == norm]
    if len(m): 
        return m.iloc[0]
    m = all_activities_df[all_activities_df["Activity"] == activity_name]
    return m.iloc[0] if len(m) else None
