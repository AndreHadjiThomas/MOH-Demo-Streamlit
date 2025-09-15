from pathlib import Path

ROOT = Path(".")
GIT_BASE_URL = ""

RES = 6
CENTER = dict(lat=37.75, lon=22.41)

BIODIVERSITY_METRICS = ROOT / "Biodiversity_Metrics_By_H3.csv"
SPECIES_FILE = lambda h3, pos: ROOT / f"species_pool_{h3}_{pos}.csv"
LAND_USE_FILE = lambda h3, pos: ROOT / f"Land_use_{h3}_{pos}.csv"

# ✅ NEW: support h3-indexed env risk files
ENV_RISK_FILE_BY_H3 = lambda h3: ROOT / f"environmental_risks_{h3}.csv"
ENV_RISK_FILE = lambda pos: ROOT / f"environmental_risks_{pos}.csv"

AQUEDUCT_FILE = lambda center_h3: ROOT / f"aqueduct_v4_{center_h3}.csv"
LANDCOVER_GLOB = "*.geojson"
ECO_FILE = lambda stem: ROOT / f"{stem}.csv"

H3_LIST_FILE = ROOT / "h3_res7_center_plus_neighbors_minimal.csv"

YEAR_MIN, YEAR_MAX = 2005, 2025

