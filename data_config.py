from pathlib import Path

# All data files live in the repo root
ROOT = Path(".")

# Optional: base URL to your git repo for clickable file links (e.g. GitHub “blob” URL)
GIT_BASE_URL = ""  # "https://github.com/<user>/<repo>/blob/main"

# H3 grid config (used only if you don’t provide a CSV with your 7 cells)
RES = 6
CENTER = dict(lat=37.75, lon=22.41)

# Core files/patterns in MAIN DIR
BIODIVERSITY_METRICS = ROOT / "Biodiversity_Metrics_By_H3.csv"
SPECIES_FILE = lambda h3, pos: ROOT / f"species_pool_{h3}_{pos}.csv"
LAND_USE_FILE = lambda h3, pos: ROOT / f"Land_use_{h3}_{pos}.csv"

# Environmental risks — prefer H3-named, fallback to position-named
ENV_RISK_FILE_BY_H3 = lambda h3: ROOT / f"environmental_risks_{h3}.csv"
ENV_RISK_FILE = lambda pos: ROOT / f"environmental_risks_{pos}.csv"

# Environmental/climate time-series per POSITION (files you listed)
ENV_TIMESERIES_FILE = lambda pos: ROOT / f"environmental_data_{pos}_.csv"

# Population or other drivers (optional)
POP_FILE = lambda h3, pos: ROOT / f"population_data_{h3}_{pos}.csv"

# Aqueduct V4 (central hex only)
AQUEDUCT_FILE = lambda center_h3: ROOT / f"aqueduct_v4_{center_h3}.csv"

# Land cover (GLC-FCS30D + DW) geojsons in MAIN DIR
LANDCOVER_GLOB = "*.geojson"

# EcoIntegrity helpers
ECO_FILE = lambda stem: ROOT / f"{stem}.csv"

# Optional: CSV with your 7 H3 cells (columns: position,h3_index)
H3_LIST_FILE = ROOT / "h3_res7_center_plus_neighbors_minimal.csv"

# Windfarm KML overlay (global path, UTF-8 name)
WIND_KML = ROOT / "Δυτικο Λυρκειο Πολυγωνο ΡΑΑΕΥ.kml"

# Year slider
YEAR_MIN, YEAR_MAX = 2005, 2025

