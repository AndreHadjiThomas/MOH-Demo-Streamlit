from pathlib import Path

# === All data files live in the repo root ===
ROOT = Path(".")

# Optional: base URL to your git repo for clickable links
# e.g. "https://github.com/<user>/<repo>/blob/main"
GIT_BASE_URL = ""

# H3 grid config (only used if you don't provide a CSV with your 7 cells)
RES = 6
CENTER = dict(lat=37.75, lon=22.41)

# Biodiversity metrics (single CSV in main dir)
BIODIVERSITY_METRICS = ROOT / "Biodiversity_Metrics_By_H3.csv"

# Species (birds, invasive, sensitive)
SPECIES_FILE = lambda h3, pos: ROOT / f"species_pool_{h3}_{pos}.csv"

# Activities / land-use per hex
LAND_USE_FILE = lambda h3, pos: ROOT / f"Land_use_{h3}_{pos}.csv"

# Protected ecosystems / environmental risks per position
ENV_RISK_FILE = lambda pos: ROOT / f"environmental_risks_{pos}.csv"

# Aqueduct V4 (central hex only)
AQUEDUCT_FILE = lambda center_h3: ROOT / f"aqueduct_v4_{center_h3}.csv"

# Land cover (GLC-FCS30D + Dynamic World) — GeoJSONs are in main dir
LANDCOVER_GLOB = "*.geojson"

# EcoIntegrity (main dir)
ECO_FILE = lambda stem: ROOT / f"{stem}.csv"

# Optional CSV listing your 7 H3 cells: columns [position,h3_index]
H3_LIST_FILE = ROOT / "h3_res7_center_plus_neighbors_minimal.csv"

# Slider bounds
YEAR_MIN, YEAR_MAX = 2005, 2025



# Slider bounds
YEAR_MIN, YEAR_MAX = 2005, 2025
