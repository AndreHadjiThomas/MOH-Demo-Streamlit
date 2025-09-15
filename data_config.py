from pathlib import Path

# === All data files live in the repo root ===
ROOT = Path(".")

# Optional: base URL to your git repo so the UI can link each chart/table
#   e.g. "https://github.com/<user>/<repo>/blob/main"
# Leave empty "" to disable links.
GIT_BASE_URL = ""

# H3 grid config (used only if you don't provide a CSV list)
RES = 6
CENTER = dict(lat=37.75, lon=22.41)

# --- File locations in MAIN DIR ---
# Biodiversity metrics (single file)
BIODIVERSITY_METRICS = ROOT / "Biodiversity_Metrics_By_H3.csv"

# Species pools (birds / invasive / sensitive)
# pattern: species_pool_{h3}_{position}.csv
SPECIES_FILE = lambda h3, pos: ROOT / f"species_pool_{h3}_{pos}.csv"

# Land-use / activities per hex
# pattern: Land_use_{h3}_{position}.csv
LAND_USE_FILE = lambda h3, pos: ROOT / f"Land_use_{h3}_{pos}.csv"

# Protected ecosystems / environmental risks per position
# pattern: environmental_risks_{position}.csv
ENV_RISK_FILE = lambda pos: ROOT / f"environmental_risks_{pos}.csv"

# Aqueduct (central only)
# pattern: aqueduct_v4_{center_h3}.csv
AQUEDUCT_FILE = lambda center_h3: ROOT / f"aqueduct_v4_{center_h3}.csv"

# Land cover (GLC-FCS30D + DW) geojsons in MAIN DIR
# pattern: <h3>_<pos>_<year>_(glc_fcs30d|landcover).geojson
LANDCOVER_GLOB = "*.geojson"

# EcoIntegrity (all in MAIN DIR)
#   high_integrity_{h3}_{position}_2022.csv
#   rapid_decline_{h3}_{position}_2005_2022.csv
#   corridors_{h3}_{position}_2022.csv
#   avg_ffi_{h3}_{position}.csv
ECO_FILE = lambda stem: ROOT / f"{stem}.csv"

# Optional pre-baked list of 7 H3 cells (center + N1..N6) in main dir:
# columns: position,h3_index
H3_LIST_FILE = ROOT / "h3_res7_center_plus_neighbors_minimal.csv"

# Slider bounds
YEAR_MIN, YEAR_MAX = 2005, 2025


# Slider bounds
YEAR_MIN, YEAR_MAX = 2005, 2025
