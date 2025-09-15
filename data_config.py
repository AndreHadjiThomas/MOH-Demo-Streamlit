from pathlib import Path

# Root where your data repo is located locally (adjust as needed)
DATA_ROOT = Path("./data")

# Optional: a base URL to your repo for clickable links in the UI.
# Examples:
#   raw:  "https://raw.githubusercontent.com/<user>/<repo>/main"
#   blob: "https://github.com/<user>/<repo>/blob/main"
GIT_BASE_URL = ""  # leave empty to disable links

# H3 grid config
RES = 6
CENTER = dict(lat=37.75, lon=22.41)

# Land cover (GLC-FCS30D + DW) GeoJSONs
LANDCOVER_DIR = DATA_ROOT / "MOH_Land_Cover"

# Biodiversity metrics (single CSV)
BIODIVERSITY_METRICS = DATA_ROOT / "Biomet/MOH/Biodiversity_Metrics_By_H3.csv"

# Species harmonized pools (per hex)
SPECIES_DIR = DATA_ROOT / "Biomet/MOH/Species_Harmonized"

# Activities / Land use (per hex)
LAND_USE_DIR = DATA_ROOT / "Biomet/MOH"

# Environmental risks (per position only)
ENV_RISK_DIR = DATA_ROOT / "Biomet/MOH"

# Aqueduct (center hex only)
AQUEDUCT_DIR = DATA_ROOT / "Biomet/MOH"

# Eco-Integrity files
ECO_DIR = DATA_ROOT / "Biomet/MOH/EcoIntegrity"
ECO_FILES = dict(
    high_integrity=ECO_DIR / "high_integrity",
    rapid_decline=ECO_DIR / "rapid_decline",
    corridors=ECO_DIR / "corridors",
    avg_ffi=ECO_DIR / "avg_ffi",
)

# Slider bounds
YEAR_MIN, YEAR_MAX = 2005, 2025
