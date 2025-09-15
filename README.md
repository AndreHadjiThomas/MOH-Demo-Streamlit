# HelMos Biodiversity — Streamlit

## Setup
```bash
pip install -r app/requirements.txt
streamlit run app/app.py
```

Place your data repo under `./data` (default), or update paths in `app/data_config.py`.

## Expected files
- `data/Biomet/MOH/Biodiversity_Metrics_By_H3.csv`
- `data/Biomet/MOH/Species_Harmonized/species_pool_{h3}_{position}.csv`
- `data/Biomet/MOH/Land_use_{h3}_{position}.csv`
- `data/Biomet/MOH/environmental_risks_{position}.csv`
- `data/Biomet/MOH/aqueduct_v4_{CENTER_H3}.csv`
- `data/MOH_Land_Cover/<h3>_<pos>_<year>_(glc_fcs30d|landcover).geojson`
- `data/Biomet/MOH/EcoIntegrity/...` (see `data_config.py`)

Optional: set `GIT_BASE_URL` to enable clickable links to source files in the UI.
