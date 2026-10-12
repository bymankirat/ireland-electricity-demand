"""Project settings."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
GRID_DIR = RAW_DIR / "eirgrid"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"
DOCS_DIR = ROOT / "docs"

# EirGrid Smart Grid Dashboard: 15-minute data, Republic of Ireland
GRID_API = ("https://www.smartgriddashboard.com/DashboardService.svc/data"
            "?area={area}&region={region}&datefrom={start}&dateto={end}")
REGION = "ROI"
AREAS = {"demand": "demandactual", "wind": "windactual", "co2": "co2intensity"}
START = "2019-01-01"

# Met Eireann hourly observations, Dublin Airport (station 532); times are UTC
MET_URL = "https://cli.fusio.net/cli/climate_data/webdata/hly532.csv"
MET_FILE = RAW_DIR / "met_hly532.csv"

# Forecast design: issued at midnight for the next 24 hours
TRAIN_END = "2023-12-31 23:00"
VALID = ("2024-01-01", "2024-12-31 23:00")      # used to choose the model
TEST_START = "2025-01-01"                        # judged once, never used for choices
COVID_EXCLUDE = ("2020-03-12", "2020-06-30")     # first lockdown: not used for training
INTERVAL = 0.80
RANDOM_STATE = 42
