import os
from dotenv import load_dotenv
load_dotenv()

DUCKDB_PATH = os.getenv("DUCKDB_PATH", "./data/floatchat.duckdb")
DATA_DIR = os.getenv("DATA_DIR", "./data")
PARQUET_DIR = os.getenv("PARQUET_DIR", "./data/parquet")
PLOTS_DIR = os.getenv("PLOTS_DIR", "./data/plots")

# Indian Ocean PoC default
DEFAULT_BBOX = (20.0, 120.0, -30.0, 25.0)  # lon_min, lon_max, lat_min, lat_max
DEFAULT_DEPTH_RANGE = (0, 2000)            # dbar
DEFAULT_TIME_RANGE = ("2022-01", "2024-12") # yyyy-mm

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PARQUET_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
