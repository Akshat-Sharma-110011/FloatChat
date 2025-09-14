"""Create a local DuckDB database and attach Parquet tables as SQL views.
Run:
  python -m src.duckdb_init
"""
import duckdb, os
from .config import DUCKDB_PATH, PARQUET_DIR

DDL = '''
CREATE TABLE IF NOT EXISTS floats AS SELECT * FROM read_parquet('{parquet}/floats.parquet');
CREATE TABLE IF NOT EXISTS profiles AS SELECT * FROM read_parquet('{parquet}/profiles.parquet');
CREATE TABLE IF NOT EXISTS observations AS SELECT * FROM read_parquet('{parquet}/observations.parquet');
-- Indices for common filters
CREATE INDEX IF NOT EXISTS idx_profiles_time ON profiles(time);
CREATE INDEX IF NOT EXISTS idx_profiles_loc ON profiles(latitude, longitude);
CREATE INDEX IF NOT EXISTS idx_obs_prof ON observations(profile_id);
'''

def main():
    db = duckdb.connect(DUCKDB_PATH)
    db.execute(DDL.format(parquet=PARQUET_DIR.replace("'","''")))
    print(f"DuckDB ready at {DUCKDB_PATH}")
    db.close()

if __name__ == "__main__":
    main()
