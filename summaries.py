"""Compute simple summaries and save text docs for RAG.
Run:
  python -m src.summaries
"""
import duckdb, os, json, pathlib
from .config import DUCKDB_PATH, DATA_DIR

def compute_summaries():
    con = duckdb.connect(DUCKDB_PATH)
    recs = []
    # overall stats
    q = con.execute("""
        SELECT 
          min(time) AS t_min, max(time) AS t_max,
          count(*) AS n_profiles, 
          approx_count_distinct(platform_number) AS n_platforms
        FROM profiles
    """).fetchone()
    recs.append({
        "id": "summary_global",
        "text": f"Profiles from {q[0]} to {q[1]}, total {q[2]} profiles across {q[3]} floats."
    })
    # variable coverage
    q2 = con.execute("""
        SELECT 
          count_if(temperature_c IS NOT NULL) AS n_temp,
          count_if(salinity_psu IS NOT NULL) AS n_salt
        FROM observations
    """).fetchone()
    recs.append({
        "id": "coverage",
        "text": f"Coverage: {q2[0]} temperature points, {q2[1]} salinity points."
    })
    con.close()
    out = pathlib.Path(DATA_DIR) / "kb_summaries.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote summaries to {out}")

if __name__ == "__main__":
    compute_summaries()
