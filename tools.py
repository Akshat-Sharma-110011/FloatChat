import duckdb, io, base64, os, math
import pandas as pd
import plotly.express as px
from geopy.distance import geodesic
from .config import DUCKDB_PATH, PLOTS_DIR

def sql_run(query: str) -> pd.DataFrame:
    q = query.strip().rstrip(';')
    # read-only guard
    banned = ("insert ", "update ", "delete ", "drop ", "alter ", "create ")
    if any(b in q.lower() for b in banned):
        raise ValueError("Only read-only SQL is permitted.")
    con = duckdb.connect(DUCKDB_PATH)
    df = con.execute(q).fetchdf()
    con.close()
    return df

def plot_profiles_map(df_profiles: pd.DataFrame) -> str:
    fig = px.scatter_mapbox(df_profiles, lat="latitude", lon="longitude",
                            hover_name="platform_number", hover_data=["time","cycle_number"], zoom=2)
    fig.update_layout(mapbox_style="open-street-map", margin=dict(l=0,r=0,t=0,b=0), height=480)
    out = os.path.join(PLOTS_DIR, "map.html")
    fig.write_html(out, include_plotlyjs="cdn")
    return out

def plot_profile_overlay(df_obs: pd.DataFrame) -> str:
    # expects columns: profile_id, level, pressure_dbar, temperature_c, salinity_psu
    # basic TEMP vs PRES plot
    fig = px.line(df_obs, x="temperature_c", y="pressure_dbar", color="profile_id", 
                  labels={"temperature_c":"Temp (°C)", "pressure_dbar":"Pressure (dbar)"}, height=480)
    fig.update_yaxes(autorange="reversed")
    out = os.path.join(PLOTS_DIR, "profiles_temp.html")
    fig.write_html(out, include_plotlyjs="cdn")
    return out

def nearest_floats(lat: float, lon: float, k: int = 5) -> pd.DataFrame:
    df = sql_run("""SELECT DISTINCT platform_number, 
                             avg(latitude) AS lat, avg(longitude) AS lon
                     FROM profiles GROUP BY platform_number""")
    df["distance_km"] = df.apply(lambda r: geodesic((lat, lon), (r.lat, r.lon)).km, axis=1)
    return df.sort_values("distance_km").head(k)
