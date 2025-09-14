
import os, re, json, pandas as pd, numpy as np
from pathlib import Path
from typing import Optional

OUT_PARQUET_DIR = "data/parquet_new"  # created relative to working dir unless absolute


def clean_bytes(val):
    if isinstance(val, str) and val.startswith("b'") and val.endswith("'"):
        # strip leading b'' and trailing spaces inside
        inner = val[2:-1]
        return inner.strip()
    return val


def find_first(globbed):
    return globbed[0] if globbed else None


def read_csv_safe(path: Optional[str]) -> pd.DataFrame:
    if not path or not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1")
    # clean common byte-like strings
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].apply(clean_bytes)
    return df


def load_inputs(folder: str):
    p = Path(folder)
    files = {f.name.lower(): str(f) for f in p.glob("*.csv")}
    # heuristics: pick by suffix fragments
    meta = next((f for name,f in files.items() if "meta" in name), None)
    traj = next((f for name,f in files.items() if "traj" in name), None)
    prof = next((f for name,f in files.items() if "prof" in name or "profile" in name), None)
    tech = next((f for name,f in files.items() if "tech" in name), None)
    return meta, traj, prof, tech


def coerce_float(s):
    try:
        return float(s)
    except Exception:
        return np.nan


def coerce_int(s):
    try:
        return int(s)
    except Exception:
        return np.nan


def coerce_time(s):
    # try pandas to_datetime; handle many formats
    try:
        return pd.to_datetime(s, errors="coerce", utc=True).tz_convert(None)
    except Exception:
        return pd.NaT


def extract_platform_series(df: pd.DataFrame):
    for col in ["platform_number", "wmo", "WMO", "platformNumber"]:
        if col in df.columns:
            s = df[col].astype(str).str.extract(r"(\d+)")[0]
            return s
    # fallback: empty series
    return pd.Series([np.nan]*len(df))


def build_floats(meta_df: pd.DataFrame, profiles_df: pd.DataFrame) -> pd.DataFrame:
    # Combine to get float list
    plats = pd.concat([extract_platform_series(meta_df), extract_platform_series(profiles_df)], ignore_index=True)
    plats = plats.dropna().astype(str)
    if plats.empty:
        return pd.DataFrame(columns=["platform_number","first_time","last_time","n_profiles"])
    # get first/last times from profiles if available
    if "time" in profiles_df.columns:
        grp = profiles_df.groupby("platform_number")["time"].agg(["min","max","count"]).reset_index()
        grp.columns = ["platform_number","first_time","last_time","n_profiles"]
        return grp
    else:
        return pd.DataFrame({"platform_number": plats.unique(), "first_time": pd.NaT, "last_time": pd.NaT, "n_profiles": np.nan})


def canonicalize_profiles(df_list):
    # Combine info from trajectory/profiles/technical to form a minimal profiles table
    frames = []
    for df in df_list:
        if df.empty: 
            continue
        # rename plausible columns to canonical
        rename_map = {}
        for c in df.columns:
            lc = c.lower()
            if lc in ("platform_number","wmo"): rename_map[c] = "platform_number"
            if lc in ("cycle_number","cycle","cycle_no"): rename_map[c] = "cycle_number"
            if lc in ("lat","latitude","roundlat"): rename_map[c] = "latitude"
            if lc in ("lon","longitude","roundlon"): rename_map[c] = "longitude"
            if lc in ("time","juld","date"): rename_map[c] = "time"
        tmp = df.rename(columns=rename_map).copy()
        # keep only useful columns
        keep = [c for c in ["platform_number","cycle_number","latitude","longitude","time"] if c in tmp.columns]
        tmp = tmp[keep]
        # clean
        if "platform_number" in tmp.columns:
            tmp["platform_number"] = tmp["platform_number"].astype(str).str.extract(r"(\d+)")[0]
        if "cycle_number" in tmp.columns:
            tmp["cycle_number"] = tmp["cycle_number"].apply(coerce_int)
        if "latitude" in tmp.columns:
            tmp["latitude"] = tmp["latitude"].apply(coerce_float)
        if "longitude" in tmp.columns:
            tmp["longitude"] = tmp["longitude"].apply(coerce_float)
        if "time" in tmp.columns:
            tmp["time"] = tmp["time"].apply(coerce_time)
        frames.append(tmp)
    if not frames:
        return pd.DataFrame(columns=["profile_id","platform_number","cycle_number","latitude","longitude","time"])
    prof = pd.concat(frames, ignore_index=True)
    # drop rows missing both lat/lon/time
    if {"latitude","longitude","time"}.issubset(prof.columns):
        prof = prof.dropna(subset=["latitude","longitude","time"], how="all")
    # deduplicate on (platform, cycle, time) where present
    dedup_keys = [c for c in ["platform_number","cycle_number","time"] if c in prof.columns]
    if dedup_keys:
        prof = prof.sort_values(dedup_keys).drop_duplicates(dedup_keys, keep="last")
    # assign profile_id
    prof = prof.reset_index(drop=True)
    prof.insert(0, "profile_id", range(len(prof)))
    return prof[["profile_id","platform_number","cycle_number","latitude","longitude","time"]]


def main(folder: str, out_dir: str = OUT_PARQUET_DIR):
    from pathlib import Path
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    meta_path, traj_path, prof_path, tech_path = load_inputs(folder)
    meta_df = read_csv_safe(meta_path)
    traj_df = read_csv_safe(traj_path)
    prof_df = read_csv_safe(prof_path)
    tech_df = read_csv_safe(tech_path)

    profiles = canonicalize_profiles([traj_df, prof_df, tech_df])
    floats = build_floats(meta_df, profiles)

    # For PoC, observations may not be present in CSV exports; write empty table
    observations = pd.DataFrame(columns=["profile_id","level","pressure_dbar","temperature_c","salinity_psu"])

    floats.to_parquet(os.path.join(out_dir,"floats.parquet"))
    profiles.to_parquet(os.path.join(out_dir,"profiles.parquet"))
    observations.to_parquet(os.path.join(out_dir,"observations.parquet"))
    print(f"[ok] Wrote Parquet to {out_dir}")
    print(f"floats: {len(floats)} rows, profiles: {len(profiles)} rows, observations: {len(observations)} rows")

if __name__ == "__main__":
    import argparse, os
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_folder", required=True, help="Folder containing *.csv (metadata/trajectory/profiles/technical)")
    ap.add_argument("--out", default=OUT_PARQUET_DIR, help="Output Parquet dir")
    args = ap.parse_args()
    main(args.csv_folder, args.out)
