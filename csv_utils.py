"""
CSV utilities for robust data loading and processing.
Handles frame timestamps, IMU data, and GPS coordinates.
"""

import os, re, time, datetime as dt
import numpy as np
import pandas as pd
from pathlib import Path

# ---------- Core helpers ----------

def _read_csv_smart(path):
    """Robust CSV reader that tries different delimiters."""
    for delim in [None, ',', ';', '\t', '|']:
        try:
            df = pd.read_csv(path, delimiter=delim, engine="python", on_bad_lines="skip")
            if df.shape[1] >= 2:
                return df
        except Exception:
            pass
    return pd.read_csv(path, engine="python", on_bad_lines="skip")


def _canon(s):
    s = str(s).strip().lower().replace(" ", "")
    return re.sub(r"[^a-z0-9_]", "", s)


def _to_epoch_seconds(val):
    """
    Convert a timestamp that might be:
      - numeric epoch in s/ms/us/ns
      - string datetime (ISO, 'YYYY-MM-DD HH:MM:SS', etc.)
    into float seconds since Unix epoch.
    """
    # numeric?
    try:
        x = float(val)
        if x > 1e13:  return x * 1e-9   # ns
        if x > 1e12:  return x * 1e-6   # µs
        if x > 1e10:  return x * 1e-3   # ms
        return x                       # s
    except Exception:
        pass

    # string datetime
    s = str(val).strip()
    # pandas to_datetime handles many formats
    dt_parsed = pd.to_datetime(s, utc=True, errors="coerce", dayfirst=False)
    if pd.isna(dt_parsed):
        # last resort: try dayfirst
        dt_parsed = pd.to_datetime(s, utc=True, errors="coerce", dayfirst=True)
    if pd.isna(dt_parsed):
        raise ValueError(f"Unparseable timestamp: {val!r}")

    return dt_parsed.value / 1e9  # ns -> s


def _find_first(df, aliases):
    cols = {_canon(c): c for c in df.columns}
    for a in aliases:
        if a in cols:
            return cols[a]
    return None


# ---------- Public loaders ----------

def load_frame_csv(path, *, filename_regex=None, allow_filename_column=True):
    """
    Load frame CSV with (frame id OR filename) + timestamp.
    - If a filename column exists, extract frame index with filename_regex.
    - Else use a numeric frame id column.
    Returns DataFrame: columns ['frame', 't_s'] sorted by frame.
    """
    df = _read_csv_smart(path)
    # Try filename first (more robust across tools)
    fname_col = None
    if allow_filename_column:
        fname_col = _find_first(df, ["filename", "file", "path", "image", "img", "name"])
    frame_col = _find_first(df, ["frame", "frame_id", "frameid", "id", "frameindex", "index"])
    ts_col    = _find_first(df, ["timestamp", "time", "stamp", "sec", "epoch", "unix", "unixtime", "t",
                                 "frameid_timestamp", "frame_timestamp"])

    if fname_col is None and frame_col is None:
        raise ValueError(f"Frame CSV must have filename or frame id. Got: {list(df.columns)}")
    if ts_col is None:
        raise ValueError(f"Frame CSV needs a timestamp column. Got: {list(df.columns)}")

    # timestamps
    t_s = df[ts_col].map(_to_epoch_seconds).astype(float).values

    # frame indices
    if fname_col is not None and filename_regex:
        rx = re.compile(filename_regex)
        def _grab(s):
            m = rx.search(str(s))
            if not m:
                return np.nan
            return int(m.group(1))
        frame = df[fname_col].map(_grab).astype('Int64').astype('float').values
    elif frame_col is not None:
        frame = df[frame_col].astype(float).values
    else:
        raise ValueError("Unable to determine frame indices; add a filename column or a frame id column.")

    out = pd.DataFrame({"frame": frame, "t_s": t_s})
    out = out.dropna().astype({"frame": int, "t_s": float}).sort_values("frame").reset_index(drop=True)
    print(f"🗂 Frame CSV: {Path(path).name} rows={len(out)} (cols used: frame, {ts_col})")
    return out


def load_imu_csv(path):
    """
    Load IMU CSV with timestamps, roll/pitch (optional), and GPS (lat/lon or EN).
    Returns DataFrame with columns: t_s, roll_rad, pitch_rad, and any of [lat,lon] or [E,N].
    """
    df = _read_csv_smart(path)
    ts_col = _find_first(df, ["timestamp", "time", "stamp", "sec", "epoch", "unix", "unixtime", "t"])
    if ts_col is None:
        raise ValueError(f"IMU CSV needs a timestamp column. Got: {list(df.columns)}")

    t_s = df[ts_col].map(_to_epoch_seconds).astype(float).values

    def _grab(alias_list):
        c = _find_first(df, alias_list)
        return (df[c].astype(float).values, c) if c else (None, None)

    roll, rn = _grab(["roll", "phi", "att_roll", "gyro_roll", "rolldeg", "roll_deg"])
    pitch, pn = _grab(["pitch", "theta", "att_pitch", "gyro_pitch", "pitchdeg", "pitch_deg"])

    def _to_rad(a):
        if a is None:
            return None
        return np.deg2rad(a) if np.nanmax(np.abs(a)) > np.pi * 2.0 else a

    roll_rad  = _to_rad(roll)  if roll  is not None else np.zeros_like(t_s)
    pitch_rad = _to_rad(pitch) if pitch is not None else np.zeros_like(t_s)

    lat_col = _find_first(df, ["lat", "latitude", "gpslat", "gps_lat"])
    lon_col = _find_first(df, ["lon", "longitude", "long", "lng", "gpslon", "gps_lon"])
    e_col   = _find_first(df, ["e", "east", "utme", "utm_e", "x"])
    n_col   = _find_first(df, ["n", "north", "utmn", "utm_n", "y"])

    out = pd.DataFrame({"t_s": t_s, "roll_rad": roll_rad, "pitch_rad": pitch_rad})
    used = [ts_col]
    if rn: used.append(rn)
    if pn: used.append(pn)

    if lat_col and lon_col:
        out["lat"] = df[lat_col].astype(float).values
        out["lon"] = df[lon_col].astype(float).values
        used += [lat_col, lon_col]
    elif e_col and n_col:
        out["E"] = df[e_col].astype(float).values
        out["N"] = df[n_col].astype(float).values
        used += [e_col, n_col]

    out = out.sort_values("t_s").reset_index(drop=True)
    print(f"🗂 IMU CSV:   {Path(path).name} rows={len(out)} cols_used={used}")
    return out


# Geographic utility (unchanged)
R_EARTH = 6378137.0

def en_to_latlon(lat0_deg, lon0_deg, dE, dN):
    lat0 = np.deg2rad(lat0_deg); lon0 = np.deg2rad(lon0_deg)
    dlat = dN / R_EARTH
    dlon = dE / (R_EARTH * np.cos(lat0))
    lat = lat0 + dlat; lon = lon0 + dlon
    return np.rad2deg(lat), np.rad2deg(lon)