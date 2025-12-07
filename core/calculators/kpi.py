# Previous Name: analysis/kpi_calculator.py
import json
import math
import numpy as np
import pandas as pd
import requests

# =========================
# 1) Weather (optional)
# =========================
def get_ambient_temperature_pune(timeout: int = 3, fallback_c: float = 28.0) -> float:
    try:
        lat, lon = 18.52, 73.86
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        j = requests.get(url, timeout=timeout).json()
        return float(j["current_weather"]["temperature"])
    except Exception:
        return float(fallback_c)

# =========================
# 2) Helpers
# =========================
def rolling_smooth(x: np.ndarray, window: int = 7) -> np.ndarray:
    """Centered rolling median smoothing (robust to spikes)."""
    s = pd.Series(x)
    sm = s.rolling(window=window, center=True).median()
    sm = sm.bfill().ffill().fillna(s)
    return sm.values

def first_time_above(t: np.ndarray, y: np.ndarray, thr: float) -> float:
    """First time y crosses above thr (with linear interpolation)."""
    for i in range(1, len(y)):
        if y[i-1] < thr <= y[i]:
            dy = y[i] - y[i-1]
            dt = t[i] - t[i-1]
            if dy == 0 or dt == 0:
                return float(t[i])
            frac = (thr - y[i-1]) / dy
            return float(t[i-1] + frac * dt)
    if len(y) and y[0] >= thr:
        return float(t[0])
    return math.nan

def crossing_time(t: np.ndarray, y: np.ndarray, thr: float, direction: str) -> float:
    for i in range(1, len(y)):
        y0, y1 = y[i-1], y[i]
        if direction == "below" and y0 > thr >= y1:
            dt = t[i] - t[i-1]
            dy = y1 - y0
            if dy == 0 or dt == 0:
                return float(t[i])
            frac = (thr - y0) / dy
            return float(t[i-1] + frac * dt)
        if direction == "above" and y0 < thr <= y1:
            dt = t[i] - t[i-1]
            dy = y1 - y0
            if dy == 0 or dt == 0:
                return float(t[i])
            frac = (thr - y0) / dy
            return float(t[i-1] + frac * dt)
    return math.nan

def interp_series_at(t: np.ndarray, y: np.ndarray, t_query: float) -> float:
    if np.isnan(t_query):
        return math.nan
    idx = np.searchsorted(t, t_query)
    if idx == 0:
        return float(y[0])
    if idx >= len(t):
        return float(y[-1])
    t0, t1 = t[idx-1], t[idx]
    y0, y1 = y[idx-1], y[idx]
    if t1 == t0:
        return float(y0)
    return float(y0 + (y1 - y0) * (t_query - t0) / (t1 - t0))

# =========================
# 3) KPI computation
# =========================
def analyze_breaker_data(
    df,
    coil_threshold: float = 0.5,
    travel_edge_mm: float = 2.0,
    arc_end_pct_open: float = 0.95,
    dlro_margin_ms: float = 10.0,
    ambient_fallback_c: float = 28.0,
):

    df.columns = [c.strip() for c in df.columns]

    # Extract series as floats
    t = df["Time_ms"].values.astype(float)
    R = df["Resistance"].values.astype(float)
    travel = df["Travel"].values.astype(float)
    close_coil = df["Close_Coil"].values.astype(float)
    trip1 = df["Trip_Coil_1"].values.astype(float)
    trip2 = df["Trip_Coil_2"].values.astype(float)

    # Basic travel stats
    travel_min = float(np.min(travel))
    travel_max = float(np.max(travel))

    # Define open/closed masks
    open_mask = travel <= (travel_min + 5.0)      # near minimum (open)
    closed_mask = travel >= (travel_max - 5.0)    # near maximum (closed)

    # Resistance baselines
    R_open_baseline = float(np.median(R[open_mask])) if np.any(open_mask) else float(np.median(R[:max(1, int(len(R)*0.2))]))
    R_closed_baseline = float(np.median(R[closed_mask])) if np.any(closed_mask) else float(np.median(R[int(len(R)*0.3):int(len(R)*0.7)]))

    # Threshold for state transition
    R_mid = (R_open_baseline + R_closed_baseline) / 2.0

    # Coil energization times
    close_sm = pd.Series(close_coil).rolling(window=3, center=True).max().bfill().ffill().values
    trip_sm  = pd.Series(trip1).rolling(window=3, center=True).max().bfill().ffill().values

    T_close_cmd = first_time_above(t, close_sm, coil_threshold)
    T_trip1_cmd = first_time_above(t, trip_sm,  coil_threshold)

    # Contact make / part
    post_close_mask = t >= T_close_cmd if not math.isnan(T_close_cmd) else np.ones_like(t, dtype=bool)
    T_make = crossing_time(t[post_close_mask], R[post_close_mask], R_mid, direction="below")

    post_trip_mask = t >= T_trip1_cmd if not math.isnan(T_trip1_cmd) else np.ones_like(t, dtype=bool)
    T_part = crossing_time(t[post_trip_mask], R[post_trip_mask], R_mid, direction="above")

    # Travel at make/part
    travel_at_make = interp_series_at(t, travel, T_make)
    travel_at_part = interp_series_at(t, travel, T_part)

    # Main wipe 
    main_wipe_mm = travel_max - travel_at_make if not math.isnan(travel_at_make) else math.nan

    # Arc wipe
    R_arc_end_thr = R_open_baseline * arc_end_pct_open
    after_part_mask = t >= T_part if not math.isnan(T_part) else np.ones_like(t, dtype=bool)
    T_arc_end = crossing_time(t[after_part_mask], R[after_part_mask], R_arc_end_thr, direction="above")
    travel_at_arc_end = interp_series_at(t, travel, T_arc_end)
    arc_wipe_mm = (travel_at_part - travel_at_arc_end) if (not math.isnan(travel_at_part) and not math.isnan(travel_at_arc_end)) else math.nan
    if arc_wipe_mm < 0 and not math.isnan(arc_wipe_mm):
        arc_wipe_mm = abs(arc_wipe_mm)

    # Contact travel distance
    contact_travel_distance_mm = travel_max - travel_min

    # Motion segments and speeds (m/s)
    travel_sm = rolling_smooth(travel, window=7)
    dt = np.gradient(t)
    dy = np.gradient(travel_sm)
    velocity_ms = np.where(dt != 0, dy / dt, 0.0)

    # Closing segment
    if not math.isnan(T_close_cmd):
        start_idx = np.argmax((t >= T_close_cmd) & (travel > (travel_min + travel_edge_mm)))
    else:
        start_idx = np.argmax(travel > (travel_min + travel_edge_mm))
    end_candidates = np.where(travel >= (travel_max - 1.0))[0]
    end_idx = end_candidates[0] if len(end_candidates) else len(t)-1
    closing_time_window_ms = t[end_idx] - t[start_idx] if end_idx > start_idx else math.nan
    closing_distance_mm = travel[end_idx] - travel[start_idx] if end_idx > start_idx else math.nan
    closing_speed_avg_ms = (closing_distance_mm / closing_time_window_ms) if (not math.isnan(closing_time_window_ms) and closing_time_window_ms > 0) else math.nan
    closing_speed_peak_ms = float(np.max(velocity_ms[start_idx:end_idx+1])) if end_idx > start_idx else math.nan

    # Opening segment
    if not math.isnan(T_trip1_cmd):
        dec = np.r_[False, np.diff(travel) < 0]
        opening_start_idx = np.argmax((t >= T_trip1_cmd) & dec)
    else:
        opening_start_idx = np.argmax(np.r_[False, np.diff(travel) < 0])
    open_end_candidates = np.where(travel <= (travel_min + 1.0))[0]
    opening_end_idx = open_end_candidates[-1] if len(open_end_candidates) else len(t)-1
    opening_time_window_ms = t[opening_end_idx] - t[opening_start_idx] if opening_end_idx > opening_start_idx else math.nan
    opening_distance_mm = travel[opening_start_idx] - travel[opening_end_idx] if opening_end_idx > opening_start_idx else math.nan
    opening_speed_avg_ms = (opening_distance_mm / opening_time_window_ms) if (not math.isnan(opening_time_window_ms) and opening_time_window_ms > 0) else math.nan
    opening_speed_peak_ms = float(np.max(velocity_ms[opening_start_idx:opening_end_idx+1])) if opening_end_idx > opening_start_idx else math.nan

    # DLRO
    start_closed = (T_make + dlro_margin_ms) if not math.isnan(T_make) else t[0]
    end_closed   = (T_part - dlro_margin_ms) if not math.isnan(T_part) else t[-1]
    closed_interval_mask = (t >= start_closed) & (t <= end_closed)
    if np.any(closed_interval_mask):
        DLRO_uohm = float(np.median(R[closed_interval_mask]))
    else:
        DLRO_uohm = float(np.median(R[closed_mask])) if np.any(closed_mask) else float(np.median(R))

    # --- ROBUST PEAK RESISTANCE LOGIC ---
    # Peak Resistance should be the maximum value during actual conduction (when contacts are touching).
    # The "open baseline" (850 in this case) represents infinite/saturated resistance and should be excluded.
    # Strategy:
    # 1. Identify the open baseline as the most frequent high value (mode of upper percentile)
    # 2. Find peak in the conduction range (values significantly below open baseline)
    
    # Find the open baseline (saturation value) - use upper 10% of data
    R_sorted = np.sort(R)
    upper_10pct_idx = int(len(R_sorted) * 0.90)
    upper_values = R_sorted[upper_10pct_idx:]
    
    # If upper values are relatively constant (std < 5% of mean), treat as saturation
    if len(upper_values) > 0:
        upper_mean = np.mean(upper_values)
        upper_std = np.std(upper_values)
        
        if upper_std < (upper_mean * 0.05):  # Very stable upper region = saturation
            saturation_value = upper_mean
        else:
            saturation_value = R_open_baseline
    else:
        saturation_value = R_open_baseline
    
    # Define conduction range: values at least 10% below saturation
    conduction_threshold = saturation_value * 0.90
    
    # Find all resistance values in the conduction range
    conduction_mask = R < conduction_threshold
    
    if np.any(conduction_mask):
        peak_resistance_uohm = float(np.max(R[conduction_mask]))
    else:
        # Fallback: use median of lower 50% of data
        lower_half_idx = int(len(R_sorted) * 0.50)
        peak_resistance_uohm = float(np.max(R_sorted[:lower_half_idx])) if lower_half_idx > 0 else float(np.median(R))
    # -------------------------------------

    # Peak coil currents
    peak_close_coil_A = float(np.max(close_coil))
    peak_trip1_coil_A = float(np.max(trip1))
    peak_trip2_coil_A = float(np.max(trip2))

    # Opening/Closing times
    closing_time_ms = (T_make - T_close_cmd) if (not math.isnan(T_make) and not math.isnan(T_close_cmd)) else math.nan
    opening_time_ms = (T_part - T_trip1_cmd) if (not math.isnan(T_part) and not math.isnan(T_trip1_cmd)) else math.nan

    # Ambient temperature
    ambient_temp_c = get_ambient_temperature_pune(fallback_c=ambient_fallback_c)

    # Contact Speed
    contact_speed_ms = None
    if not math.isnan(closing_speed_avg_ms) and not math.isnan(opening_speed_avg_ms):
        contact_speed_ms = (closing_speed_avg_ms + opening_speed_avg_ms) / 2.0
    elif not math.isnan(closing_speed_avg_ms):
        contact_speed_ms = closing_speed_avg_ms
    elif not math.isnan(opening_speed_avg_ms):
        contact_speed_ms = opening_speed_avg_ms

    kpis = [
        {"name": "Closing Time", "value": float(round(closing_time_ms, 2)) if not math.isnan(closing_time_ms) else None, "unit": "ms"},
        {"name": "Opening Time", "value": float(round(opening_time_ms, 2)) if not math.isnan(opening_time_ms) else None, "unit": "ms"},
        {"name": "DLRO Value", "value": float(round(DLRO_uohm, 2)), "unit": "µΩ"},
        {"name": "Peak Resistance", "value": float(round(peak_resistance_uohm, 2)), "unit": "µΩ"},
        {"name": "Main Wipe", "value": float(round(main_wipe_mm, 2)) if not math.isnan(main_wipe_mm) else None, "unit": "mm"},
        {"name": "Arc Wipe", "value": float(round(arc_wipe_mm, 2)) if not math.isnan(arc_wipe_mm) else None, "unit": "mm"},
        {"name": "Contact Travel Distance", "value": float(round(contact_travel_distance_mm, 2)), "unit": "mm"},
        {"name": "Contact Speed", "value": float(round(contact_speed_ms, 2)) if contact_speed_ms is not None else None, "unit": "m/s"},
        {"name": "Peak Close Coil Current", "value": float(round(peak_close_coil_A, 2)), "unit": "A"},
        {"name": "Peak Trip Coil 1 Current", "value": float(round(peak_trip1_coil_A, 2)), "unit": "A"},
        {"name": "Peak Trip Coil 2 Current", "value": float(round(peak_trip2_coil_A, 2)), "unit": "A"},
        {"name": "Ambient Temperature", "value": float(round(ambient_temp_c, 1)), "unit": "°C"},
    ]

    return {"kpis": kpis}

# ==========================================
# 3. MAIN INTERFACE
# ==========================================

def calculate_kpis(df):
    """
    Main entry point. Calculates KPIs.
    Returns a dictionary containing KPIs in the flat format expected by app.py.
    """
    result = analyze_breaker_data(df)
    kpi_list = result['kpis']
    
    # Convert list of dicts to flat dict
    # Expected keys: closing_time, opening_time, dlro, peak_resistance, main_wipe, arc_wipe, 
    # contact_travel, contact_speed, peak_close_coil, peak_trip_coil_1, peak_trip_coil_2, ambient_temp
    
    flat_kpis = {}
    
    key_map = {
        "Closing Time": "closing_time",
        "Opening Time": "opening_time",
        "DLRO Value": "dlro",
        "Peak Resistance": "peak_resistance",
        "Main Wipe": "main_wipe",
        "Arc Wipe": "arc_wipe",
        "Contact Travel Distance": "contact_travel",
        "Contact Speed": "contact_speed",
        "Peak Close Coil Current": "peak_close_coil",
        "Peak Trip Coil 1 Current": "peak_trip_coil_1",
        "Peak Trip Coil 2 Current": "peak_trip_coil_2",
        "Ambient Temperature": "ambient_temp"
    }
    
    for item in kpi_list:
        name = item['name']
        value = item['value']
        if name in key_map:
            flat_kpis[key_map[name]] = value if value is not None else 0.0
            
    # Add default SF6 pressure as it's not in the new logic but used in app
    flat_kpis["sf6_pressure"] = 7.0
    
    return {
        "kpis": flat_kpis
    }
