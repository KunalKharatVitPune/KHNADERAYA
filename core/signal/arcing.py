# Previous Name: analysis/arcing_analysis.py
import pandas as pd
import numpy as np
from .segmentation import identify_arcing_events

def calculate_velocity(df, window=5):
    """
    Calculates velocity (m/s) from Travel (mm) and Time (ms).
    """
    # Ensure we work on a copy to avoid SettingWithCopyWarning
    df = df.copy()
    df['Travel_Smooth'] = df['Travel'].rolling(window=window, center=True).mean().fillna(df['Travel'])
    dy = np.gradient(df['Travel_Smooth'])
    dx = np.gradient(df['Time_ms'])
    velocity = np.where(dx != 0, dy / dx, 0)
    return np.abs(velocity)

def calculate_arcing_parameters(df, llm=None):
    """
    Calculates T0, T1, T2, T3, T4 and derived arcing health metrics.
    
    If `llm` is provided, it uses the AI agent to identify the event timestamps (Robust).
    Otherwise, it uses deterministic threshold logic (Fallback).
    """
    results = {
        "events": {},
        "metrics": {},
        "status": "Unknown",
        "method": "Deterministic"
    }
    
    try:
        # 1. Calculate Velocity
        velocity = calculate_velocity(df)
        df['Velocity'] = velocity
        
        t0, t1, t2, t4 = None, None, None, None
        
        # --- A. LLM-Based Detection (Preferred) ---
        if llm:
            try:
                ai_result = identify_arcing_events(df, llm)
                if "events" in ai_result:
                    events = ai_result["events"]
                    t0 = events.get("T0_breaker_closed")
                    t1 = events.get("T1_motion_start")
                    t2 = events.get("T2_main_separation")
                    t4 = events.get("T4_arcing_separation")
                    results["method"] = "AI-Enhanced"
                    results["ai_reasoning"] = ai_result.get("reasoning", "")
            except Exception as e:
                print(f"AI Segmentation failed, falling back to deterministic: {e}")
        
        # --- B. Deterministic Fallback ---
        if t2 is None or t4 is None:
            # T0: Breaker Closed
            t0_window = df[df['Time_ms'] < 10]
            r_static = t0_window['Resistance'].mean() if not t0_window.empty else 0.0
            t0 = 0.0 # Nominal
            
            # T1: Motion Starts
            start_pos = df['Travel'].iloc[0]
            motion_mask = abs(df['Travel'] - start_pos) > 1.0
            if motion_mask.any():
                t1 = float(df.loc[motion_mask.idxmax(), 'Time_ms'])
            
            # T2 & T4: State-Based
            R_ARC_MIN = 150.0
            R_OPEN_MIN = 1500.0
            
            search_start = t1 if t1 else 0
            arcing_candidates = df[(df['Time_ms'] > search_start) & (df['Resistance'] >= R_ARC_MIN)]
            
            if not arcing_candidates.empty:
                t2 = float(df.loc[arcing_candidates.index[0], 'Time_ms'])
                
                open_candidates = df[(df['Time_ms'] > t2) & (df['Resistance'] >= R_OPEN_MIN)]
                if not open_candidates.empty:
                    t4 = float(df.loc[open_candidates.index[0], 'Time_ms'])

        # --- Store Events ---
        results['events']['T0_static_resistance'] = 0.0 # Placeholder, calc below
        if t0 is not None: results['events']['T0_time'] = t0
        if t1 is not None: results['events']['T1_motion_start'] = t1
        if t2 is not None: results['events']['T2_main_separation'] = t2
        if t4 is not None: results['events']['T4_arcing_separation'] = t4
        
        # T3: Duration
        if t2 and t4:
            results['events']['T3_duration_ms'] = round(t4 - t2, 2)
        
        # --- Metric Calculation (Deterministic Math) ---
        
        if t2 and t4:
            # T0 Resistance Value
            t0_val = df[df['Time_ms'] < 10]['Resistance'].mean()
            results['events']['T0_static_resistance'] = round(t0_val, 2) if not np.isnan(t0_val) else 0.0

            # Ra: Average Arcing Resistance (Exclude T4 spike)
            # We add a small buffer to T2 to avoid the rising edge
            arcing_zone = df[(df['Time_ms'] >= t2 + 0.5) & (df['Time_ms'] < t4 - 0.5)]
            
            if not arcing_zone.empty:
                ra = arcing_zone['Resistance'].mean()
            else:
                # Fallback if zone is too small (single point)
                ra = df[(df['Time_ms'] >= t2) & (df['Time_ms'] <= t4)]['Resistance'].mean()
                
            results['metrics']['Ra_avg_arcing_res'] = round(ra, 2)
            
            # Speed at Separation
            speed_during_arc = arcing_zone['Velocity'].mean() if not arcing_zone.empty else 0.0
            results['metrics']['Speed_at_separation'] = round(speed_during_arc, 2)
            
            # Da: Arcing Contact Wipe
            da = (t4 - t2) * speed_during_arc
            results['metrics']['Da_arcing_wipe'] = round(da, 2)
            
            # Wear Index: Ra * Da
            ra_mOhm = ra / 1000.0
            wear_index = ra_mOhm * da
            results['metrics']['Wear_Index'] = round(wear_index, 2)
            
            # --- Health Assessment ---
            if da < 10.0: da_status = "Critical (Short Wipe)"
            elif da < 15.0: da_status = "Warning (Low Wipe)"
            else: da_status = "Healthy"
            results['metrics']['Da_Status'] = da_status
            
            if wear_index > 10.0: wear_status = "Critical (Replace Interrupter)"
            elif wear_index > 5.0: wear_status = "Warning (Worn)"
            else: wear_status = "Healthy"
            results['metrics']['Wear_Status'] = wear_status
            
            results['status'] = "Success"
            
        else:
            results['status'] = "Incomplete Data (Could not find T2/T4)"
            
    except Exception as e:
        results['status'] = f"Error: {str(e)}"
        
    return results
