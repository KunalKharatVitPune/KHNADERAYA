import sys
import os
import pandas as pd
import numpy as np
import json

# Add parent directory to path to import analysis
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.calculators.kpi import analyze_breaker_data
from core.calculators.cbhi import compute_cbhi

def create_mock_df():
    # Create a time series
    t = np.linspace(0, 200, 2000) # 0 to 200ms, 0.1ms step
    
    # Mock signals
    # Travel: starts at 0 (open), goes to 100 (closed) around 50ms, stays, then opens around 150ms
    travel = np.zeros_like(t)
    travel[t < 50] = 0
    travel[(t >= 50) & (t < 150)] = 100
    travel[t >= 150] = 0
    # Add some transition slope
    for i in range(len(t)):
        if 40 < t[i] < 60:
            travel[i] = (t[i] - 40) * 5
        elif 140 < t[i] < 160:
            travel[i] = 100 - (t[i] - 140) * 5
    
    # Resistance: High (open) -> Low (closed) -> High (open)
    resistance = np.ones_like(t) * 10000 # Open
    resistance[(t >= 55) & (t < 145)] = 50 # Closed (50 uOhm)
    
    # Coils
    close_coil = np.zeros_like(t)
    close_coil[(t > 10) & (t < 30)] = 2.0 # Pulse
    
    trip_coil_1 = np.zeros_like(t)
    trip_coil_1[(t > 110) & (t < 130)] = 2.0 # Pulse
    
    trip_coil_2 = np.zeros_like(t)
    
    df = pd.DataFrame({
        "Time_ms": t,
        "Resistance": resistance,
        "Travel": travel,
        "Close_Coil": close_coil,
        "Trip_Coil_1": trip_coil_1,
        "Trip_Coil_2": trip_coil_2
    })
    
    return df

def test_integration():
    print("Creating mock DataFrame...")
    df = create_mock_df()
    
    print("Running analyze_breaker_data...")
    kpi_result = analyze_breaker_data(df)
    
    assert "kpis" in kpi_result
    kpis_list = kpi_result["kpis"]
    print("KPIs:", json.dumps(kpis_list, indent=2))
    
    # Verify expected KPIs exist
    expected_names = [
        "Closing Time", "Opening Time", "DLRO Value", "Peak Resistance", 
        "Main Wipe", "Arc Wipe", "Contact Travel Distance", "Contact Speed", 
        "Peak Close Coil Current", "Peak Trip Coil 1 Current", "Peak Trip Coil 2 Current", 
        "Ambient Temperature"
    ]
    
    found_names = [item['name'] for item in kpis_list]
    for name in expected_names:
        assert name in found_names, f"Missing KPI: {name}"
        
    print("Running compute_cbhi...")
    score = compute_cbhi(kpis_list)
    print(f"CBHI Score: {score}")
    
    assert isinstance(score, (int, float))
    assert 0 <= score <= 100
    
    print("Verification SUCCESS!")

if __name__ == "__main__":
    test_integration()

