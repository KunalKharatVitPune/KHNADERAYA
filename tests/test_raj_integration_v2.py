import sys
import os
import pandas as pd
import numpy as np
import json

# Add parent directory to path to import analysis
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set dummy API key for testing imports and basic logic (if needed)
os.environ["GOOGLE_API_KEY"] = "TEST_KEY"

try:
    from core.engines.rules import analyze_dcrm_advanced
    from core.agents.diagnosis import standardize_input
    print("Successfully imported Raj modules.")
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

def create_mock_df():
    # Create a time series
    t = np.linspace(0, 400, 401) 
    
    # Mock signals
    # Resistance: High (open) -> Low (closed) -> High (open)
    resistance = np.ones_like(t) * 1000000 # Open
    resistance[(t >= 100) & (t < 300)] = 40 # Closed (40 uOhm)
    
    df = pd.DataFrame({
        "Time_ms": t,
        "Resistance": resistance
    })
    
    return df

def test_integration():
    print("Creating mock DataFrame...")
    df = create_mock_df()
    
    print("Standardizing input...")
    std_df = standardize_input(df)
    row_values = std_df.iloc[0].values.tolist()
    
    print("Row values length:", len(row_values))
    assert len(row_values) == 401
    
    # Mock KPIs
    raj_kpis = {
        "Closing Time (ms)": 100.0,
        "Opening Time (ms)": 35.0,
        "Contact Speed (m/s)": 5.0,
        "DLRO Value (µΩ)": 40.0,
        "Peak Resistance (µΩ)": 50.0,
        "Peak Close Coil Current (A)": 5.0,
        "Peak Trip Coil 1 Current (A)": 5.0,
        "Peak Trip Coil 2 Current (A)": 5.0,
        "SF6 Pressure (bar)": 6.0,
        "Ambient Temperature (°C)": 25.0
    }
    
    print("Running analyze_dcrm_advanced (Rule Engine)...")
    result = analyze_dcrm_advanced(row_values, raj_kpis)
    
    print("Rule Engine Result Keys:", result.keys())
    assert "Fault_Detection" in result
    assert "overall_health_assessment" in result
    
    print("Faults detected:", json.dumps(result["Fault_Detection"], indent=2))
    
    # We can't easily test the AI agent without a real API key and network, 
    # but we can verify the rule engine logic which is deterministic.
    
    print("Verification SUCCESS!")

if __name__ == "__main__":
    test_integration()
