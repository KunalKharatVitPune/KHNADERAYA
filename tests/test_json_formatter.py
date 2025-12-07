import sys
import os
import pandas as pd
import numpy as np
import json
from unittest.mock import MagicMock

# Add parent directory to path to import analysis
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set dummy API key
os.environ["GOOGLE_API_KEY"] = "TEST_KEY"

try:
    from core.utils.report_generator import generate_dcrm_json
    print("Successfully imported response_formatter.")
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

def test_json_generation():
    print("Creating mock data...")
    # Mock DataFrame
    df = pd.DataFrame({
        "Time_ms": np.linspace(0, 400, 401),
        "Current": np.random.rand(401) * 100,
        "Resistance": np.random.rand(401) * 1000,
        "Travel": np.random.rand(401) * 200
    })
    
    # Mock KPIs
    kpis = {
        "closing_time": 45.0,
        "opening_time": 30.0,
        "contact_speed": 5.5,
        "dlro": 40.0,
        "peak_resistance": 50.0,
        "peak_close_coil": 5.0,
        "peak_trip_coil_1": 5.0,
        "peak_trip_coil_2": 5.0,
        "sf6_pressure": 6.5,
        "ambient_temp": 25.0,
        "main_wipe": 15.0,
        "arc_wipe": 30.0,
        "contact_travel": 180.0
    }
    
    # Mock Results
    rule_result = {"Fault_Detection": [{"defect_name": "Test Fault", "description": "Desc", "Severity": "Low"}]}
    ai_result = {"Fault_Detection": []}
    
    # Mock LLM
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = json.dumps({
        "aiVerdict": {
            "aiAdvice": [{"id": "1", "title": "Test Advice"}],
            "confidence": 90,
            "faultLabel": "Test Label",
            "severity": "Low"
        },
        "phaseWiseAnalysis": [{"phaseNumber": 1, "name": "Phase 1"}]
    })
    mock_llm.invoke.return_value = mock_response
    
    print("Generating JSON...")
    json_output = generate_dcrm_json(df, kpis, 95, rule_result, ai_result, mock_llm)
    
    print("Verifying structure...")
    assert "aiVerdict" in json_output
    assert "phaseWiseAnalysis" in json_output
    assert "kpis" in json_output
    assert "waveform" in json_output
    assert len(json_output["waveform"]) == 401
    assert json_output["healthScore"] == 95
    
    print("JSON Output Keys:", json_output.keys())
    print("Verification SUCCESS!")

if __name__ == "__main__":
    test_json_generation()
