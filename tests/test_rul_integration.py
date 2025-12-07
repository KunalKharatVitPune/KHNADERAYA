import pandas as pd
import json
import sys
import os

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.utils.report_generator import generate_dcrm_json

# Mock LLM
class MockLLM:
    def invoke(self, messages):
        return type('obj', (object,), {
            'content': json.dumps({
                "aiVerdict": {
                    "faultLabel": "Healthy",
                    "severity": "Low",
                    "confidence": 95,
                    "rulEstimate": "Old Estimate", # Should be overwritten
                    "uncertainty": "Old Uncertainty"
                }
            })
        })

def test_rul_integration():
    # Load data
    try:
        df = pd.read_csv('df3_final.csv')
    except FileNotFoundError:
        # Create dummy df if file not found
        df = pd.DataFrame({
            'Time_ms': range(100),
            'Resistance': [100]*100,
            'Current': [10]*100,
            'Travel': [50]*100
        })
    
    # Mock inputs
    kpis = {
        "closing_time": 45.0,
        "opening_time": 35.0,
        "dlro": 50.0,
        "peak_resistance": 300.0,
        "contact_speed": 5.0,
        "peak_close_coil": 5.0,
        "peak_trip_coil_1": 5.0,
        "peak_trip_coil_2": 5.0,
        "sf6_pressure": 6.0,
        "ambient_temp": 25.0,
        "main_wipe": 15.0,
        "arc_wipe": 15.0,
        "contact_travel": 550.0
    }
    
    cbhi_score = 95.0
    
    rule_result = {"Fault_Detection": []}
    ai_result = {"Fault_Detection": []}
    
    llm = MockLLM()
    
    # Run generation
    print("Running generate_dcrm_json...")
    result = generate_dcrm_json(
        df=df,
        kpis=kpis,
        cbhi_score=cbhi_score,
        rule_result=rule_result,
        ai_result=ai_result,
        llm=llm
    )
    
    # Check RUL
    ai_verdict = result.get("aiVerdict", {})
    rul = ai_verdict.get("rulEstimate")
    uncertainty = ai_verdict.get("uncertainty")
    ai_advice = ai_verdict.get("aiAdvice", [])
    
    print(f"RUL Estimate: {rul}")
    print(f"Uncertainty: {uncertainty}")
    print(f"AI Advice Items: {len(ai_advice)}")
    
    if rul and rul != "Old Estimate" and uncertainty != "Old Uncertainty":
        print("SUCCESS: RUL was calculated and overwrote the LLM placeholder.")
    else:
        print("FAILURE: RUL was not updated.")

    if ai_advice and len(ai_advice) >= 3:
        print("SUCCESS: AI Advice was generated.")
    else:
        print("FAILURE: AI Advice was not generated or insufficient items.")

if __name__ == "__main__":
    test_rul_integration()
