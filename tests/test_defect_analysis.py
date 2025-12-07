import pandas as pd
import sys
import os
from unittest.mock import MagicMock

# Add the project root to the path so we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.engines.diagnostics import perform_defect_analysis

def test_defect_analysis():
    print("Testing perform_defect_analysis...")
    
    # Create a dummy DataFrame
    df = pd.DataFrame({
        'Time_ms': range(100),
        'Current': [0] * 100,
        'Resistance': [100] * 100,
        'Travel': [0] * 100
    })
    
    # Mock the LLM
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = """
    {
      "image_url": "test_url",
      "overall_condition": "Healthy",
      "executive_lead": "Test Executive Lead",
      "detected_issues": [],
      "analysis_metrics": {
        "static_resistance_Rp_uOhm": 50.0,
        "signal_noise_level": "Low",
        "wipe_quality": "Normal"
      },
      "maintenance_recommendation": "None"
    }
    """
    mock_llm.invoke.return_value = mock_response
    
    # Run the analysis
    try:
        result = perform_defect_analysis(df, mock_llm)
        print("Analysis successful!")
        print("Result:", result)
        
        # Basic assertions
        assert result['overall_condition'] == "Healthy"
        assert result['executive_lead'] == "Test Executive Lead"
        
    except Exception as e:
        print(f"Analysis failed: {e}")
        raise

if __name__ == "__main__":
    test_defect_analysis()
