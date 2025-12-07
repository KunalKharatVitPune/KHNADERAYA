import sys
import os
import pandas as pd
import json

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.engines.advanced_rules import AdvancedDCRMEngine

def test_engine():
    print("Testing AdvancedDCRMEngine...")
    
    # Load test data
    try:
        df = pd.read_csv("test.csv")
        print(f"Loaded test.csv with shape {df.shape}")
    except FileNotFoundError:
        print("test.csv not found, creating synthetic data...")
        import numpy as np
        t = np.linspace(0, 400, 401)
        r = np.ones_like(t) * 100000
        r[100:300] = 40 + np.random.normal(0, 2, 200) # Healthy main contact
        # Add some "chatter"
        r[150:200] += 10 * np.sin(2 * np.pi * 60 * t[150:200]/1000) 
        df = pd.DataFrame({'Time_ms': t, 'Resistance': r})

    # Initialize Engine
    engine = AdvancedDCRMEngine()
    
    # Mock Segments (since we removed internal logic)
    # Based on the synthetic data creation above:
    # Main contact is 100:300
    # Closing is 80:100 (implied, though synthetic data didn't explicitly create it, let's assume)
    # Opening is 300:320
    segments = {
        'phase2_start': 80, 'phase2_end': 100,
        'phase3_start': 100, 'phase3_end': 300,
        'phase4_start': 300, 'phase4_end': 320,
        'valid': True
    }
    
    # Run Analysis
    try:
        report = engine.analyze(df, segments)
        print("\nAnalysis Successful!")
        print(json.dumps(report, indent=2))
        
        # Check for expected keys
        assert "advanced_analysis" in report
        assert "health_score" in report["advanced_analysis"]
        assert "physics_metrics" in report["advanced_analysis"]
        
        metrics = report["advanced_analysis"]["physics_metrics"]
        print(f"\nExtracted Metrics:")
        print(f"Main Resistance: {metrics['main_contact_resistance_uohm']} uOhm")
        print(f"Chatter Power: {metrics['mechanical_chatter_power']}")
        print(f"Roughness RMS: {metrics['surface_roughness_rms']}")
        
    except Exception as e:
        print(f"\nAnalysis Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_engine()
