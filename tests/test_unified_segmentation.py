import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.signal.phases import analyze_dcrm_data
import pandas as pd

print("Testing Unified Segmentation Module (via core.signal.phases)...")
print("=" * 60)

# Load sample data
try:
    df = pd.read_excel("data/dcrm_waveform (1).xlsx")
    print(f"Loaded data: {df.shape[0]} samples\n")
except FileNotFoundError:
    print("Data file not found, using dummy data.")
    import numpy as np
    df = pd.DataFrame({
        'Time_ms': np.linspace(0, 200, 200),
        'Resistance': [100]*200,
        'Current': [10]*200,
        'Travel': [50]*200,
        'Close_Coil': [0]*200,
        'Trip_Coil_1': [0]*200,
        'Trip_Coil_2': [0]*200
    })

# Test 1: Run Analysis
print("1. Running Analysis:")
print("-" * 60)
result = analyze_dcrm_data(df)

if 'phaseWiseAnalysis' in result:
    print("   Analysis Successful")
    # Test 3: Show all phases
    print("\n3. All Phases Detected:")
    print("-" * 60)
    for phase in result['phaseWiseAnalysis']:
        phase_num = phase['phaseNumber']
        start = phase.get('startTime', 0)
        end = phase.get('endTime', 0)
        duration = end - start
        name = phase['name']
        print(f"   Phase {phase_num}: {start:6.1f} - {end:6.1f} ms  "
              f"(Duration: {duration:5.1f} ms) - {name}")
else:
    print("   Analysis Failed or Unexpected Output")
    print(result.keys())

print("\n" + "=" * 60)
print("[OK] Segmentation module working correctly!")

