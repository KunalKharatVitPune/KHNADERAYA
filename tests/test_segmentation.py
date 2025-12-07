import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.signal.phases import analyze_dcrm_data
import pandas as pd

# Load data
try:
    df = pd.read_excel('data/dcrm_waveform (1).xlsx')
    print(f"Loaded data: {df.shape[0]} samples, {df.shape[1]} channels")
except FileNotFoundError:
    print("Data file not found, skipping data load.")
    # Create dummy data
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

# Run segmentation
result = analyze_dcrm_data(df)

# Print results
print("\n" + "="*60)
print("SEGMENTATION RESULTS")
print("="*60)

if 'phaseWiseAnalysis' in result:
    for phase in result['phaseWiseAnalysis']:
        phase_num = phase['phaseNumber']
        name = phase['name']
        start = phase.get('startTime', 0)
        end = phase.get('endTime', 0)
        duration = end - start
        
        print(f"\nPhase {phase_num}: {name}")
        print(f"  Time: {start:.1f} - {end:.1f} ms")
        print(f"  Duration: {duration:.1f} ms")
else:
    print("No phaseWiseAnalysis found in result.")
    print(result.keys())

print("\n" + "="*60)
print("METADATA")
print("="*60)
# analyze_dcrm_data doesn't return metadata in the same way, so we skip specific metadata checks
print("Segmentation completed.")

