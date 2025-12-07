import pandas as pd
import sys
import os
import numpy as np

# Add the project root to the path so we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.signal.arcing import calculate_arcing_parameters

def test_arcing_analysis():
    print("Testing calculate_arcing_parameters...")
    
    # Create a synthetic DCRM trace
    # 0-50ms: Closed (R=40)
    # 50-60ms: Motion starts (Travel changes)
    # 60ms: T2 Main Parting (R jumps to 200)
    # 60-80ms: Arcing Zone (R=200)
    # 80ms: T4 Arcing Parting (R jumps to Infinity/2000)
    
    time = np.arange(0, 100, 1) # 0 to 99 ms
    resistance = np.zeros(100)
    travel = np.zeros(100)
    
    # Fill Resistance
    resistance[0:60] = 40.0
    resistance[60:80] = 200.0
    resistance[80:] = 2000.0
    
    # Fill Travel (Linear opening for simple speed calc)
    # Speed = 1 m/s = 1 mm/ms
    travel[0:50] = 0.0
    travel[50:] = np.arange(0, 50, 1)
    
    df = pd.DataFrame({
        'Time_ms': time,
        'Resistance': resistance,
        'Travel': travel,
        'Current': [0]*100 # Not used in this calc
    })
    
    result = calculate_arcing_parameters(df)
    
    print("Status:", result['status'])
    print("Events:", result['events'])
    print("Metrics:", result['metrics'])
    
    # Assertions
    events = result['events']
    metrics = result['metrics']
    
    # T2 should be around 60ms
    assert events['T2_main_separation'] == 60.0
    
    # T4 should be around 80ms
    assert events['T4_arcing_separation'] == 80.0
    
    # Ra should be 200.0
    assert metrics['Ra_avg_arcing_res'] == 200.0
    
    # Da calculation check
    # Speed is approx 1.0 m/s
    # Duration = 20ms
    # Da = 20 * 1.0 = 20.0 mm
    # Allow small margin for gradient calculation differences
    assert 19.0 < metrics['Da_arcing_wipe'] < 21.0
    
    print("Test Passed!")

if __name__ == "__main__":
    test_arcing_analysis()
