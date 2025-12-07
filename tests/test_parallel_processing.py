import unittest
import time
import concurrent.futures
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mocking the processing function to simulate delay
def mock_process_phase(phase_name, delay, api_key):
    print(f"Starting {phase_name} with key {api_key}...")
    time.sleep(delay)
    print(f"Finished {phase_name}")
    return f"Result for {phase_name} using {api_key}"

class TestParallelProcessing(unittest.TestCase):
    def test_parallel_execution(self):
        """
        Verify that 3 tasks taking 2 seconds each finish in roughly 2 seconds total, not 6.
        """
        api_keys = ["KEY_1", "KEY_2", "KEY_3"]
        phases = ["r", "y", "b"]
        delay = 2.0
        
        start_time = time.time()
        
        results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_to_phase = {
                executor.submit(mock_process_phase, phase, delay, api_keys[i]): phase 
                for i, phase in enumerate(phases)
            }
            
            for future in concurrent.futures.as_completed(future_to_phase):
                phase = future_to_phase[future]
                try:
                    data = future.result()
                    results[phase] = data
                except Exception as exc:
                    print(f'{phase} generated an exception: {exc}')
                    
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\nTotal duration: {duration:.2f} seconds")
        print(f"Results: {results}")
        
        # Assertions
        self.assertLess(duration, 3.0, "Parallel execution took too long! Should be close to 2s, not 6s.")
        self.assertEqual(len(results), 3, "Did not get results for all 3 phases")
        self.assertTrue("KEY_1" in results['r'])
        self.assertTrue("KEY_2" in results['y'])
        self.assertTrue("KEY_3" in results['b'])

if __name__ == '__main__':
    unittest.main()
