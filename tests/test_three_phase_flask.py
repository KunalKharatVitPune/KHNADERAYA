"""
Test script for Three-Phase DCRM Flask API
==========================================
Tests the three-phase endpoint with 3 CSV files.
"""

import requests
import json
import sys

# Configuration
API_URL = "http://localhost:5000"
BREAKER_ID = "6926e63d4614721a79b7b24e"
CSV_FILE = "df3_final (1).csv"  # We'll use the same file for all 3 phases for testing

def test_health_check():
    """Test the health check endpoint"""
    print("Testing health check endpoint...")
    try:
        response = requests.get(f"{API_URL}/api/health")
        if response.status_code == 200:
            print("Health check passed!")
            print(json.dumps(response.json(), indent=2))
            return True
        else:
            print(f"Health check failed with status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Make sure the server is running!")
        print("   Run: python dcrm_flask_api.py")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_three_phase_analysis():
    """Test the three-phase analysis endpoint"""
    print(f"\nTesting three-phase analysis endpoint...")
    
    try:
        # Open the CSV file 3 times (for R, Y, B phases)
        with open(CSV_FILE, 'rb') as fileR, \
             open(CSV_FILE, 'rb') as fileY, \
             open(CSV_FILE, 'rb') as fileB:
            
            files = {
                'fileR': (CSV_FILE, fileR, 'text/csv'),
                'fileY': (CSV_FILE, fileY, 'text/csv'),
                'fileB': (CSV_FILE, fileB, 'text/csv')
            }
            
            data = {
                'operator': 'Test Engineer'
            }
            
            url = f"{API_URL}/api/circuit-breakers/{BREAKER_ID}/tests/upload-three-phase"
            
            print(f"Uploading to: {url}")
            print("Processing 3 phases (this may take 90-180 seconds)...")
            
            response = requests.post(url, files=files, data=data, timeout=300)
            
            if response.status_code == 200:
                print("Three-phase analysis completed successfully!")
                
                # Parse response
                result = response.json()
                
                # Display key results
                print("\nKey Results:")
                print(f"   Breaker ID: {result.get('breakerId')}")
                print(f"   Overall Health Score: {result.get('healthScore')}/100")
                print(f"   Operator: {result.get('operator')}")
                print(f"   Created At: {result.get('createdAt')}")
                
                # Check each phase
                for phase in ['r', 'y', 'b']:
                    if phase in result:
                        phase_data = result[phase]
                        print(f"\n   {phase.upper()} Phase:")
                        print(f"      Health Score: {phase_data.get('healthScore')}/100")
                        print(f"      CBHI Score: {phase_data.get('cbhi', {}).get('score')}/100")
                        print(f"      Findings: {phase_data.get('findings')}")
                        
                        # Check waveform structure
                        waveform = phase_data.get('waveform', [])
                        if waveform:
                            first_point = waveform[0]
                            required_fields = ['time', 'current', 'resistance', 'travel', 'shap', 
                                             'close_coil', 'trip_coil_1', 'trip_coil_2']
                            has_all_fields = all(field in first_point for field in required_fields)
                            print(f"      Waveform fields: {'Complete' if has_all_fields else 'Missing fields'}")
                            if has_all_fields:
                                print(f"         Sample: time={first_point['time']}, current={first_point['current']}, "
                                      f"close_coil={first_point['close_coil']}")
                        
                        # Check effectAnalysis
                        ai_verdict = phase_data.get('aiVerdict', {})
                        effect_analysis = ai_verdict.get('effectAnalysis', {})
                        has_performance_gains = 'performanceGains' in effect_analysis
                        has_risk_mitigation = 'riskMitigation' in effect_analysis
                        print(f"      Performance Gains: {'Present' if has_performance_gains else 'Missing'}")
                        print(f"      Risk Mitigation: {'Present' if has_risk_mitigation else 'Missing'}")
                
                # Save full response
                output_file = "test_three_phase_response.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                print(f"\nFull response saved to: {output_file}")
                
                return True
            else:
                print(f"Analysis failed with status {response.status_code}")
                print(f"   Error: {response.text}")
                return False
                
    except FileNotFoundError:
        print(f"CSV file not found: {CSV_FILE}")
        print("   Make sure the file exists in the current directory")
        return False
    except requests.exceptions.Timeout:
        print("Request timeout. Analysis took too long.")
        return False
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("Three-Phase DCRM Flask API Test Suite")
    print("=" * 60)
    
    # Test 1: Health check
    if not test_health_check():
        print("\nHealth check failed. Stopping tests.")
        sys.exit(1)
    
    # Test 2: Three-phase analysis
    if not test_three_phase_analysis():
        print("\nThree-phase analysis test failed.")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
