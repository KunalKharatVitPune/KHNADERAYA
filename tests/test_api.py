"""
Test script for DCRM API
========================
Simple script to test the API endpoint with the sample CSV file.
"""

import requests
import json
import sys

# Configuration
API_URL = "http://localhost:5000"
BREAKER_ID = "6926e63d4614721a79b7b24e"
CSV_FILE = "df3_final (1).csv"

def test_health_check():
    """Test the health check endpoint"""
    print("🔍 Testing health check endpoint...")
    try:
        response = requests.get(f"{API_URL}/api/health")
        if response.status_code == 200:
            print("✅ Health check passed!")
            print(json.dumps(response.json(), indent=2))
            return True
        else:
            print(f"❌ Health check failed with status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Make sure the server is running!")
        print("   Run: python dcrm_api.py")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_analysis():
    """Test the main analysis endpoint"""
    print(f"\n🔍 Testing analysis endpoint with {CSV_FILE}...")
    
    try:
        # Open and upload the CSV file
        with open(CSV_FILE, 'rb') as f:
            files = {'file': (CSV_FILE, f, 'text/csv')}
            url = f"{API_URL}/api/circuit-breakers/{BREAKER_ID}/tests/upload"
            
            print(f"📤 Uploading to: {url}")
            print("⏳ Processing (this may take 30-60 seconds)...")
            
            response = requests.post(url, files=files, timeout=120)
            
            if response.status_code == 200:
                print("✅ Analysis completed successfully!")
                
                # Parse response
                result = response.json()
                
                # Display key results
                print("\n📊 Key Results:")
                print(f"   Breaker ID: {result.get('breakerId')}")
                print(f"   Health Score: {result.get('healthScore')}/100")
                print(f"   CBHI Score: {result.get('cbhi', {}).get('score')}/100")
                print(f"   Findings: {result.get('findings')}")
                
                # Save full response
                output_file = "test_response.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                print(f"\n💾 Full response saved to: {output_file}")
                
                return True
            else:
                print(f"❌ Analysis failed with status {response.status_code}")
                print(f"   Error: {response.text}")
                return False
                
    except FileNotFoundError:
        print(f"❌ CSV file not found: {CSV_FILE}")
        print("   Make sure the file exists in the current directory")
        return False
    except requests.exceptions.Timeout:
        print("❌ Request timeout. Analysis took too long.")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_invalid_file():
    """Test error handling with invalid file"""
    print("\n🔍 Testing error handling with invalid file...")
    
    try:
        # Create a temporary invalid file
        invalid_content = b"This is not a valid CSV"
        files = {'file': ('invalid.txt', invalid_content, 'text/plain')}
        url = f"{API_URL}/api/circuit-breakers/{BREAKER_ID}/tests/upload"
        
        response = requests.post(url, files=files)
        
        if response.status_code == 400:
            print("✅ Error handling works correctly!")
            print(f"   Error response: {response.json()}")
            return True
        else:
            print(f"⚠️ Unexpected status code: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("DCRM API Test Suite")
    print("=" * 60)
    
    # Test 1: Health check
    if not test_health_check():
        print("\n❌ Health check failed. Stopping tests.")
        sys.exit(1)
    
    # Test 2: Main analysis
    if not test_analysis():
        print("\n❌ Analysis test failed.")
        sys.exit(1)
    
    # Test 3: Error handling
    test_invalid_file()
    
    print("\n" + "=" * 60)
    print("✅ All tests completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()
