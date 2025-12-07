# DCRM API Testing Guide

## 🎯 Two Ways to Test the API

You have **two easy options** to test your DCRM API:

---

## Option 1: HTML Frontend (Visual Testing) ⭐ RECOMMENDED

### Steps:

1. **Start the API server:**
   ```bash
   python dcrm_api.py
   ```
   Wait for: `Uvicorn running on http://0.0.0.0:5000`

2. **Open the test frontend:**
   - Simply double-click `test_frontend.html` in File Explorer
   - OR open it in your browser: `file:///c:/codes/dcrm/pranit/csv_kpi/combined/test_frontend.html`

3. **Test the API:**
   - Enter Breaker ID (default is already filled)
   - Click "Choose CSV File" and select `df3_final (1).csv`
   - Click "Upload & Analyze"
   - Wait 30-60 seconds for processing
   - View the complete JSON response!

### Features:
✅ Beautiful visual interface  
✅ Shows key metrics (Health Score, CBHI, Findings)  
✅ Displays complete JSON response  
✅ Copy JSON to clipboard  
✅ Download JSON as file  
✅ Real-time status updates  
✅ Error handling with clear messages  

---

## Option 2: Python Test Script (Command Line)

### Steps:

1. **Start the API server** (in one terminal):
   ```bash
   python dcrm_api.py
   ```

2. **Run the test script** (in another terminal):
   ```bash
   python test_api.py
   ```

### What it does:
✅ Tests health check endpoint  
✅ Uploads sample CSV  
✅ Verifies analysis works  
✅ Tests error handling  
✅ Saves results to `test_response.json`  

---

## Option 3: FastAPI Built-in Docs (Interactive)

### Steps:

1. **Start the API server:**
   ```bash
   python dcrm_api.py
   ```

2. **Open Swagger UI in browser:**
   ```
   http://localhost:5000/docs
   ```

3. **Test the endpoint:**
   - Click on `POST /api/circuit-breakers/{breaker_id}/tests/upload`
   - Click "Try it out"
   - Enter breaker_id: `6926e63d4614721a79b7b24e`
   - Upload CSV file
   - Click "Execute"
   - View response!

---

## 🔍 Quick Comparison

| Method | Best For | Pros | Cons |
|--------|----------|------|------|
| **HTML Frontend** | Visual testing, demos | Beautiful UI, easy to use, copy/download | Need to open in browser |
| **Python Script** | Automated testing, CI/CD | Automated, saves to file | Command line only |
| **Swagger UI** | API exploration | Interactive docs, built-in | Less visual appeal |

---

## 💡 Recommended Workflow

1. **Development:** Use HTML Frontend (`test_frontend.html`)
2. **Automated Testing:** Use Python Script (`test_api.py`)
3. **API Documentation:** Use Swagger UI (`/docs`)

---

## 📝 Expected Output

All methods will return JSON with this structure:

```json
{
  "breakerId": "6926e63d4614721a79b7b24e",
  "healthScore": 95,
  "cbhi": {
    "score": 95,
    "history": [...]
  },
  "kpis": [...],
  "phaseWiseAnalysis": [...],
  "aiVerdict": {...},
  "waveform": [...],
  "findings": "Minor Contact Pitting & Bounce"
}
```

---

## ⚠️ Troubleshooting

**HTML Frontend shows "Cannot connect to API":**
- Make sure API server is running (`python dcrm_api.py`)
- Check if server is on port 5000
- Look for CORS errors in browser console (F12)

**Python test script fails:**
- Ensure `df3_final (1).csv` exists in the directory
- Check if API server is running
- Verify port 5000 is not blocked

**Analysis takes too long:**
- Normal processing time is 30-60 seconds
- AI processing (Gemini) can be slow
- Check your internet connection

---

## 🎉 Quick Start (All-in-One)

```bash
# Terminal 1: Start API
python dcrm_api.py

# Terminal 2: Run automated test
python test_api.py

# Browser: Open visual test
# Double-click test_frontend.html
```

That's it! You're ready to test your DCRM API! 🚀
