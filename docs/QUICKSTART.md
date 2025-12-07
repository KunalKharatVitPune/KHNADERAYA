# Three-Phase DCRM Flask API - Quick Start Guide

## 🚀 Quick Start (3 Steps)

### Step 1: Start the Flask API
```bash
python dcrm_flask_api.py
```

**Expected output:**
```
 * Running on http://0.0.0.0:5000
 * Restarting with stat
 * Debugger is active!
```

### Step 2: Open the Test Frontend
Double-click: `test_three_phase_frontend.html`

OR open in browser: `file:///c:/codes/dcrm/pranit/csv_kpi/combined/test_three_phase_frontend.html`

### Step 3: Upload and Test
1. Enter Breaker ID (default: `6926e63d4614721a79b7b24e`)
2. Enter Operator Name (default: `Test Engineer`)
3. Upload 3 CSV files:
   - 🔴 Red Phase (R): `df3_final (1).csv`
   - 🟡 Yellow Phase (Y): `df3_final (1).csv`
   - 🔵 Blue Phase (B): `df3_final (1).csv`
4. Click "Upload & Analyze All Phases"
5. Wait 90-180 seconds for processing
6. View results!

---

## 📝 Alternative: Command Line Test

```bash
# Terminal 1: Start API
python dcrm_flask_api.py

# Terminal 2: Run automated test
python test_three_phase_flask.py
```

---

## 🎯 What You'll See

### Overall Results:
- Overall Health Score (average of 3 phases)
- Breaker ID
- Operator Name
- Creation Timestamp

### Per-Phase Results:
- Red Phase (R): Health Score, CBHI, Findings
- Yellow Phase (Y): Health Score, CBHI, Findings
- Blue Phase (B): Health Score, CBHI, Findings

### Complete JSON:
- Full three-phase analysis
- Enhanced waveform with coil currents
- AI recommendations with performanceGains & riskMitigation
- All fields matching `sample.json` structure

---

## 📊 API Endpoints

### Three-Phase Analysis
```
POST /api/circuit-breakers/{breaker_id}/tests/upload-three-phase
```

**Request:**
- `fileR`: CSV file for Red phase
- `fileY`: CSV file for Yellow phase
- `fileB`: CSV file for Blue phase
- `operator`: (optional) Operator name

**Response:** Combined three-phase JSON

### Health Check
```
GET /api/health
```

---

## 🔧 Configuration

### Change Port
Edit `dcrm_flask_api.py` line ~400:
```python
app.run(host="0.0.0.0", port=5000, debug=True)
```

### Change API URL in Frontend
Edit `test_three_phase_frontend.html` line ~300:
```javascript
const API_URL = 'http://localhost:5000';
```

---

## ✅ Features Implemented

✅ Flask-based API (instead of FastAPI)
✅ Three-phase processing (fileR, fileY, fileB)
✅ Enhanced waveform (includes close_coil, trip_coil_1, trip_coil_2)
✅ Updated AI recommendations (performanceGains, riskMitigation)
✅ Combined JSON structure matching sample.json
✅ Beautiful HTML frontend with 3-file upload
✅ Automated test script
✅ Complete documentation

---

## 🎉 You're All Set!

Everything is ready to use. Just start the API and open the HTML frontend!
