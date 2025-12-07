# DCRM Analysis API - Quick Start Guide

## 📋 Overview
This FastAPI wrapper provides a REST API endpoint for DCRM (Dynamic Contact Resistance Measurement) analysis. It accepts CSV uploads and returns comprehensive JSON analysis reports.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements_api.txt
```

### 2. Configure Deployment URL
Open `dcrm_api.py` and update line 46:
```python
DEPLOYMENT_URL = "http://localhost:5000"  # Change to your deployed URL
```

### 3. Run the API Server
```bash
python dcrm_api.py
```

The API will start on `http://localhost:5000`

## 📡 API Endpoints

### Main Analysis Endpoint
**POST** `/api/circuit-breakers/{breaker_id}/tests/upload`

**Parameters:**
- `breaker_id` (path parameter): Circuit breaker identifier
- `file` (form-data): CSV file with DCRM test data

**Example Request (using curl):**
```bash
curl -X POST \
  "http://localhost:5000/api/circuit-breakers/6926e63d4614721a79b7b24e/tests/upload" \
  -F "file=@df3_final.csv"
```

**Example Request (using Python requests):**
```python
import requests

url = "http://localhost:5000/api/circuit-breakers/6926e63d4614721a79b7b24e/tests/upload"
files = {'file': open('df3_final.csv', 'rb')}
response = requests.post(url, files=files)
print(response.json())
```

**Example Request (using JavaScript/Fetch):**
```javascript
const formData = new FormData();
formData.append('file', csvFile);  // csvFile is a File object

fetch('http://localhost:5000/api/circuit-breakers/6926e63d4614721a79b7b24e/tests/upload', {
  method: 'POST',
  body: formData
})
.then(response => response.json())
.then(data => console.log(data));
```

### Health Check Endpoints

**GET** `/`
- Simple health check
- Returns API status and version

**GET** `/api/health`
- Detailed health check
- Returns status of all components (LLM, ViT model, etc.)

## 📄 CSV Format Requirements

The uploaded CSV must contain these columns:
- `Time_ms` - Time in milliseconds
- `Resistance` - Resistance values in µΩ
- `Current` - Current values
- `Travel` - Travel distance
- `Close_Coil` - Close coil current
- `Trip_Coil_1` - Trip coil 1 current
- `Trip_Coil_2` - Trip coil 2 current

**Minimum rows:** 100 (typically ~400 rows)

## 📤 Response Format

The API returns a comprehensive JSON report matching the structure in `data/dcrm-sample-response.txt`, including:

- **aiVerdict**: AI-generated fault analysis and recommendations
- **breakerId**: Circuit breaker ID from the request
- **cbhi**: Composite Breaker Health Index with history
- **kpis**: Array of Key Performance Indicators
- **phaseWiseAnalysis**: Detailed analysis of 5 operational phases
- **waveform**: Time-series data with SHAP values
- **findings**: Summary of detected faults
- **healthScore**: Overall health score (0-100)

## ⚠️ Error Handling

The API returns detailed error messages for common issues:

### Invalid File Type (400)
```json
{
  "error": "Invalid file type",
  "message": "Only CSV files are accepted",
  "received": "data.xlsx"
}
```

### Missing Columns (400)
```json
{
  "error": "Missing required columns",
  "missing": ["Travel", "Current"],
  "required": ["Time_ms", "Resistance", "Current", ...],
  "found": ["Time_ms", "Resistance", ...]
}
```

### Insufficient Data (400)
```json
{
  "error": "Insufficient data",
  "message": "CSV must contain at least 100 rows of data",
  "received_rows": 50
}
```

### Analysis Failure (500)
```json
{
  "error": "Analysis failed",
  "message": "An error occurred during DCRM analysis",
  "error_type": "ValueError",
  "error_details": "..."
}
```

## 🔧 Configuration Options

### Change Port
Edit the last line in `dcrm_api.py`:
```python
uvicorn.run(app, host="0.0.0.0", port=5000)  # Change port here
```

### Enable/Disable CORS
Modify the CORS middleware settings (line 60):
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific domains for security
    ...
)
```

### Update Google API Key
Change the API key on line 20:
```python
os.environ["GOOGLE_API_KEY"] = "your-api-key-here"
```

## 🌐 Deployment

### Local Development
```bash
python dcrm_api.py
```

### Production (using Gunicorn + Uvicorn)
```bash
pip install gunicorn
gunicorn dcrm_api:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:5000
```

### Docker Deployment
Create a `Dockerfile`:
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements_api.txt .
RUN pip install -r requirements_api.txt
COPY . .
CMD ["python", "dcrm_api.py"]
```

Build and run:
```bash
docker build -t dcrm-api .
docker run -p 5000:5000 dcrm-api
```

## 📊 API Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:5000/docs
- **ReDoc**: http://localhost:5000/redoc

These provide interactive API documentation and testing interfaces.

## 🧪 Testing

Test with the sample CSV:
```bash
curl -X POST \
  "http://localhost:5000/api/circuit-breakers/test-123/tests/upload" \
  -F "file=@df3_final (1).csv" \
  -o response.json
```

## 🔍 Monitoring

The API logs all requests and errors to the console. For production, consider:
- Setting up proper logging (to files or logging services)
- Adding request/response monitoring
- Implementing rate limiting
- Adding authentication if needed

## 💡 Tips

1. **Processing Time**: Analysis typically takes 30-60 seconds due to AI processing
2. **Concurrent Requests**: The API can handle multiple requests, but heavy AI processing may slow down responses
3. **File Size**: Keep CSV files under 5MB for optimal performance
4. **Caching**: Consider implementing caching for repeated analyses of the same data

## 📞 Support

For issues or questions, check:
- API logs in the console
- Error messages in the response
- The `/api/health` endpoint for component status
