@echo off
echo ========================================
echo Starting DCRM API Server
echo ========================================
echo.
echo Server will start on http://localhost:5000
echo Press Ctrl+C to stop the server
echo.
echo ========================================
echo.

python -m uvicorn dcrm_api:app --host 0.0.0.0 --port 5000 --reload

pause
