"""
DCRM Analysis API Wrapper
==========================
FastAPI wrapper for the DCRM analysis pipeline.
Accepts CSV uploads via POST and returns comprehensive JSON analysis.

Endpoint: POST /api/circuit-breakers/{breaker_id}/tests/upload
"""

import os
import json
import traceback
from typing import Optional
import sys

# Add project root to sys.path to allow importing from core
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Previous Name: fastapi_app.py
from fastapi import FastAPI, File, UploadFile, HTTPException, Path
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from io import StringIO

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Ensure API key is set
if not os.getenv("GOOGLE_API_KEY"):
    print("WARNING: GOOGLE_API_KEY not found in environment variables. Please check your .env file.")

from langchain_google_genai import ChatGoogleGenerativeAI
from core.calculators.kpi import calculate_kpis
from core.calculators.cbhi import compute_cbhi
from core.signal.phases import analyze_dcrm_data
from core.engines.rules import analyze_dcrm_advanced
from core.agents.diagnosis import detect_fault, standardize_input
from core.utils.report_generator import generate_dcrm_json
from core.agents.recommendation import generate_recommendations

# Optional ViT Model
try:
    from core.models.vit_classifier import predict_dcrm_image, plot_resistance_for_vit
    VIT_AVAILABLE = True
except Exception as e:
    print(f"ViT Model not available: {e}")
    VIT_AVAILABLE = False
    predict_dcrm_image = None
    plot_resistance_for_vit = None

# =============================================================================
# CONFIGURATION - CHANGE THIS URL AFTER DEPLOYMENT
# =============================================================================
DEPLOYMENT_URL = "http://localhost:5000"  # Change this to your deployed URL
# Example: DEPLOYMENT_URL = "https://your-domain.com"
# =============================================================================

# Initialize FastAPI app
app = FastAPI(
    title="DCRM Analysis API",
    description="Circuit Breaker Dynamic Contact Resistance Measurement Analysis",
    version="1.0.0"
)

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this based on your security requirements
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize LLM (reused across requests)
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "DCRM Analysis API",
        "version": "1.0.0",
        "deployment_url": DEPLOYMENT_URL
    }


@app.post("/api/circuit-breakers/{breaker_id}/tests/upload")
async def analyze_dcrm(
    breaker_id: str = Path(..., description="Circuit breaker ID"),
    file: UploadFile = File(..., description="CSV file with DCRM test data")
):
    """
    Analyze DCRM test data from uploaded CSV file.
    
    Expected CSV format:
    - Columns: Time_ms, Resistance, Current, Travel, Close_Coil, Trip_Coil_1, Trip_Coil_2
    - ~400 rows of time-series data
    
    Returns:
    - Comprehensive JSON analysis report matching dcrm-sample-response.txt structure
    """
    
    # Validate file type
    if not file.filename.endswith('.csv'):
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Invalid file type",
                "message": "Only CSV files are accepted",
                "received": file.filename
            }
        )
    
    try:
        # Read CSV file
        contents = await file.read()
        csv_string = contents.decode('utf-8')
        df = pd.read_csv(StringIO(csv_string))
        
        # Validate required columns
        required_columns = ['Time_ms', 'Resistance', 'Current', 'Travel', 
                          'Close_Coil', 'Trip_Coil_1', 'Trip_Coil_2']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Missing required columns",
                    "missing": missing_columns,
                    "required": required_columns,
                    "found": list(df.columns)
                }
            )
        
        # Validate data size
        if len(df) < 100:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Insufficient data",
                    "message": "CSV must contain at least 100 rows of data",
                    "received_rows": len(df)
                }
            )
        
        # =====================================================================
        # MAIN PROCESSING PIPELINE
        # =====================================================================
        
        # 1. Calculate KPIs
        kpi_results = calculate_kpis(df)
        kpis = kpi_results['kpis']
        
        # 2. Phase Segmentation (AI-based)
        phase_analysis_result = analyze_dcrm_data(df, llm)
        
        # 3. Prepare KPIs for Rule Engine and AI Agent
        raj_kpis = {
            "Closing Time (ms)": kpis.get('closing_time'),
            "Opening Time (ms)": kpis.get('opening_time'),
            "Contact Speed (m/s)": kpis.get('contact_speed'),
            "DLRO Value (µΩ)": kpis.get('dlro'),
            "Peak Resistance (µΩ)": kpis.get('peak_resistance'),
            "Peak Close Coil Current (A)": kpis.get('peak_close_coil'),
            "Peak Trip Coil 1 Current (A)": kpis.get('peak_trip_coil_1'),
            "Peak Trip Coil 2 Current (A)": kpis.get('peak_trip_coil_2'),
            "SF6 Pressure (bar)": kpis.get('sf6_pressure'),
            "Ambient Temperature (°C)": kpis.get('ambient_temp'),
            "Main Wipe (mm)": kpis.get('main_wipe'),
            "Arc Wipe (mm)": kpis.get('arc_wipe'),
            "Contact Travel Distance (mm)": kpis.get('contact_travel')
        }
        
        raj_ai_kpis = {
            "kpis": [
                {"name": "Closing Time", "unit": "ms", "value": kpis.get('closing_time')},
                {"name": "Opening Time", "unit": "ms", "value": kpis.get('opening_time')},
                {"name": "DLRO Value", "unit": "µΩ", "value": kpis.get('dlro')},
                {"name": "Peak Resistance", "unit": "µΩ", "value": kpis.get('peak_resistance')},
                {"name": "Contact Speed", "unit": "m/s", "value": kpis.get('contact_speed')},
                {"name": "Peak Close Coil Current", "unit": "A", "value": kpis.get('peak_close_coil')},
                {"name": "Peak Trip Coil 1 Current", "unit": "A", "value": kpis.get('peak_trip_coil_1')},
                {"name": "Peak Trip Coil 2 Current", "unit": "A", "value": kpis.get('peak_trip_coil_2')},
                {"name": "SF6 Pressure", "unit": "bar", "value": kpis.get('sf6_pressure')},
                {"name": "Ambient Temperature", "unit": "°C", "value": kpis.get('ambient_temp')}
            ]
        }
        
        # 4. Standardize resistance data for Rule Engine
        temp_df = df[['Resistance']].copy()
        if len(temp_df) < 401:
            last_val = temp_df.iloc[-1, 0]
            padding = pd.DataFrame({'Resistance': [last_val] * (401 - len(temp_df))})
            temp_df = pd.concat([temp_df, padding], ignore_index=True)
        
        std_df = standardize_input(temp_df)
        row_values = std_df.iloc[0].values.tolist()
        
        # 5. Run Rule Engine Analysis
        rule_engine_result = analyze_dcrm_advanced(row_values, raj_kpis)
        
        # 6. Run AI Agent Analysis
        ai_agent_result = detect_fault(df, raj_ai_kpis)
        
        # 7. Run ViT Model (if available)
        vit_result = None
        vit_plot_path = "temp_vit_plot.png"
        
        plot_generated = False
        try:
            if plot_resistance_for_vit and plot_resistance_for_vit(df, vit_plot_path):
                plot_generated = True
        except Exception as e:
            print(f"ViT Plot generation failed: {e}")
        
        if plot_generated and VIT_AVAILABLE and predict_dcrm_image:
            try:
                vit_class, vit_conf, vit_details = predict_dcrm_image(vit_plot_path)
                if vit_class:
                    vit_result = {
                        "class": vit_class,
                        "confidence": vit_conf,
                        "details": vit_details
                    }
            except Exception as e:
                print(f"ViT Prediction failed: {e}")
        
        # 8. Calculate CBHI Score
        cbhi_phase_data = {}
        if 'phaseWiseAnalysis' in phase_analysis_result:
            for phase in phase_analysis_result['phaseWiseAnalysis']:
                p_name = f"Phase {phase.get('phaseNumber')}"
                cbhi_phase_data[p_name] = {
                    "status": phase.get('status', 'Unknown'),
                    "confidence": phase.get('confidence', 0)
                }
        
        cbhi_score = compute_cbhi(raj_ai_kpis['kpis'], ai_agent_result, cbhi_phase_data)
        
        # 9. Generate Recommendations
        recommendations = generate_recommendations(
            kpis=kpis,
            cbhi_score=cbhi_score,
            rule_faults=rule_engine_result.get("Fault_Detection", []),
            ai_faults=ai_agent_result.get("Fault_Detection", []),
            llm=llm
        )
        
        # 10. Generate Final JSON Report
        full_report = generate_dcrm_json(
            df=df,
            kpis=kpis,
            cbhi_score=cbhi_score,
            rule_result=rule_engine_result,
            ai_result=ai_agent_result,
            llm=llm,
            vit_result=vit_result,
            phase_analysis_result=phase_analysis_result,
            recommendations=recommendations
        )
        
        # Add breaker_id to response
        full_report['breakerId'] = breaker_id
        
        # Return JSON response
        return JSONResponse(content=full_report, status_code=200)
    
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    
    except pd.errors.EmptyDataError:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Empty CSV file",
                "message": "The uploaded CSV file is empty or contains no data"
            }
        )
    
    except pd.errors.ParserError as e:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "CSV parsing error",
                "message": "Failed to parse CSV file. Please check the file format.",
                "details": str(e)
            }
        )
    
    except Exception as e:
        # Log the full error for debugging
        error_trace = traceback.format_exc()
        print(f"ERROR in DCRM analysis: {error_trace}")
        
        # Return clean error to client
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Analysis failed",
                "message": "An error occurred during DCRM analysis",
                "error_type": type(e).__name__,
                "error_details": str(e)
            }
        )


@app.get("/api/health")
async def health_check():
    """Detailed health check with component status"""
    return {
        "status": "healthy",
        "components": {
            "llm": "operational",
            "vit_model": "available" if VIT_AVAILABLE else "unavailable",
            "kpi_calculator": "operational",
            "rule_engine": "operational",
            "ai_agent": "operational",
            "phase_analysis": "operational"
        },
        "deployment_url": DEPLOYMENT_URL
    }


if __name__ == "__main__":
    import uvicorn
    
    # Run the API server
    # Change host and port as needed
    uvicorn.run(
        app, 
        host="0.0.0.0",  # Listen on all interfaces
        port=5001,       # Change port if needed
        log_level="info"
    )
