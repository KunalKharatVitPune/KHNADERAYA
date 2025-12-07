"""
DCRM Analysis Flask API - Three Phase Support
==============================================
Flask API wrapper for the DCRM analysis pipeline.
Accepts 3 CSV uploads (R, Y, B phases) via POST and returns comprehensive JSON analysis.

Endpoint: POST /api/circuit-breakers/{breaker_id}/tests/upload-three-phase
"""

import os
import json
import traceback
import uuid
from datetime import datetime, timezone
import sys
import concurrent.futures

# Add project root to sys.path to allow importing from core
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Previous Name: flask_app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
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

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access

def get_llm(api_key=None):
    """
    Factory function to create an LLM instance with a specific API key.
    If no key is provided, falls back to the default GOOGLE_API_KEY.
    """
    if not api_key:
        api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        raise ValueError("No Google API Key provided and GOOGLE_API_KEY not found in env.")
        
    return ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0, google_api_key=api_key)


def process_single_phase_csv(args):
    """
    Process a single phase CSV through the complete DCRM pipeline.
    Designed to be run in a separate thread.
    
    Args:
        args: Tuple containing (df, breaker_id, api_key, phase_name)
        
    Returns:
        dict: Complete analysis results for one phase
    """
    df, breaker_id, api_key, phase_name = args
    
    try:
        print(f"[{phase_name.upper()}] Starting processing with key ending in ...{api_key[-4:] if api_key else 'None'}")
        
        # Initialize local LLM for this thread
        llm = get_llm(api_key)
        
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
        
        # 6. Run AI Agent Analysis with error handling
        try:
            ai_agent_result = detect_fault(df, raj_ai_kpis)
            print(f"[{phase_name.upper()}] AI Agent analysis completed successfully")
        except Exception as e:
            print(f"[{phase_name.upper()}] AI Agent failed: {e}. Using fallback.")
            # Fallback: Use rule engine result as AI result
            ai_agent_result = {
                "Fault_Detection": rule_engine_result.get("Fault_Detection", []),
                "overall_health_assessment": rule_engine_result.get("overall_health_assessment", {}),
                "classifications": rule_engine_result.get("classifications", [])
            }
        
        # 7. Run ViT Model (if available)
        vit_result = None
        vit_plot_path = f"temp_vit_plot_{phase_name}_{uuid.uuid4().hex[:8]}.png" # Unique path for parallel safety
        
        plot_generated = False
        try:
            if plot_resistance_for_vit and plot_resistance_for_vit(df, vit_plot_path):
                plot_generated = True
        except Exception as e:
            print(f"[{phase_name.upper()}] ViT Plot generation failed: {e}")
        
        if plot_generated and VIT_AVAILABLE and predict_dcrm_image:
            try:
                # Pass API key to ViT as well if needed, though currently it might use env var
                # The updated vit_classifier uses requests to a deployed model, so API key is for Gemini part
                vit_class, vit_conf, vit_details = predict_dcrm_image(vit_plot_path, api_key=api_key)
                if vit_class:
                    vit_result = {
                        "class": vit_class,
                        "confidence": vit_conf,
                        "details": vit_details
                    }
            except Exception as e:
                print(f"[{phase_name.upper()}] ViT Prediction failed: {e}")
            finally:
                # Cleanup temp file
                if os.path.exists(vit_plot_path):
                    try:
                        os.remove(vit_plot_path)
                    except:
                        pass
        
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
        
        # 9. Generate Recommendations with error handling
        try:
            recommendations = generate_recommendations(
                kpis=kpis,
                cbhi_score=cbhi_score,
                rule_faults=rule_engine_result.get("Fault_Detection", []),
                ai_faults=ai_agent_result.get("Fault_Detection", []),
                llm=llm
            )
            print(f"[{phase_name.upper()}] Recommendations generated successfully")
        except Exception as e:
            print(f"[{phase_name.upper()}] Recommendations failed: {e}. Using fallback.")
            # Fallback: Create basic recommendations from rule engine
            recommendations = {
                "maintenanceActions": [],
                "futureFaultsPdf": []
            }
            # Extract from rule faults
            for fault in rule_engine_result.get("Fault_Detection", []):
                if fault.get("Severity") in ["High", "Critical"]:
                    recommendations["maintenanceActions"].append({
                        "action": f"Address {fault.get('defect_name')}",
                        "priority": "High",
                        "timeframe": "Immediate"
                    })
        
        # 10. Generate Final JSON Report with error handling
        try:
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
            print(f"[{phase_name.upper()}] Final report generated successfully")
        except Exception as e:
            print(f"[{phase_name.upper()}] Report generation failed: {e}. Using fallback.")
            # Fallback: Create minimal valid report
            full_report = {
                "_id": f"fallback_{phase_name}_{uuid.uuid4().hex[:8]}",
                "phase": phase_name,
                "status": "partial_success",
                "error": str(e),
                "ruleBased_result": rule_engine_result,
                "vitResult": vit_result,
                "kpis": kpis,
                "cbhi": {"score": cbhi_score},
                "phaseWiseAnalysis": phase_analysis_result.get('phaseWiseAnalysis', [])
            }
        
        print(f"[{phase_name.upper()}] Processing complete.")
        return full_report

    except Exception as e:
        print(f"[{phase_name.upper()}] Error: {e}")
        traceback.print_exc()
        # Return a partial error result so the whole request doesn't fail
        return {
            "error": str(e),
            "phase": phase_name
        }


@app.route('/')
def root():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "DCRM Analysis Flask API",
        "version": "2.1.0",
        "deployment_url": DEPLOYMENT_URL
    })


@app.route('/api/health')
def health_check():
    """Detailed health check with component status"""
    return jsonify({
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
    })


@app.route('/api/circuit-breakers/<breaker_id>/tests/upload-three-phase', methods=['POST'])
def analyze_three_phase_dcrm(breaker_id):
    """
    Analyze DCRM test data from 3 uploaded CSV files (R, Y, B phases).
    Uses parallel processing with multiple API keys to speed up execution.
    
    Expected files in request:
    - fileR: Red phase CSV
    - fileY: Yellow phase CSV  
    - fileB: Blue phase CSV
    
    Returns:
    - Comprehensive JSON analysis report with combined three-phase results
    """
    
    try:
        # Validate files are present
        if 'fileR' not in request.files or 'fileY' not in request.files or 'fileB' not in request.files:
            return jsonify({
                "error": "Missing required files",
                "message": "All three phase files are required: fileR, fileY, fileB",
                "received": list(request.files.keys())
            }), 400
        
        fileR = request.files['fileR']
        fileY = request.files['fileY']
        fileB = request.files['fileB']
        
        # Validate file types
        for file in [fileR, fileY, fileB]:
            if not file.filename.endswith('.csv'):
                return jsonify({
                    "error": "Invalid file type",
                    "message": "Only CSV files are accepted",
                    "received": file.filename
                }), 400
        
        # Prepare DataFrames
        dfs = {}
        for phase_name, file in [('r', fileR), ('y', fileY), ('b', fileB)]:
            file.seek(0)
            csv_string = file.read().decode('utf-8')
            try:
                df = pd.read_csv(StringIO(csv_string))
                
                # Basic validation
                if len(df) < 100:
                    raise ValueError(f"Insufficient data in {phase_name.upper()} phase")
                
                dfs[phase_name] = df
            except Exception as e:
                return jsonify({
                    "error": f"Error reading {phase_name.upper()} CSV",
                    "details": str(e)
                }), 400

        # Get API Keys
        # Fallback to main key if specific ones aren't set
        main_key = os.getenv("GOOGLE_API_KEY")
        keys = {
            'r': os.getenv("GOOGLE_API_KEY_1", main_key),
            'y': os.getenv("GOOGLE_API_KEY_2", main_key),
            'b': os.getenv("GOOGLE_API_KEY_3", main_key)
        }
        
        # Prepare tasks
        tasks = []
        for phase in ['r', 'y', 'b']:
            tasks.append((dfs[phase], breaker_id, keys[phase], phase))
            
        # Execute in parallel
        results = {}
        health_scores = []
        
        print("Starting parallel processing of 3 phases...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            # Map tasks to futures
            future_to_phase = {
                executor.submit(process_single_phase_csv, task): task[3] 
                for task in tasks
            }
            
            for future in concurrent.futures.as_completed(future_to_phase):
                phase = future_to_phase[future]
                try:
                    result = future.result()
                    results[phase] = result
                    if 'healthScore' in result:
                        health_scores.append(result['healthScore'])
                except Exception as exc:
                    print(f'{phase} generated an exception: {exc}')
                    results[phase] = {"error": str(exc)}

        # Combine results into three-phase structure (removed breakerId and operator)
        combined_result = {
            "_id": str(uuid.uuid4()).replace('-', '')[:24],
            "createdAt": datetime.now(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S GMT"),
            "healthScore": round(sum(health_scores) / len(health_scores), 1) if health_scores else 0,
            "r": results.get('r', {}),
            "y": results.get('y', {}),
            "b": results.get('b', {})
        }
        
        return jsonify(combined_result), 200
    
    except Exception as e:
        # Log the full error for debugging
        error_trace = traceback.format_exc()
        print(f"ERROR in three-phase DCRM analysis: {error_trace}")
        
        # Return clean error to client
        return jsonify({
            "error": "Analysis failed",
            "message": "An error occurred during DCRM analysis",
            "error_type": type(e).__name__,
            "error_details": str(e)
        }), 500


if __name__ == "__main__":
    # Print all registered routes for debugging
    print("Registered Routes:")
    print(app.url_map)
    
    # Run the Flask app
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True,
        use_reloader=False  # Disable reloader to prevent restarts during processing
    )
