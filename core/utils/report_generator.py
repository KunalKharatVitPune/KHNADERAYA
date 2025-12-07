# Previous Name: analysis/response_formatter.py
import json
import pandas as pd
import datetime
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from core.signal.phases import analyze_dcrm_data
from core.calculators.rul import calculate_rul_and_uncertainty
from core.agents.advice import generate_ai_advice

def generate_dcrm_json(df, kpis, cbhi_score, rule_result, ai_result, llm, vit_result=None, phase_analysis_result=None, recommendations=None):
    """
    Generates a comprehensive DCRM analysis JSON report matching the sample format.
    
    Args:
        df (pd.DataFrame): The DCRM data.
        kpis (dict): Calculated KPIs.
        cbhi_score (float): The overall health score.
        rule_result (dict): Results from the Rule Engine.
        ai_result (dict): Results from the AI Agent.
        llm (ChatGoogleGenerativeAI): The LLM instance to use for narrative generation.
        vit_result (dict, optional): Results from ViT model.
        phase_analysis_result (dict, optional): Pre-calculated phase analysis.
        recommendations (dict, optional): Recommendations and future faults.
        
    Returns:
        dict: The formatted JSON report.
    """
    
    # 1. Perform Phase-Wise Analysis (Programmatic + LLM Enhanced)
    # We pass the LLM to analyze_dcrm_data so it can generate descriptions for each phase
    if phase_analysis_result is None:
        phase_analysis_result = analyze_dcrm_data(df, llm)
    
    phase_wise_analysis = phase_analysis_result.get('phaseWiseAnalysis', [])

    # 2. Prepare Context for LLM (Verdict Generation)
    # Summarize faults for the prompt
    rule_faults = rule_result.get("Fault_Detection", [])
    ai_faults = ai_result.get("Fault_Detection", [])
    
    faults_summary = "Deterministic Faults:\n"
    if not rule_faults:
        faults_summary += "- None (Healthy)\n"
    else:
        for f in rule_faults:
            faults_summary += f"- {f.get('defect_name')}: {f.get('description')} (Severity: {f.get('Severity')})\n"
            
    faults_summary += "\nAI Agent Insights:\n"
    if not ai_faults:
        faults_summary += "- None\n"
    else:
        for f in ai_faults:
            faults_summary += f"- {f.get('defect_name')}: {f.get('description')} (Severity: {f.get('Severity')})\n"

    if vit_result:
        faults_summary += f"\nViT Model Prediction:\n- Class: {vit_result.get('class', 'Unknown')}\n- Confidence: {vit_result.get('confidence', 0)*100:.2f}%\n"

    kpi_summary = json.dumps(kpis, indent=2)
    
    # Summarize phase analysis for the verdict prompt
    phase_summary_text = "Phase-Wise Analysis Summary:\n"
    for phase in phase_wise_analysis:
        phase_summary_text += f"Phase {phase['phaseNumber']} ({phase['name']}): {phase['diagnosticVerdict']} (Confidence: {phase.get('confidence', 0)}%)\n"

    # 3. Construct Prompt for AI Verdict
    prompt = f"""
    You are a DCRM (Dynamic Contact Resistance Measurement) expert. 
    Based on the following analysis results, generate the 'aiVerdict' section for a JSON report.
    
    CONTEXT:
    Overall Health Score (CBHI): {cbhi_score}/100
    
    KPIs:
    {kpi_summary}
    
    DETECTED FAULTS:
    {faults_summary}
    
    PHASE ANALYSIS SUMMARY:
    {phase_summary_text}
    
    REQUIREMENTS:
    1. Generate a JSON object with ONE key: "aiVerdict".
    2. "aiVerdict" must include:
       - "aiAdvice": List of actionable advice objects (id, title, description, priority, confidence, expectedImpact, color).
       - "confidence": Overall confidence score (0-100).
       - "effectAnalysis": Object with "shortTerm", "longTerm", "performanceGains", and "riskMitigation" arrays.
         * "shortTerm": Array of immediate effects (within 1-3 months)
         * "longTerm": Array of long-term effects (6+ months)
         * "performanceGains": Array of specific performance improvements expected from recommended actions
         * "riskMitigation": Array of specific risks that will be mitigated by taking action
       - "faultLabel": A short summary label for the condition (e.g., "Minor Contact Wear").
       - "rulEstimate": string (e.g., "10000-12000 cycles").
       - "severity": "Low", "Medium", or "High".
       - "severityReason": Explanation for the severity.
       - "topShapContributors": List of objects (feature, contribution). Make up reasonable features if not provided.
       - "uncertainty": string (e.g., "±1500 cycles").
       
    NOTE: Do NOT generate "futureFaultsPdf" or "maintenanceActions" - these will be provided separately.
        
    3. Return ONLY valid JSON. Do not include markdown formatting.
    
    Example effectAnalysis structure:
    "effectAnalysis": {{
      "shortTerm": ["Effect 1", "Effect 2"],
      "longTerm": ["Effect 1", "Effect 2"],
      "performanceGains": ["Improved contact resistance stability", "Enhanced operational reliability"],
      "riskMitigation": ["Prevent catastrophic failures through early detection", "Reduce unplanned outages by 75%"]
    }}
    """
    
    # 4. Call LLM
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        content = response.content.replace("```json", "").replace("```", "").strip()
        llm_data = json.loads(content)
    except Exception as e:
        # Fallback if LLM fails
        print(f"LLM Generation failed: {e}")
        llm_data = {
            "aiVerdict": {"faultLabel": "Analysis Failed", "severity": "Unknown"}
        }

    # 5. Calculate RUL using optimized logic
    try:
        rul_result = calculate_rul_and_uncertainty(
            kpis=kpis,
            cbhi_score=cbhi_score,
            ai_verdict=llm_data,
            phase_data=phase_analysis_result
        )
        
        # Override LLM generated RUL with calculated values
        if "aiVerdict" in llm_data:
            llm_data["aiVerdict"]["rulEstimate"] = rul_result.get("rulEstimate")
            llm_data["aiVerdict"]["uncertainty"] = rul_result.get("uncertainty")
            
    except Exception as e:
        print(f"RUL Calculation failed: {e}")

    # 6. Generate Strategic AI Advice
    try:
        # Prepare data for advice agent
        rul_data = {
            "rulEstimate": llm_data.get("aiVerdict", {}).get("rulEstimate", "Unknown"),
            "uncertainty": llm_data.get("aiVerdict", {}).get("uncertainty", "Unknown")
        }
        
        # Ensure recommendations structure is correct
        recs_data = {
            "maintenanceActions": recommendations.get("maintenanceActions", []) if recommendations else [],
            "futureFaultsPdf": recommendations.get("futureFaultsPdf", []) if recommendations else []
        }
        
        # Ensure KPIs structure is correct
        kpis_data = {"kpis": []}
        # Convert flat kpis to list format if needed, or just pass the dict if agent handles it
        # The agent expects {"kpis": [{"name":..., "value":...}]}
        # We can reuse formatted_kpis logic later, but let's do a quick conversion here
        temp_kpis_list = []
        for k, v in kpis.items():
            temp_kpis_list.append({"name": k, "value": v})
        kpis_data["kpis"] = temp_kpis_list

        advice_result = generate_ai_advice(
            rul_data=rul_data,
            recommendations_data=recs_data,
            kpis_data=kpis_data,
            cbhi_score=cbhi_score,
            phase_analysis=phase_analysis_result,
            ai_verdict=llm_data.get("aiVerdict", {})
        )
        
        if "aiAdvice" in advice_result:
            llm_data["aiVerdict"]["aiAdvice"] = advice_result["aiAdvice"]
            
    except Exception as e:
        print(f"AI Advice Generation failed: {e}")

    # 5. Construct Final JSON
    
    # Map KPIs to list format
    # Sample format: {"name": "Main Wipe", "unit": "mm", "value": 15.1}
    # Current kpis dict: {'main_wipe': 15.1, ...}
    kpi_mapping = [
        {"key": "main_wipe", "name": "Main Wipe", "unit": "mm"},
        {"key": "dlro", "name": "DLRO Value", "unit": "µΩ"},
        {"key": "contact_travel", "name": "Contact Travel distance", "unit": "mm"},
        {"key": "arc_wipe", "name": "Arc Wipe", "unit": "mm"},
        {"key": "contact_speed", "name": "Contact Speed", "unit": "m/s"},
        {"key": "sf6_pressure", "name": "SF6 Pressure", "unit": "bar"},
        {"key": "peak_resistance", "name": "Peak Resistance", "unit": "µΩ"},
        {"key": "opening_time", "name": "Opening Time", "unit": "ms"},
        {"key": "closing_time", "name": "Closing Time", "unit": "ms"},
        {"key": "ambient_temp", "name": "Ambient Temp", "unit": "°C"},
        {"key": "peak_close_coil", "name": "Peak Close Coil", "unit": "A"},
        {"key": "peak_trip_coil_1", "name": "Peak Trip Coil 1", "unit": "A"},
        {"key": "peak_trip_coil_2", "name": "Peak Trip Coil 2", "unit": "A"}
    ]
    
    formatted_kpis = []
    for item in kpi_mapping:
        if item["key"] in kpis:
            formatted_kpis.append({
                "name": item["name"],
                "unit": item["unit"],
                "value": kpis[item["key"]]
            })

    # Waveform Data
    # Sample: [{"time": 0.0, "current": 230.0, "resistance": 850.0, "travel": 200.0, "shap": 0.0, "close_coil": 150.0, "trip_coil_1": 80.0, "trip_coil_2": 50.0}, ...]
    # We need to ensure we don't send too much data if the file is huge, but usually DCRM is manageable.
    # We'll downsample if needed or just take what's there.
    waveform_data = []
    # Ensure columns exist
    cols = df.columns
    has_time = 'Time_ms' in cols
    has_cur = 'Current' in cols
    has_res = 'Resistance' in cols
    has_trav = 'Travel' in cols
    has_close_coil = 'Close_Coil' in cols
    has_trip_coil_1 = 'Trip_Coil_1' in cols
    has_trip_coil_2 = 'Trip_Coil_2' in cols
    
    for idx, row in df.iterrows():
        point = {
            "time": row['Time_ms'] if has_time else idx,
            "current": float(row['Current']) if has_cur else 0.0,
            "resistance": float(row['Resistance']) if has_res else 0.0,
            "travel": float(row['Travel']) if has_trav else 0.0,
            "shap": 0.0,  # Placeholder
            "close_coil": float(row['Close_Coil']) if has_close_coil else 0.0,
            "trip_coil_1": float(row['Trip_Coil_1']) if has_trip_coil_1 else 0.0,
            "trip_coil_2": float(row['Trip_Coil_2']) if has_trip_coil_2 else 0.0
        }
        waveform_data.append(point)

    # Merge Recommendations into aiVerdict with proper ordering
    ai_verdict = llm_data.get("aiVerdict", {})
    
    # Create properly ordered aiVerdict structure (keeping related fields together)
    ordered_ai_verdict = {}
    
    # 1. aiAdvice first
    if "aiAdvice" in ai_verdict:
        ordered_ai_verdict["aiAdvice"] = ai_verdict["aiAdvice"]
    
    # 2. futureFaultsPdf second (inside aiVerdict)
    if recommendations:
        ordered_ai_verdict["futureFaultsPdf"] = recommendations.get("futureFaultsPdf", [])
    
    # 3. maintenanceActions third (inside aiVerdict)
    if recommendations:
        ordered_ai_verdict["maintenanceActions"] = recommendations.get("maintenanceActions", [])
    
    # 4. Core verdict metadata - keep these CONSECUTIVE and TOGETHER
    ordered_ai_verdict["confidence"] = ai_verdict.get("confidence", 0)
    ordered_ai_verdict["faultLabel"] = ai_verdict.get("faultLabel", "Unknown")
    ordered_ai_verdict["severity"] = ai_verdict.get("severity", "Unknown")
    ordered_ai_verdict["severityReason"] = ai_verdict.get("severityReason", "")
    ordered_ai_verdict["rulEstimate"] = ai_verdict.get("rulEstimate", "Unknown")
    ordered_ai_verdict["uncertainty"] = ai_verdict.get("uncertainty", "Unknown")
    
    # 5. effectAnalysis if present
    if "effectAnalysis" in ai_verdict:
        ordered_ai_verdict["effectAnalysis"] = ai_verdict["effectAnalysis"]
    
    # 6. topShapContributors last
    if "topShapContributors" in ai_verdict:
        ordered_ai_verdict["topShapContributors"] = ai_verdict["topShapContributors"]
    
    ai_verdict = ordered_ai_verdict
    
    # Final Assembly (removed breakerId, operator, userId, testId fields - only keeping healthScore)
    final_json = {
        "_id": "generated_id_" + datetime.datetime.now().strftime("%Y%m%d%H%M%S"),
        "aiVerdict": ai_verdict,
        "cbhi": {
            "history": [], # Placeholder
            "score": cbhi_score
        },
        "duration": "45 min", # Placeholder
        "findings": ai_verdict.get("faultLabel", "Unknown"),
        "healthScore": cbhi_score,
        "info": {
            "faultCondition": ai_verdict.get("faultLabel", "Unknown"),
            "id": "generated_id_" + datetime.datetime.now().strftime("%Y%m%d%H%M%S"),
            "sf6PressureStatus": "Normal", # Logic could be added
            "temperature": kpis.get("ambient_temp", 25),
            "testDate": datetime.datetime.now().strftime("%d/%m/%Y"),
            "type": "SF6 400kV" # Placeholder
        },
        "kpis": formatted_kpis,
        "phaseWiseAnalysis": phase_wise_analysis,
        "vitResult": vit_result,
        "ruleBased_result": rule_result,
        "status": "completed",
        "testDate": datetime.datetime.now().strftime("%a, %d %b %Y %H:%M:%S GMT"),
        "type": "SF6 400kV",
        "waveform": waveform_data
    }
    
    return final_json

