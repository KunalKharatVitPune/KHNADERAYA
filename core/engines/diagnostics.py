# Previous Name: analysis/advanced_diagnostics.py
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
import json
from langchain_core.messages import HumanMessage
import streamlit as st

def get_defect_prompt(data_str):
    return f"""
    System Role: Principal DCRM & Kinematic Analyst
    Role:
    You are an expert High-Voltage Circuit Breaker Diagnostician. Your task is to interpret Dynamic Contact Resistance (DCRM) traces to detect specific electrical and mechanical faults.
    
    Critical "Anti-Overfitting" Directive:
    You must distinguish between Systematic Defects and Artifacts.
    True Degradation: Flag issues only when the visual signature is statistically significant and exceeds the "noise floor."
    
    Capability:
    Identify Multiple Concurrent Issues if present. (e.g., A breaker can have both misalignment and contact wear).
    there will mostly be 3 line charts in the input
    green resistance profile
    blue current profile 
    red travel profile
    
    Data (Sampled): {data_str}
    
    1. Diagnostic Heuristics & Defect Taxonomy
    Map the visual DCRM trace to ONLY the following defect types. Use the specific Visual Heuristics to confirm detection.
    
    Defect Type | Visual Heuristic (The "Hint") | Mechanical Significance (Root Cause)
    --- | --- | ---
    Main Contact Issue (Corrosion/Oxidation) | "The Significant Grass"<br>In the fully closed plateau, look for pronounced, erratic instability. <br>• Ignore: Uniform, low-amplitude fuzz (sensor noise).<br>• Flag: Jagged, irregular peaks/valleys with significant amplitude (e.g., > 15–20 μΩ variance). The trace looks like a "rough rocky road," not just a "gravel path." | Surface Pathology: The Silver (Ag) plating is compromised (fretting corrosion) or heavy oxidation has occurred. The current path is constantly shifting through microscopic non-conductive spots.
    Arcing Contact Wear | "Big Spikes & Short Wipe"<br>Resistance spikes are frequent and significantly large (high amplitude). Crucially, the duration of the arcing zone (the time between first touch and main contact touch) is noticeably shorter than expected. | Ablation: The Tungsten-Copper (W-Cu) tips are heavily eroded. The contact length has physically diminished, risking failure to commutate current during opening.
    Misalignment (Main) | "The Struggle to Settle"<br>There are significant, high-amplitude peaks just before the trace tries to settle into the stable plateau. These are not bounces; they are "struggles" to mate that persist longer than 3-5ms. | Mechanical Centering: The moving contact pin is hitting the side or edge of the stationary rosette fingers before forcing its way in. Caused by loose nuts, kinematic play, or guide ring failure.
    Misalignment (Arcing) | "Rough Entry"<br>Erratic resistance spikes occurring specifically during the initial entry (commutation), well before the main contacts engage. | Tip Eccentricity: The arcing pin is not entering the nozzle concentrically. It is scraping the nozzle throat or hitting the side, indicating a bent rod or skewed interrupter.
    Slow Mechanism | "Stretched Time"<br>The entire resistance profile is elongated along the X-axis. Events happen later than normal. | Energy Starvation: Low spring charge, hydraulic pressure loss, or high friction due to hardened grease in the linkage.
    
    2. Analysis Logic (The "Signal-to-Noise" Filter)
    Before declaring a defect, run these logic checks:
    The "Noise Floor" Test (For Main Contacts):
    Is the plateau variance uniform and small (< 10 μΩ)? -> Classify as Healthy (Sensor/Manufacturing artifact).
    Is the variance erratic, jagged, and large (> 15 μΩ)? -> Classify as Corrosion/Oxidation.
    The "Duration" Test (For Misalignment):
    Are the pre-plateau peaks < 2ms? -> Ignore (Benign Bounce).
    Do the peaks persist > 3-5ms before settling? -> Classify as Misalignment.
    The "Combination" Check:
    Does the trace show both "Rough Entry" AND "Stretched Time"? -> Report Both (Misalignment + Slow Mechanism).
    
    3. Output Structure
    Provide a concise Executive Lead followed by the JSON.
    
    Executive Lead (3-4 Lines)
    Status: Healthy | Warning | Critical.
    Key Findings: Summary of valid defects found (ignoring sensor noise).
    Action: "Return to service" or specific repair instruction.
    
    JSON Schema
    Return ONLY this JSON object:
    {{
      "image_url": "string",
      "overall_condition": "Healthy|Warning|Critical",
      "executive_lead": "string (The 3-4 line summary)",
      "detected_issues": [
        {{
          "issue_type": "Main Contact Issue (Corrosion/Oxidation)|Arcing Contact Wear|Misalignment (Main)|Misalignment (Arcing)|Slow Mechanism",
          "confidence": "High|Medium|Low",
          "visual_evidence": "string (e.g., 'Plateau instability >20 micro-ohms detected, exceeding sensor noise threshold.')",
          "mechanical_significance": "string (Root cause from table)",
          "severity": "Low|Medium|High"
        }}
      ],
      "analysis_metrics": {{
        "static_resistance_Rp_uOhm": "float",
        "signal_noise_level": "Low (Sensor/Mfg)|High (Defect)",
        "wipe_quality": "Normal|Short|Erratic"
      }},
      "maintenance_recommendation": "string"
    }}
    """

def perform_defect_analysis(df, llm):
    """
    Performs advanced defect analysis using the specialized DCRM prompt.
    """
    try:
        # 1. Prepare Text Data
        data_str = df.to_string(index=False)
        prompt_text = get_defect_prompt(data_str)

        # 2. Prepare Image Data (Simplified for LLM Vision)
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=df['Time_ms'], y=df['Current'], name="Current", line=dict(color='blue')), secondary_y=False)
        fig.add_trace(go.Scatter(x=df['Time_ms'], y=df['Resistance'], name="Resistance", line=dict(color='green')), secondary_y=False)
        fig.add_trace(go.Scatter(x=df['Time_ms'], y=df['Travel'], name="Travel", line=dict(color='red')), secondary_y=True)
        fig.update_layout(title="DCRM Graph for Defect Analysis", showlegend=True)
        
        # Convert plot to image bytes
        img_bytes = fig.to_image(format="png", width=1024, height=600)
        base64_image = base64.b64encode(img_bytes).decode('utf-8')

        # 3. Construct Multimodal Message
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt_text},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                }
            ]
        )

        # 4. Invoke LLM
        response = llm.invoke([message])
        content = response.content.replace("```json", "").replace("```", "").strip()
        
        # Extract JSON part if there's extra text
        try:
            start_idx = content.find("{")
            end_idx = content.rfind("}") + 1
            if start_idx != -1 and end_idx != -1:
                json_str = content[start_idx:end_idx]
                result = json.loads(json_str)
            else:
                raise ValueError("No JSON found in response")
        except json.JSONDecodeError:
             # Fallback: try to parse the whole content
             result = json.loads(content)

        return result

    except Exception as e:
        st.error(f"Defect Analysis failed: {str(e)}")
        return {
            "overall_condition": "Unknown",
            "executive_lead": f"Analysis failed due to error: {str(e)}",
            "detected_issues": [],
            "analysis_metrics": {},
            "maintenance_recommendation": "Check system logs."
        }
