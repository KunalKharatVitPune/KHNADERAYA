# Previous Name: analysis/agents/ai_agent.py
import os
import sys
import time
import json
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import google.generativeai as genai
import numpy as np

# Set UTF-8 encoding for console output (handles µ, Ω, etc.)
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# =========================
# CONFIGURATION
# =========================
API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=API_KEY)

MODEL_NAME = "gemini-2.5-flash"

BASELINE_OFFSET_UOHM = float(os.getenv("DCRM_BASELINE_UOHM", "0"))
PLOT_PATH = "resistance_vs_time.png"

# =========================
# AGENT 1: PRIMARY CONTACT & COIL DEFECT DETECTOR (Classes 1-5, 11-12)
# =========================
AGENT1_SYSTEM_PROMPT = """
Role: You are a High Voltage Circuit Breaker Contact & Coil Diagnostic AI specialized in Dynamic Contact Resistance Measurement (DCRM).
You must analyze ONLY the Resistance-vs-Time waveform (uOhm) from T_0 to T_400 ms. Green line = Resistance.

Physics Regions (nominal):
- Region 1 (Open): Resistance > 10,000,000 uOhm (effectively open).
- Region 2 (Arcing Make / Left Shoulder): ~3500 uOhm nominal shoulder before main contact engages.
- Region 3 (Main Contact Floor): ~20–40 uOhm nominal, smooth, low noise.
- Region 4 (Arcing Break / Right Shoulder): ~3500 uOhm nominal shoulder after main contact disengages.
- Region 5 (Open): Return to infinite resistance.

in any response dont address regions 1,2,3,4,5 directly as newcomers will not understand these terminologies. instead for them address through below mapping:

- "Pre contact Travel": Region 1
- "Arcing Contact Engagement & Arc Initiation": Region 2
- "Main Contact Conduction": Region 3
- "Main Contact Parting & Arc Elongation": Region 4
- "Final Open Phase": Region 5

Defect classes & strict signatures:
1) Healthy:
   - Region 3 mean ~20–50 uOhm, std low; transitions sharp (sigmoid-like) into and out of Region 3.
   - Regions 2 & 4 stable, without severe spikes.
2) Main Contact Wear:
   - Region 3 is ELEVATED (>100 uOhm after baseline correction) and NOISY (std > 25 uOhm or visible grassy drift).
3) Arcing Contact Wear:
   - Region 3 healthy, but Region 2 and/or 4 exhibit severe upward spikes (>6000 uOhm) or heavy instability.
4) Main Contact Misalignment:
   - Region 3 may show telegraph noise (square-wave-like jumping).
   - Transition 2→3 or 3→4 has a mid-slope step/shelf (non-smooth pause).
5) Arcing Contact Misalignment:
   - Timing asymmetry: Opening shoulder (Region 4) significantly wider/longer than Closing shoulder (Region 2).
   - Rounded bounces on Opening shoulder.
11) Close Coil Damage:
    - Peak Close Coil Current (KPI) is very low (<2A) or zero during a detected closing operation.
    - Normal close coil current: 4-7A for EHV breakers.
12) Trip Coil Damage:
    - BOTH Peak Trip Coil 1 AND Peak Trip Coil 2 currents are very low (<2A) or zero.
    - CRITICAL: At least ONE trip coil must work (>=2A). If Trip Coil 1 OR Trip Coil 2 is working (>=2A), then NO FAULT.
    - Only report fault if BOTH coils fail simultaneously.
    - Normal trip coil current: 4-7A per coil for EHV breakers.

Mandatory analysis steps:
- Compute features from the provided time-series:
  * Region 3 mean, median, std, min, max (approx. nominal window 120–320 ms).
  * Presence and magnitude of spikes in Regions 2 & 4 (e.g., >6000 uOhm).
  * Transition smoothness: detect shelves/steps around 100–120 ms (closing) and 320–340 ms (opening).
- If region boundaries are unclear, state that and lower confidence.
- DO NOT declare a defect unless thresholds are clearly met. Prefer "Healthy" with rationale if ambiguous.
- Consider baseline offset explicitly.
- Severity depends on confidence: above 85% then High, 50-85% then Medium, else Low.
- Confidence depends on how much the signature deviates from healthy and how many feature of a particular defect it satisfies.

Verification Notes (apply before finalizing):
- For Main Contact Wear: Look for consistent, uniform low-high medium to high spikes across entire Region 3 plateau.
- For Main Contact Misalignment: Look for square-shaped (telegraphic) spikes in Region 3, AND a two-or-more-step transition from Region 2 to 3.
- For Arcing Contact Wear: Look for very high up/down spikes in Regions 2 and 4. Spikes are 95% similar on both sides.
- For Arcing Contact Misalignment: Look for square-shaped (telegraphic) or sinusoidal high-frequency noise spikes in Regions 2 and 4. Curve MUST be asymmetric (Region 4 wider/longer than Region 2). Region 3 duration might be reduced.

Output: Return ONLY valid JSON (no markdown code fences). Structure:
{
  "Fault_Detection": [
    {
      "defect_name": "Exact Class Name or 'Healthy'",
      "Confidence": "XX.XX %",
      "Severity": "Low/Medium/High",
      "description": "1–2 short sentences citing quantified features and KPIs.",
    }
  ],
  "primary_defect_class": "1-5 or 11-12 or Healthy"
}
List ONLY the most likely primary defect. If healthy, return only Healthy.
"""

# =========================
# AGENT 2: SECONDARY MECHANICAL/OPERATIONAL DEFECT DETECTOR (Classes 6-10)
# =========================
AGENT2_SYSTEM_PROMPT = """
Role: You are a Circuit Breaker Mechanical & Operational Diagnostic AI. You analyze DCRM waveforms and KPIs for secondary mechanical/operational issues with EXTREME STRICTNESS.

You will receive:
1. DCRM waveform data (Resistance vs Time, T_0 to T_400 ms)
2. KPI values with industry-standard nominal ranges
3. PRIMARY DEFECT from Agent 1 (Classes 1-5, 11-12)

Your task: Detect ONLY Classes 6-10 defects with OVERWHELMING EVIDENCE. Confidence must be >75%.

=== DEFECT CLASSES WITH PHYSICS-BASED SIGNATURES ===

6) Operating Mechanism Malfunction (Slow/Fast Operation)
   Physical Basis:
   - Operating mechanism (spring, hydraulic, pneumatic) drives contacts. Weak springs, hydraulic leaks, sticky linkages affect speed.
   
   Industry Standard KPI Ranges (Ministry of Power / POWERGRID norms for EHV breakers):
   - Closing Time: 80-100 ms (nominal)
   - Opening Time: 30-40 ms (nominal)
   - Contact Speed: 4.5-6.5 m/s (nominal)
   
   DCRM Curve Manifestation:
   - **Horizontal Shift**: ENTIRE curve shifts left (fast) or right (slow) compared to nominal timeline.
   - **Slope Changes**: Resistance transitions (Region 2->3, Region 3->4) have altered steepness across ENTIRE operation.
   - **Delayed Closure/Opening**: Time from first contact movement to full closure/opening is >20% off nominal.
   
   Detection Criteria (STRICT):
   - Closing Time >120 ms OR <64 ms (>20% deviation from 80-100 ms range)
   - Opening Time >48 ms OR <24 ms (>20% deviation from 30-40 ms range)
   - Contact Speed <3.6 m/s OR >7.8 m/s (>20% deviation from 4.5-6.5 m/s)
   - AND waveform shows consistent timing shift across ALL phases
   
   Confidence Requirements:
   - 90-95%: If 2+ KPIs exceed threshold by >25%
   - 80-89%: If 1 KPI exceeds threshold by >20%
   - <80%: Insufficient evidence, DO NOT report
   also most notable point in it see region 1 and 5, if there are spikes in this region lines are not straight at high resistance then 100% its operating mechanism malfunction

7) Damping System Fault (Excessive Bouncing/Oscillation)
   Physical Basis:
   - Dampers (dashpots, hydraulic dampers) absorb kinetic energy, prevent contact rebound. Failure causes persistent oscillation.
   
   DCRM Curve Manifestation:
   - **Excessive Bounces at Main Contact Closure**: Multiple rapid spikes (>5 distinct bounces) back towards arcing resistance AFTER initial closure.
   - **Prolonged Oscillation in Region 3**: Main contact plateau shows persistent fluctuations (decaying sinusoidal pattern, NOT random noise).
   - **Secondary Bounces at Opening**: Clean separation disrupted by re-closure spikes.
   
   Detection Criteria (VERY STRICT):
   - >5 distinct bounces with amplitude >100 µΩ in Region 3 (120-320 ms window)
   - Bounce pattern shows decaying oscillation (each bounce smaller than previous)
   - NOT simple noise or wear (std in Region 3 must show structured oscillation, not random)
   
   Confidence Requirements:
   - 85-95%: Clear decaying oscillation pattern with >7 bounces
   - 75-84%: 5-6 distinct bounces with structured pattern
   - <75%: Could be noise/wear, DO NOT report

8) Pressure System Leakage (SF6 Gas Chamber)
   Physical Basis:
   - SF6 gas provides insulation and arc quenching. Leak reduces pressure, weakens dielectric strength, prolongs arc.
   
   Industry Standard:
   - SF6 Pressure: 5.5-6.5 bar (at 20°C) for typical EHV breakers
   
   DCRM Curve Manifestation:
   - **Prolonged Arc-Quenching**: Region 4->5 transition (arcing break to open) takes >20 ms (nominal: 10-15 ms).
   - **Less Sharp Opening**: Opening slope less steep due to extended arc duration.
   - **Higher Peak Arcing Resistance**: Arc resistance >5000 µΩ sustained for longer duration.
   
   Detection Criteria (EXTREMELY STRICT):
   - Requires external SF6 Pressure KPI <5.0 bar
   - OR (if no pressure KPI): Arc-quenching duration >25 ms AND primary defect is Arcing Contact Wear with confidence >85%
   - WITHOUT pressure KPI, confidence CANNOT exceed 70%
   
   Confidence Requirements:
   - 85-95%: SF6 Pressure KPI <4.5 bar AND prolonged arc
   - 70-84%: No pressure KPI but severe arc prolongation (>30 ms) AND severe Arcing Wear
   - <70%: Insufficient evidence, DO NOT report

9) Linkage/Connecting Rod Obstruction/Damage
   Physical Basis:
   - Operating rod and linkages transmit force from mechanism to contacts. Obstruction, bending, friction, or looseness impede smooth movement.
   
   DCRM Curve Manifestation:
   - **"Stutter" or "Hesitation"**: Momentary pauses, jerks, or sudden slope changes within smooth transitions (Region 2->3, Region 3->4).
   - **Square Plateaus Within Slopes**: Small flat sections (20-50 ms duration) interrupting the sigmoid curve, indicating momentary stuck movement.
   - **Sudden Speed Changes**: Abrupt changes in transition rate (detected as derivative discontinuities).
   - **High-Frequency Mechanical Noise**: Increased jitter/vibration during transitions (NOT smooth noise).
   
   Detection Criteria (VERY STRICT):
   - >3 distinct "stutters" (flat plateaus >10 ms within transition slopes)
   - AND increased operating time (>10% longer than nominal)
   - Stutter signature must be VERY distinct (resistance plateaus for >10 ms, then sudden drops/rises)
   
   Confidence Requirements:
   - 85-95%: >5 distinct stutters with clear mechanical impedance pattern
   - 75-84%: 3-4 stutters AND operating time >15% off nominal
   - <75%: Could be normal transition, DO NOT report

10) Fixed Contact Damage/Deformation
    Physical Basis:
    - Fixed contacts can be damaged (scoring, pitting, bending, high connection resistance). Distinct from moving contact wear.
    
    Industry Standard:
    - DLRO Value (Dynamic Low Resistance Ohmmeter): <50 µΩ for healthy EHV breaker
    - Acceptable Range: 50-80 µΩ (moderate concern)
    - Critical: >80 µΩ (indicates fixed contact or connection issues)
    
    DCRM Curve Manifestation:
    - **Elevated Baseline Resistance**: Main Contact Plateau (Region 3) consistently higher (>80 µΩ) even when curve shape is smooth.
    - **Low Variability**: Region 3 std <15 µΩ (smooth plateau, NOT noisy like Main Contact Wear).
    - **Asymmetric Wear Pattern**: Closing and opening transitions show different resistance levels (fixed contact deformation).
    - **Localized Spikes in Plateau**: Specific recurring spikes at same time points (damaged spot on fixed contact).
    
    Detection Criteria (STRICT):
    - DLRO Value >80 µΩ (from KPI)
    - AND Region 3 mean >80 µΩ with std <15 µΩ (smooth but elevated)
    - AND Primary defect is NOT Main Contact Wear (avoid duplication)
    - IF Primary is Main Contact Wear: Can list as SECONDARY but with confidence <70%
    
    Confidence Requirements:
    - 80-90%: DLRO >100 µΩ AND smooth curve (std <10 µΩ)
    - 70-79%: DLRO 80-100 µΩ AND smooth curve
    - 50-69%: Listed as secondary to Main Contact Wear
    - <50%: Primary defect already explains, DO NOT report

=== CRITICAL RULES ===
1. BE EXTREMELY STRICT. Only report if confidence >75% AND evidence is OVERWHELMING.
2. If Primary defect (from Agent 1) already explains the waveform, DO NOT add redundant secondary defects.
3. Check ALL thresholds and criteria. If ANY criterion is not met, DO NOT report that defect.
4. For Class 10 (Fixed Contact): Only if DLRO >80 µΩ AND std <15 µΩ. If Main Contact Wear is primary, list as secondary with reduced confidence.
5. If no secondary defect meets strict criteria, return: "No Secondary Defect Detected"
6. Confidence for Classes 6-10 should be 75-90% range unless extreme deviations (>95% rare).
7. Compare waveform timing against nominal windows: Region 2 (85-105 ms), Region 3 (120-300 ms), Region 4 (315-335 ms).

Output: Return ONLY valid JSON (no markdown):
{
  "Secondary_Fault_Detection": [
    {
      "defect_name": "Exact Class Name or 'No Secondary Defect Detected'",
      "Confidence": "XX.XX %",
      "Severity": "Low/Medium/High",
      "description": "1–2 short sentences citing quantified features and KPIs.",
    }
  ]
}
List at most 1-2 secondary defects. If uncertain or evidence weak, return "No Secondary Defect Detected".
"""

def parse_kpis_from_json(kpis_json):
    """
    Convert KPI JSON format to dictionary for internal use.
    
    Input format:
    {
        "kpis": [
            {"name": "Closing Time", "unit": "ms", "value": 87.8},
            ...
        ]
    }
    
    Output format:
    {
        "Closing Time (ms)": 87.8,
        ...
    }
    """
    if kpis_json is None:
        return {}
    
    # If already in dict format, return as-is
    if isinstance(kpis_json, dict) and "kpis" not in kpis_json:
        return kpis_json
    
    # Parse from JSON format
    kpi_dict = {}
    kpis_list = kpis_json.get("kpis", [])
    
    for kpi in kpis_list:
        name = kpi.get("name", "")
        unit = kpi.get("unit", "")
        value = kpi.get("value", None)
        
        # Create key in format "Name (unit)"
        key = f"{name} ({unit})" if unit else name
        kpi_dict[key] = value
    
    return kpi_dict


def standardize_input(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame with one row and columns T_0...T_400 containing Resistance values (uOhm).
    """
    if 'Resistance' not in df.columns:
        potential_resistance_cols = [c for c in df.columns if not c.startswith('T_')]
        if not potential_resistance_cols:
             raise KeyError("CSV must contain a 'Resistance' column.")
        
        if len(potential_resistance_cols) > 1:
            print(f"Warning: Multiple non-T_ columns found. Using first one as 'Resistance'.")
        
        df = df.rename(columns={potential_resistance_cols[0]: 'Resistance'})

    df = df[['Resistance']]

    if df.shape[0] >= 401 and df.shape[1] == 1:
        values = df.iloc[:401, 0].values.reshape(1, -1)
        cols = [f"T_{i}" for i in range(401)]
        return pd.DataFrame(values, columns=cols)

    elif df.shape[1] >= 401:
        df = df.iloc[:, :401]
        df.columns = [f"T_{i}" for i in range(401)]
        return df

    else:
        raise ValueError(f"Input shape {df.shape} invalid. Expected 401 Resistance points.")

def plot_resistance(row_values, save_path=PLOT_PATH):
    """Saves a line plot Resistance (uOhm) vs Time (ms)."""
    t = list(range(401))
    plt.figure(figsize=(10, 4.2), dpi=150)
    plt.plot(t, row_values, color="green", linewidth=1.6, label="Resistance (uΩ)")
    plt.title("Dynamic Resistance vs Time (DCRM)")
    plt.xlabel("Time (ms)")
    plt.ylabel("Resistance (μΩ)")
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 400)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def validate_trip_coil_logic(kpis):
    """
    Validates Trip Coil logic: At least ONE trip coil must work.
    Returns (is_fault, description)
    """
    trip_coil_1 = kpis.get("Peak Trip Coil 1 Current (A)", 0)
    trip_coil_2 = kpis.get("Peak Trip Coil 2 Current (A)", 0)
    
    # At least ONE must be >= 2A (working)
    if trip_coil_1 >= 2.0 or trip_coil_2 >= 2.0:
        return False, f"Trip coils functional (Coil 1: {trip_coil_1}A, Coil 2: {trip_coil_2}A)"
    
    # BOTH are below 2A = FAULT
    return True, f"BOTH trip coils failed (Coil 1: {trip_coil_1}A, Coil 2: {trip_coil_2}A - both below 2A threshold)"

def build_agent1_prompt(row_values, baseline_offset_uohm, kpis):
    """Build prompt for Agent 1 (Primary Contact & Coil Defects)."""
    data_str = ",".join(map(str, row_values))
    
    # Parse KPIs if in JSON format
    kpis_dict = parse_kpis_from_json(kpis)
    
    kpi_str = "\n".join([f"- {name}: {value}" for name, value in kpis_dict.items()])
    
    # Validate Trip Coil logic
    trip_fault, trip_msg = validate_trip_coil_logic(kpis_dict)

    prompt = f"""
Analyze this DCRM waveform for PRIMARY CONTACT & COIL DEFECTS (Classes 1-5, 11-12).

DCRM Data:
- Array length: {len(row_values)} (T_0 to T_400 ms)
- Baseline offset: {baseline_offset_uohm} uOhm
- Attached image: Resistance vs Time plot (green line)

KPIs:
{kpi_str}

TRIP COIL PRE-VALIDATION:
{trip_msg}
{"⚠️ REPORT TRIP COIL DAMAGE (Class 12)" if trip_fault else "✓ Trip coils are functional - DO NOT report Trip Coil Damage"}

Tasks:
1) Compute Region 3 features (mean, std, spikes).
2) Analyze Regions 2 & 4 for arcing issues.
3) Check Close Coil Current (normal: 4-7A, fault if <2A).
4) For Trip Coils: {"Report Class 12 (Trip Coil Damage)" if trip_fault else "DO NOT report Class 12 - at least one coil is working"}.
5) Classify as one of: Healthy, Main Contact Wear, Arcing Contact Wear, Main Contact Misalignment, Arcing Contact Misalignment, Close Coil Damage, {"Trip Coil Damage" if trip_fault else "(NOT Trip Coil Damage)"}.
6) Return ONLY the PRIMARY defect with highest confidence.

Data array (uOhm):
[{data_str}]
"""
    return prompt

def build_agent2_prompt(row_values, baseline_offset_uohm, kpis, agent1_result):
    """Build prompt for Agent 2 (Secondary Mechanical Defects)."""
    data_str = ",".join(map(str, row_values))
    
    # Parse KPIs if in JSON format
    kpis_dict = parse_kpis_from_json(kpis)
    
    kpi_str = "\n".join([f"- {name}: {value}" for name, value in kpis_dict.items()])
    
    agent1_summary = json.dumps(agent1_result.get("Fault_Detection", []), indent=2)
    primary_class = agent1_result.get("primary_defect_class", "Unknown")

    prompt = f"""
Analyze this DCRM waveform for SECONDARY MECHANICAL/OPERATIONAL DEFECTS (Classes 6-10).

PRIMARY DEFECT (from Agent 1):
Class: {primary_class}
Details:
{agent1_summary}

DCRM Data:
- Array length: {len(row_values)} (T_0 to T_400 ms)
- Baseline offset: {baseline_offset_uohm} uOhm
- Attached image: Resistance vs Time plot (green line)

KPIs with Industry Standards:
{kpi_str}

Industry Standard Nominal Ranges (Ministry of Power / POWERGRID norms for EHV breakers):
- Closing Time: 80-100 ms (acceptable range)
- Opening Time: 30-40 ms (acceptable range)
- Contact Speed: 4.5-6.5 m/s (acceptable range)
- DLRO Value: <50 µΩ (healthy), 50-80 µΩ (moderate), >80 µΩ (critical)
- Close Coil Current: 4-7A (nominal)
- Trip Coil Current: 4-7A per coil (nominal)
- SF6 Pressure: 5.5-6.5 bar at 20°C (if applicable)

Tasks:
1) BE EXTREMELY STRICT. Only report if confidence >75% AND evidence is OVERWHELMING.
2) For each Class 6-10, check ALL detection criteria and thresholds listed in system instructions.
3) Calculate % deviation from nominal ranges for timing KPIs.
4) Analyze waveform for physics-based signatures:
   - Class 6: Horizontal shift of ENTIRE curve, altered transition slopes
   - Class 7: >5 distinct decaying bounces in Region 3
   - Class 8: Prolonged arc-quenching (Region 4->5 >25 ms)
   - Class 9: >3 distinct stutters (flat plateaus within slopes)
   - Class 10: DLRO >80 µΩ AND Region 3 std <15 µΩ
5) Do NOT duplicate primary defect explanations.
6) If ALL criteria are not met for a defect, DO NOT report it.
7) If uncertain or evidence weak, return "No Secondary Defect Detected".

Expected Region Timing (nominal):
- Region 1 (Open before): 0-85 ms
- Region 2 (Arcing Make): 85-105 ms
- Region 3 (Main Contact): 120-300 ms
- Region 4 (Arcing Break): 315-335 ms
- Region 5 (Open after): 335-400 ms

Data array (uOhm):
[{data_str}]
"""
    return prompt

def call_agent1(row_values, kpis_data):
    """Agent 1: Primary Contact & Coil Defect Detection."""
    plot_resistance(row_values, PLOT_PATH)
    file = genai.upload_file(path=PLOT_PATH)
    
    prompt = build_agent1_prompt(row_values, BASELINE_OFFSET_UOHM, kpis_data)
    
    model = genai.GenerativeModel(MODEL_NAME, system_instruction=AGENT1_SYSTEM_PROMPT)
    response = model.generate_content([prompt, file])
    
    text = (response.text or "").strip()
    text = text.replace("```json", "").replace("```", "").strip()
    
    try:
        obj = json.loads(text)
        return obj
    except Exception as e:
        return {"Error": f"Agent 1 failed. {e}", "raw_response": text}

def call_agent2(row_values, kpis_data, agent1_result):
    """Agent 2: Secondary Mechanical Defect Detection (STRICT)."""
    prompt = build_agent2_prompt(row_values, BASELINE_OFFSET_UOHM, kpis_data, agent1_result)
    
    # Reuse same image with error handling
    try:
        file = genai.upload_file(path=PLOT_PATH)
    except Exception as upload_error:
        print(f"⚠️ Agent 2: File upload failed: {upload_error}. Skipping Agent 2.")
        return {"Error": f"File upload failed: {upload_error}", "agent2_skipped": True}
    
    try:
        model = genai.GenerativeModel(MODEL_NAME, system_instruction=AGENT2_SYSTEM_PROMPT)
        response = model.generate_content([prompt, file])
        
        # Check if response has valid content
        if not response.text or response.text.strip() == "":
            print(f"⚠️ Agent 2: Empty response. Finish reason: {response.candidates[0].finish_reason if response.candidates else 'Unknown'}")
            return {"Error": "Agent 2 returned empty response", "finish_reason": response.candidates[0].finish_reason if response.candidates else None}
        
        text = response.text.strip()
        text = text.replace("```json", "").replace("```", "").strip()
        
        try:
            obj = json.loads(text)
            return obj
        except Exception as e:
            return {"Error": f"Agent 2 JSON parse failed. {e}", "raw_response": text}
            
    except Exception as e:
        print(f"⚠️ Agent 2: Generation failed: {e}")
        return {"Error": f"Agent 2 generation failed: {e}", "agent2_failed": True}

def merge_results(agent1_result, agent2_result):
    """Merge Agent 1 and Agent 2 results into final comprehensive report."""
    final_report = {
        "Fault_Detection": [],
        "overall_health_assessment": {
            "Contacts (moving & arcing)": "Normal",
            "SF6 Gas Chamber": "Normal",
            "Operating Mechanism": "Normal",
            "Coil": "Normal"
        }
    }
    
    # Add Agent 1 primary defects
    if "Fault_Detection" in agent1_result:
        final_report["Fault_Detection"].extend(agent1_result["Fault_Detection"])
    
    # Add Agent 2 secondary defects (only if Agent 2 succeeded and not "No Secondary Defect Detected")
    if "Error" not in agent2_result and "Secondary_Fault_Detection" in agent2_result:
        for defect in agent2_result["Secondary_Fault_Detection"]:
            if defect.get("defect_name", "").lower() != "no secondary defect detected":
                final_report["Fault_Detection"].append(defect)
    elif "Error" in agent2_result:
        print(f"⚠️ Agent 2 failed, using only Agent 1 results: {agent2_result.get('Error')}")
    
    # Assess overall health based on detected defects
    for defect in final_report["Fault_Detection"]:
        name = defect.get("defect_name", "").lower()
        severity = defect.get("Severity", "").lower()
        confidence = float(defect.get("Confidence", "0").replace("%", "").strip())
        
        # Determine risk level
        if confidence >= 85 and severity == "high":
            risk = "High Risk"
        elif confidence >= 50:
            risk = "Moderate Risk"
        else:
            risk = "Normal"
        
        # Map to health categories
        if any(x in name for x in ["main contact", "arcing contact", "contact wear", "contact misalignment"]):
            if final_report["overall_health_assessment"]["Contacts (moving & arcing)"] != "High Risk":
                final_report["overall_health_assessment"]["Contacts (moving & arcing)"] = risk
        
        if "sf6" in name or "pressure" in name:
            if final_report["overall_health_assessment"]["SF6 Gas Chamber"] != "High Risk":
                final_report["overall_health_assessment"]["SF6 Gas Chamber"] = risk
        
        if any(x in name for x in ["operating mechanism", "damping", "linkage", "rod"]):
            if final_report["overall_health_assessment"]["Operating Mechanism"] != "High Risk":
                final_report["overall_health_assessment"]["Operating Mechanism"] = risk
        
        if "coil" in name:
            if final_report["overall_health_assessment"]["Coil"] != "High Risk":
                final_report["overall_health_assessment"]["Coil"] = risk
    
    return final_report

# def main():
#     df = pd.read_csv("df3_final.csv")
#     df = standardize_input(df)

#     # Analyze all rows if small DF; otherwise sample
#     indices = df.index if len(df) <= 3 else df.sample(3, random_state=42).index

#     time_cols = [c for c in df.columns if c.startswith('T_')]
#     for idx in indices:
#         row_values_raw = df.loc[idx, time_cols].values.tolist()

#         # Sample KPIs in new JSON format (based on industry standards and Ministry of Power / POWERGRID norms)
#         sample_kpis = {
#             "kpis": [
#                 {
#                     "name": "Closing Time",
#                     "unit": "ms",
#                     "value": 103.5
#                 },
#                 {
#                     "name": "Opening Time",
#                     "unit": "ms",
#                     "value": 37.0
#                 },
#                 {
#                     "name": "DLRO Value",
#                     "unit": "µΩ",
#                     "value": 299.93
#                 },
#                 {
#                     "name": "Peak Resistance",
#                     "unit": "µΩ",
#                     "value": 408.0
#                 },
#                 {
#                     "name": "Contact Travel Distance",
#                     "unit": "mm",
#                     "value": 550.0
#                 },
#                 {
#                     "name": "Main Wipe",
#                     "unit": "mm",
#                     "value": 46.0
#                 },
#                 {
#                     "name": "Arc Wipe",
#                     "unit": "mm",
#                     "value": 63.0
#                 },
#                 {
#                     "name": "Contact Speed",
#                     "unit": "m/s",
#                     "value": 5.2
#                 },
#                 {
#                     "name": "Peak Close Coil Current",
#                     "unit": "A",
#                     "value": 5.5
#                 },
#                 {
#                     "name": "Peak Trip Coil 1 Current",
#                     "unit": "A",
#                     "value": 5.5
#                 },
#                 {
#                     "name": "Peak Trip Coil 2 Current",
#                     "unit": "A",
#                     "value": 0.0
#                 },
#                 {
#                     "name": "Ambient Temperature",
#                     "unit": "°C",
#                     "value": 28.4
#                 }
#             ]
#         }

#         # STEP 1: Call Agent 1 (Primary Defects)
#         agent1_result = call_agent1(row_values_raw, sample_kpis)
        
#         # STEP 2: Call Agent 2 (Secondary Defects) - runs in parallel after Agent 1
#         agent2_result = call_agent2(row_values_raw, sample_kpis, agent1_result)
        
#         # STEP 3: Merge Results and Print ONLY Final JSON
#         final_report = merge_results(agent1_result, agent2_result)
        
#         # Print with proper Unicode handling (ensure_ascii=False prevents \u00b5 escape sequences)
#         print(json.dumps(final_report, indent=2, ensure_ascii=False))
#         print()  # Empty line separator between rows





def detect_fault(df,sample_kpis):
    df = standardize_input(df)

    # Analyze all rows if small DF; otherwise sample
    indices = df.index if len(df) <= 3 else df.sample(3, random_state=42).index

    time_cols = [c for c in df.columns if c.startswith('T_')]
    for idx in indices:
        row_values_raw = df.loc[idx, time_cols].values.tolist()

        # STEP 1: Call Agent 1 (Primary Defects)
        agent1_result = call_agent1(row_values_raw, sample_kpis)
        
        # STEP 2: Call Agent 2 (Secondary Defects) - runs in parallel after Agent 1
        agent2_result = call_agent2(row_values_raw, sample_kpis, agent1_result)
        
        # STEP 3: Merge Results and Print ONLY Final JSON
        final_report = merge_results(agent1_result, agent2_result)
        
        return final_report



if __name__ == "__main__":
    df = pd.read_csv("df3_final.csv")
    # Sample KPIs in new JSON format (based on industry standards and Ministry of Power / POWERGRID norms)
    sample_kpis = {
            "kpis": [
                {
                    "name": "Closing Time",
                    "unit": "ms",
                    "value": 103.5
                },
                {
                    "name": "Opening Time",
                    "unit": "ms",
                    "value": 37.0
                },
                {
                    "name": "DLRO Value",
                    "unit": "µΩ",
                    "value": 299.93
                },
                {
                    "name": "Peak Resistance",
                    "unit": "µΩ",
                    "value": 408.0
                },
                {
                    "name": "Contact Travel Distance",
                    "unit": "mm",
                    "value": 550.0
                },
                {
                    "name": "Main Wipe",
                    "unit": "mm",
                    "value": 46.0
                },
                {
                    "name": "Arc Wipe",
                    "unit": "mm",
                    "value": 63.0
                },
                {
                    "name": "Contact Speed",
                    "unit": "m/s",
                    "value": 5.2
                },
                {
                    "name": "Peak Close Coil Current",
                    "unit": "A",
                    "value": 5.5
                },
                {
                    "name": "Peak Trip Coil 1 Current",
                    "unit": "A",
                    "value": 5.5
                },
                {
                    "name": "Peak Trip Coil 2 Current",
                    "unit": "A",
                    "value": 0.0
                },
                {
                    "name": "Ambient Temperature",
                    "unit": "°C",
                    "value": 28.4
                }
            ]
        }

    result = detect_fault(df,sample_kpis)
    print(json.dumps(result, indent=2, ensure_ascii=False))
