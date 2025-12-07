# Previous Name: analysis/agents/recommendation_agent.py
import json
import google.generativeai as genai

# Configure Gemini API
API_KEY = "AIzaSyDSGda4-5VLmd-y09K6sBfHqoqk1QUL6Xo"
genai.configure(api_key=API_KEY)
MODEL_NAME = "gemini-2.5-flash"

# =========================
# DEEP DOMAIN KNOWLEDGE PROMPT
# =========================
RECOMMENDATIONS_PROMPT = """
Role: You are an expert High Voltage Circuit Breaker Maintenance Engineer with 25+ years of experience in DCRM (Dynamic Contact Resistance Measurement) diagnostics, predictive maintenance, and failure analysis for SF6 circuit breakers in EHV (220 kV and onwards) substations.

Your Task: Based on comprehensive circuit breaker diagnostic data, provide:
1. **Maintenance Recommendations** (Priority 1, 2, 3)
2. **Future Fault Predictions** (minimum 3, with probabilities and timelines)

===== DOMAIN KNOWLEDGE BASE =====

**Circuit Breaker Defects (12 Classes) - Deep Knowledge Base:**

1. **Healthy**: All parameters within spec, DLRO 20-50µΩ, smooth phase transitions, stable contact pressure, no timing deviations

2. **Main Contact Wear**: 
   - DLRO >100µΩ (healthy: 20-70µΩ), elevated plateau resistance during Phase 3
   - Surface erosion, material loss, rough contact surfaces with "grassy noise" signature
   - Causes: Thermal stress, mechanical friction, oxidation, improper contact pressure
   - Progression: Early (70-100µΩ) → Moderate (100-180µΩ) → Severe (180-280µΩ) → Critical (>280µΩ)
   - Risks: Overheating → Contact welding → Failed interruption → Explosion hazard

3. **Arcing Contact Wear**: 
   - High-amplitude spikes during Phase 2/4 arcing zones (>5000µΩ), sustained arc duration
   - Contact pitting, erosion, craters from arc flash, prolonged arcing time
   - Causes: Excessive fault interruptions, poor SF6 quality, contact misalignment
   - Risks: Arc flash → Main contact damage → Failed interruption → Catastrophic failure

4. **Main Contact Misalignment**: 
   - "Telegraph pattern" with square-wave jumps (>120µΩ steps), uneven contact engagement
   - Stepped resistance plateaus, asymmetric contact pressure distribution
   - Causes: Mechanical wear, linkage looseness, contact holder deformation, improper assembly
   - Risks: Localized heating → Uneven wear → Contact welding → Mechanical seizure

5. **Arcing Contact Misalignment**: 
   - Asymmetric Phase 2 vs Phase 4 durations (ratio >1.6), sinusoidal bounce patterns
   - One contact engages before the other, timing mismatch between poles
   - Causes: Linkage rod bending, pole asynchrony, damping system failure
   - Risks: Unbalanced arcing → Accelerated wear → Phase-to-phase timing errors

6. **Operating Mechanism Malfunction**: 
   - Abnormal closing/opening times (>20% deviation), reduced contact speed (<4.5 m/s)
   - Sluggish operation, stutter patterns, inconsistent travel curves
   - Causes: Lubrication degradation, bearing wear, spring fatigue, control linkage issues
   - Risks: Timing drift → Failed synchronization → Protection coordination loss

7. **Damping System Fault**: 
   - Contact bounce (>5 oscillations, >100µΩ amplitude), sinusoidal resistance patterns
   - Excessive mechanical vibration during closing, oil leakage from dampers
   - Causes: Hydraulic oil degradation, seal failure, piston wear, gas spring leakage
   - Risks: Contact welding → Mechanical damage → Spring breakage → Stuck breaker

8. **SF6 Pressure Leakage**: 
   - Prolonged arcing duration (>25ms), poor arc quenching, elevated arc resistance
   - Pressure below spec (<5.5 bar), gas purity degradation, moisture ingress
   - Causes: Gasket deterioration, seal failure, manufacturing defects, thermal cycling
   - Risks: Reduced dielectric strength → Internal flashover → Arc quenching failure → Explosion

9. **Linkage/Rod Obstruction**: 
   - Stutter patterns during travel (>3 distinct flat plateaus >10ms), mechanical binding
   - Irregular travel curve, sudden speed changes, torque fluctuations
   - Causes: Foreign object ingress, ice formation, corrosion, misaligned rods
   - Risks: Incomplete stroke → Failed operation → Mechanical jam → Safety hazard

10. **Fixed Contact Damage**: 
    - Elevated DLRO (>80µΩ) with smooth stable Phase 3 (no grassy noise = stationary contact issue)
    - DC offset shift, baseline resistance elevation, contact deformation
    - Causes: Thermal damage, arc erosion of stationary contact, improper installation
    - Risks: Increased heating → Insulation damage → Tracking → Ground fault

11. **Close Coil Damage**: 
    - Close coil current <2A (healthy: 4-7A), failed closing operation, coil overheating
    - Open/short circuit in coil winding, control circuit failure
    - Causes: Electrical overstress, insulation breakdown, mechanical damage, moisture ingress
    - Risks: Failed closing → Breaker stuck open → System de-energization

12. **Trip Coil Damage**: 
    - Trip coil 1 or 2 current <2A, redundancy loss, failed opening operation
    - Both coils failed = catastrophic (breaker cannot trip during fault)
    - Causes: Coil burnout, circuit failure, auxiliary contact malfunction
    - Risks: Failed interruption → Sustained fault current → Equipment damage → Fire hazard

**KPI Thresholds (Healthy Ranges):**
- Closing Time: 70-110 ms
- Opening Time: 20-40 ms
- DLRO Value: 20-100 µΩ (ideal: 20-50 µΩ)
- Peak Resistance: 500-1000000000000 µΩ (during conduction excluding open baseline)
- Main Wipe: 10-20 mm
- Arc Wipe: 15-25 mm
- Contact Travel Distance: 150-200 mm
- Contact Speed: 2.0-6.0 m/s
- Coil Currents: 1-7 A (nominal: 4-7A)
- Ambient Temperature: 10-40°C

**CBHI Score Interpretation:**
- 90-100: Excellent - Routine monitoring only
- 75-89: Good - Minor preventive actions
- 60-74: Fair - Scheduled maintenance needed
- 40-59: Poor - Urgent attention required
- 0-39: Critical - Immediate intervention

**Common Failure Progressions:**
- Contact Wear → Increased Resistance → Overheating → Welding/Failure
- Gas Leak → Poor Arc Quenching → Contact Damage → Catastrophic Failure
- Mechanism Issues → Timing Drift → Coordination Loss → System Fault
- Coil Degradation → Incomplete Operation → Stuck Breaker → Protection Failure

**Maintenance Action Guidelines:**
- **Priority 1 (Critical)**: 0-1 month, safety/reliability risk, Red (#B71C1C bg:#FFCDD2)
- **Priority 2 (Important)**: 1-3 months, performance degradation, Amber (#F57F17 bg:#FFF9C4)
- **Priority 3 (Preventive)**: 3-6 months, optimization/prevention, Blue (#0D47A1 bg:#BBDEFB)
- **Priority 4 (Additional)**: If needed, Red (#B71C1C bg:#FFCDD2)
- **Priority 5 (Additional)**: If needed, Amber (#F57F17 bg:#FFF9C4)
- **CRITICAL: Each priority level must have EXACTLY ONE action** (create Priority 4, 5, etc. if needed)
- **Total actions: Minimum 3, Maximum 5** (distribute across priorities as needed)

**Future Fault Risk Levels & STRICT Color Mapping:**
- **High Risk**: Probability >60%, Timeline <12 months
- **Medium Risk**: Probability 30-60%, Timeline 12-24 months
- **Low Risk**: Probability <30%, Timeline >24 months
- **HARDCODED Colors (by position, NOT risk_level)**: 1st=#2E7D32, 2nd=#F57F17, 3rd=#0D47A1, 4th=#2E7D32, 5th=#F57F17
- **Total predictions: Minimum 3, Maximum 5** (ensure variety of risk levels)

===== INPUT DATA =====

**KPIs (Key Performance Indicators):**
{kpis_json}

**CBHI Score (Circuit Breaker Health Index):**
{cbhi_score} / 100

**Detected Faults:**
{faults_summary}

===== OUTPUT REQUIREMENTS =====

Return ONLY valid JSON (no markdown fences, no extra text). 

**CRITICAL JSON FORMATTING RULES:**
- All strings must use double quotes (")
- Escape any double quotes within strings using backslash (\")
- Do NOT use smart quotes, apostrophes, or special Unicode quotes
- Ensure all brackets and braces are properly matched
- Use standard ASCII characters only in field values
- Multi-line text should be a single string with spaces, not actual line breaks

**LANGUAGE STYLE - CRITICAL:**
- **justification**: Write as NATURAL EXPLANATION (not bullet points). Explain the CAUSE and EFFECT relationship clearly. 
  - Example GOOD: "SF6 pressure declining from 7.0 to 6.8 bar over recent tests. Gas quality directly affects arc quenching and contact performance."
  - Example BAD: "KPI: SF6 6.8 bar; Phase 2: arc; AI: leak 45%"
  - MAX 150 characters total
  
- **evidence** (for future faults): Write as EXPLANATORY SENTENCE describing the trend or pattern observed.
  - Example GOOD: "Contact resistance is significantly elevated, indicating material degradation. This will worsen over time, leading to overheating."
  - Example BAD: "DLRO +15%/mo; Phase 3: plateau; Wear: 68% conf"
  - MAX 120 characters total
  
- **Use proper technical narratives, NOT statistical shorthand**

Structure (EXACT format and ordering required):

{{
  "maintenanceActions": [
    {{
      "actions": [
        {{
          "id": "1",
          "justification": "Natural language explanation citing key evidence from KPIs/Faults",
          "timeline": "Within 1 month",
          "title": "Clear, actionable maintenance task",
          "whatToLookFor": [
            "Specific inspection point 1",
            "Specific measurement/check 2",
            "Expected finding/threshold 3",
            "Corrective action trigger 4"
          ]
        }}
      ],
      "bgColor": "#FFCDD2",
      "color": "#B71C1C",
      "priority": "Priority 1"
    }},
    {{
      "actions": [
        {{
          "id": "2",
          "justification": "Clear explanation with evidence",
          "timeline": "Within 3 months",
          "title": "Second most critical task",
          "whatToLookFor": ["item1", "item2", "item3", "item4"]
        }}
      ],
      "bgColor": "#FFF9C4",
      "color": "#F57F17",
      "priority": "Priority 2"
    }},
    {{
      "actions": [
        {{
          "id": "3",
          "justification": "Brief evidence-based reasoning",
          "timeline": "Within 6 months",
          "title": "Preventive maintenance task",
          "whatToLookFor": ["item1", "item2", "item3", "item4"]
        }}
      ],
      "bgColor": "#BBDEFB",
      "color": "#0D47A1",
      "priority": "Priority 3"
    }}
  ],
  "futureFaultsPdf": [
    {{
      "color": "#2E7D32",
      "evidence": "Natural language explanation of trends and patterns observed.",
      "fault": "Specific fault name from 12 defect classes",
      "id": "1",
      "probability": 68,
      "risk_level": "high",
      "timeline": "6 - 12 Months"
    }},
    {{
      "color": "#F57F17",
      "evidence": "Descriptive sentence about observed patterns and correlations.",
      "fault": "Another potential fault",
      "id": "2",
      "probability": 42,
      "risk_level": "medium",
      "timeline": "12 - 18 Months"
    }},
    {{
      "color": "#0D47A1",
      "evidence": "Long-term risk explanation based on industry experience.",
      "fault": "Third potential fault",
      "id": "3",
      "probability": 25,
      "risk_level": "low",
      "timeline": "> 24 Months"
    }}
  ]
}}

===== ANALYSIS INSTRUCTIONS =====

**Step 1: Deep Multi-Source Analysis**
- Cross-reference KPIs, CBHI, and Detected Faults to identify CORRELATIONS and PATTERNS
- Look for evidence chains: e.g., High DLRO → Elevated resistance → Main Wear verdict → Overheating risk
- Consider ALL 12 defect classes and identify which ones apply (even if probability is lower)

**Step 2: Maintenance Action Prioritization (3-5 actions total)**
- **CRITICAL RULE: Each priority level has EXACTLY ONE action** (no multiple actions per priority)
- **CRITICAL RULE: Provide 3-5 priority levels** (create Priority 4, 5 if needed)
- **CRITICAL RULE: The "id" field MUST match priority number** (Priority 1 → id="1", Priority 2 → id="2", etc.)
- **HARDCODED Color scheme (DO NOT CHANGE)**: 
  * Priority 1: color="#B71C1C", bgColor="#FFCDD2"
  * Priority 2: color="#F57F17", bgColor="#FFF9C4"
  * Priority 3: color="#0D47A1", bgColor="#BBDEFB"
  * Priority 4 (if any): color="#B71C1C", bgColor="#FFCDD2"
  * Priority 5 (if any): color="#F57F17", bgColor="#FFF9C4"

**Step 3: Future Fault Predictions (3-5 predictions)**
- **CRITICAL RULE: Provide 3-5 predictions with DIVERSE risk levels and timelines**
- **CRITICAL RULE: The "id" field is sequential** (1, 2, 3, 4, 5)
- **HARDCODED Color scheme (DO NOT CHANGE)**: 
  * 1st fault: color="#2E7D32"
  * 2nd fault: color="#F57F17"
  * 3rd fault: color="#0D47A1"
  * 4th fault (if any): color="#2E7D32"
  * 5th fault (if any): color="#F57F17"
- **Color assignment is FIXED by position, NOT by risk_level**
- **Progression logic to apply:**
  - Contact Wear → Overheating → Welding → Failed interruption (6-12 months high risk)
  - Gas Leak → Poor arc quenching → Contact damage → Catastrophic failure (12-18 months medium)
  - Mechanism wear → Timing drift → Coordination loss → Protection failure (18-24 months medium)
  - Coil degradation (one failed) → Second coil stress → Redundancy loss (>24 months low)
  - Arcing wear → Main contact damage → System fault (12-18 months medium)

**Remember**: Your output will be used by field maintenance engineers. Be PRECISE, ACTIONABLE, and INSIGHTFUL.
Generate thoughtful, data-driven, practical insights that can immediately guide maintenance decisions.
"""


def generate_recommendations(kpis, cbhi_score, rule_faults, ai_faults, llm):
    """
    Generates maintenance actions and future fault predictions based on DCRM analysis.
    
    Args:
        kpis (dict): Calculated KPIs.
        cbhi_score (float): The overall health score.
        rule_faults (list): Faults detected by the Rule Engine.
        ai_faults (list): Faults detected by the AI Agent.
        llm: The LLM instance (not used, using Gemini directly).
        
    Returns:
        dict: A dictionary containing 'maintenanceActions' and 'futureFaultsPdf'.
    """
    
    # 1. Prepare Context
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

    kpi_summary = json.dumps(kpis, indent=2, ensure_ascii=False)
    
    # 2. Format the prompt with actual data
    prompt = RECOMMENDATIONS_PROMPT.format(
        kpis_json=kpi_summary,
        cbhi_score=cbhi_score,
        faults_summary=faults_summary
    )
    
    # 3. Configure generation
    generation_config = {
        "temperature": 0.4,  
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
    }
    
    # 4. Initialize model
    model = genai.GenerativeModel(
        model_name=MODEL_NAME,
        generation_config=generation_config
    )
    
    # 5. Generate with retry logic
    max_retries = 3
    for attempt in range(max_retries):
        try:
            if attempt == 0:
                print("🔮 Generating recommendations and future fault predictions...")
            else:
                print(f"🔄 Retry attempt {attempt}/{max_retries-1}...")
            
            # Add JSON validation instruction on retry
            if attempt > 0:
                prompt_with_retry = prompt + "\n\nIMPORTANT: Ensure all text in justification, evidence, and whatToLookFor fields has properly escaped quotes. Use single quotes within text or escape double quotes with backslash."
            else:
                prompt_with_retry = prompt
            
            # Generate response
            response = model.generate_content(prompt_with_retry)
            
            # Check if response has valid content
            if not response or not response.text or response.text.strip() == "":
                print(f"⚠️ Empty response from API. Finish reason: {response.candidates[0].finish_reason if response and response.candidates else 'Unknown'}")
                if attempt == max_retries - 1:
                    return {
                        "error": "Empty response from API (quota/safety block)",
                        "finish_reason": response.candidates[0].finish_reason if response and response.candidates else None,
                        "maintenanceActions": [],
                        "futureFaultsPdf": []
                    }
                continue  # Try again
            
            # Extract JSON from response
            response_text = response.text.strip()
            
            # Clean up markdown code fences if present
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            
            response_text = response_text.strip()
            
            # Additional cleanup: fix common JSON issues
            # Replace smart quotes with regular quotes
            response_text = response_text.replace('"', '"').replace('"', '"')
            response_text = response_text.replace("'", "'").replace("'", "'")
            
            # Parse JSON
            result = json.loads(response_text)
            
            print("✅ Successfully generated recommendations and predictions")
            return result
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing error (attempt {attempt+1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                print(f"Raw response excerpt:\n{response_text[:1000]}...")
                # Try to salvage what we can
                return {
                    "error": f"Failed to parse AI response after {max_retries} attempts",
                    "raw_response": response_text[:2000],
                    "maintenanceActions": [],
                    "futureFaultsPdf": []
                }
            # Wait a bit before retry
            import time
            time.sleep(1)
            
        except Exception as e:
            print(f"❌ Error generating recommendations: {e}")
            return {
                "error": str(e),
                "maintenanceActions": [],
                "futureFaultsPdf": []
            }
    
    return {
        "error": "Max retries exceeded",
        "maintenanceActions": [],
        "futureFaultsPdf": []
    }
