# Previous Name: analysis/agents/advice_agent.py
import os
import sys
import json
import google.generativeai as genai

# Set UTF-8 encoding for console output
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# =========================
# CONFIGURATION
# =========================
API_KEY = os.environ.get("GOOGLE_API_KEY", "AIzaSyDSGda4-5VLmd-y09K6sBfHqoqk1QUL6Xo")
genai.configure(api_key=API_KEY)

MODEL_NAME = "gemini-2.0-flash"

# =========================
# AI ADVICE GENERATION PROMPT
# =========================
AI_ADVICE_PROMPT = """
Role: You are an elite Circuit Breaker Condition Monitoring & Diagnostics Expert with deep domain expertise in DCRM (Dynamic Contact Resistance Measurement), operating mechanism health, SF6 gas systems, contact metallurgy, and circuit breaker failure modes. You understand the nuances of contact bounce, contact wear progression, trip coil degradation, insulation breakdown, and mechanical timing issues.

Your Task: Based on comprehensive diagnostic data including RUL analysis, maintenance recommendations, and future fault predictions, generate 3-5 **Strategic Advisory Recommendations** that provide actionable insights grounded in circuit breaker engineering principles and field maintenance best practices.

===== INPUT DATA =====

**RUL Analysis (Remaining Useful Life):**
{rul_json}

**Maintenance Actions:**
{maintenance_json}

**Future Fault Predictions:**
{future_faults_json}

**KPIs (Key Performance Indicators):**
{kpis_json}

**CBHI Score:**
{cbhi_score} / 100

**Phase-Wise Analysis:**
{phase_analysis_json}

**AI Fault Detection:**
{ai_verdict_json}

===== AI ADVICE GUIDELINES =====

**Purpose**: Provide **domain-rich, engineering-focused** recommendations that demonstrate deep understanding of circuit breaker systems. Focus on:
- Contact resistance mitigation strategies and bounce reduction techniques
- Operating mechanism calibration and spring tension optimization
- SF6 gas management protocols and leak detection
- Contact refurbishment timing based on resistance trends
- Trip coil redundancy monitoring and protection system integrity
- Mechanical timing adjustments and damping system optimization

**Advice Categories to Consider (Based on Maintenance Actions & Future Faults):**
1. **Critical Interventions** (Priority: Critical, Color: #B71C1C)
   - Address Priority 1 maintenance actions and high-probability future faults
   - Focus on trip coil failures, severe contact wear (>250µΩ DLRO), SF6 leaks
   - Examples: "Optimize Contact Bounce Mitigation Strategy", "Enhance SF6 Gas Management Protocol"

2. **High-Priority Optimizations** (Priority: High, Color: #F57F17)
   - Address Priority 2 maintenance actions and medium-probability future faults
   - Focus on operating mechanism issues, contact refurbishment, insulation degradation
   - Examples: "Recalibrate Operating Mechanism Timing", "Implement Contact Wear Tracking Program"

3. **Preventive Enhancements** (Priority: High/Medium, Color: #F57F17 or #0D47A1)
   - Address Priority 3+ maintenance actions and low-probability future faults
   - Focus on long-term reliability improvements and preventive measures
   - Examples: "Optimize Lubrication & Damping Systems", "Enhance Phase-Wise Diagnostic Monitoring"

**CRITICAL JSON FORMATTING RULES:**
- All strings must use double quotes (")
- Escape any double quotes within strings using backslash (\")
- Do NOT use smart quotes, apostrophes, or special Unicode quotes
- Ensure all brackets and braces are properly matched
- Use standard ASCII characters only in field values

**Output Structure (EXACT format required):**

{{
  "aiAdvice": [
    {{
      "color": "#B71C1C",
      "confidence": 94,
      "description": "Clear, domain-specific description focusing on circuit breaker systems. Explain the engineering rationale and field maintenance approach. Reference specific components like contact springs, damping dashpots, SF6 purity, contact metallurgy, or mechanism linkages. Keep it concise but technically rich (MAX 250 characters).",
      "effectAnalysis": {{
        "longTerm": [
          "Long-term benefit 1 (e.g., 'Extended circuit breaker service life by 25-30%')",
          "Long-term benefit 2 (e.g., 'Reduced maintenance costs by $50,000-75,000 annually')",
          "Long-term benefit 3 (e.g., 'Improved grid reliability and reduced downtime')",
          "Long-term benefit 4 (e.g., 'Predictive maintenance scheduling optimization')"
        ],
        "performanceGains": [
          "Performance gain 1 (e.g., 'Improved contact resistance stability')",
          "Performance gain 2 (e.g., 'Enhanced operational reliability')"
        ],
        "riskMitigation": [
          "Risk mitigation 1 (e.g., 'Prevent catastrophic failures through early detection')",
          "Risk mitigation 2 (e.g., 'Reduce unplanned outages by 75%')"
        ],
        "shortTerm": [
          "Short-term benefit 1 (e.g., 'Immediate improvement in contact resistance stability')",
          "Short-term benefit 2 (e.g., 'Reduced contact bounce amplitude by 60-70%')",
          "Short-term benefit 3 (e.g., 'Enhanced SF6 pressure monitoring accuracy')",
          "Short-term benefit 4 (e.g., 'Real-time fault detection capabilities')"
        ]
      }},
      "expectedImpact": "Primary quantifiable or qualitative benefit (e.g., 'Reduce contact wear rate by 40%', 'Prevent contact welding risk', 'Extend asset life by 2-3 years'). Be specific and measurable (MAX 100 characters).",
      "id": "1",
      "priority": "Critical",
      "title": "Domain-specific, action-oriented title (e.g., 'Optimize Contact Bounce Mitigation Strategy', 'Enhance SF6 Gas Management Protocol'). MAX 80 characters."
    }},
    {{
      "color": "#F57F17",
      "confidence": 89,
      "description": "Second most critical advice with engineering depth.",
      "effectAnalysis": {{
        "longTerm": ["benefit 1", "benefit 2", "benefit 3", "benefit 4"],
        "performanceGains": ["gain 1", "gain 2"],
        "riskMitigation": ["mitigation 1", "mitigation 2"],
        "shortTerm": ["benefit 1", "benefit 2", "benefit 3", "benefit 4"]
      }},
      "expectedImpact": "Expected benefit or risk reduction.",
      "id": "2",
      "priority": "High",
      "title": "Second priority engineering recommendation"
    }},
    {{
      "color": "#F57F17",
      "confidence": 92,
      "description": "Third strategic recommendation with domain expertise.",
      "effectAnalysis": {{
        "longTerm": ["benefit 1", "benefit 2", "benefit 3", "benefit 4"],
        "performanceGains": ["gain 1", "gain 2"],
        "riskMitigation": ["mitigation 1", "mitigation 2"],
        "shortTerm": ["benefit 1", "benefit 2", "benefit 3", "benefit 4"]
      }},
      "expectedImpact": "Quantifiable impact or performance improvement.",
      "id": "3",
      "priority": "High",
      "title": "Third priority optimization strategy"
    }}
  ]
}}

===== ANALYSIS INSTRUCTIONS =====

**Step 1: Synthesize All Input Data**
- Analyze RUL estimate and uncertainty to assess urgency
- Review maintenance priorities to identify critical failure modes
- Examine future fault predictions to understand progression risks
- Cross-reference KPIs, phase analysis, and AI verdict for root cause insights

**Step 2: Generate Strategic AI Advice (3-5 items)**
- **CRITICAL: Generate exactly 3-5 advice items** (minimum 3, maximum 5)
- **CRITICAL: id field is sequential** ("1", "2", "3", "4", "5")
- **CRITICAL JSON FIELD ORDER**: color → confidence → description → effectAnalysis → expectedImpact → id → priority → title
- **effectAnalysis Structure**: MUST contain all 4 arrays: "longTerm" (4 items), "performanceGains" (2 items), "riskMitigation" (2 items), "shortTerm" (4 items)
- **HARDCODED Color scheme**:
  * 1st advice: "#B71C1C" (Critical priority)
  * 2nd advice: "#F57F17" (High priority)
  * 3rd advice: "#F57F17" (High priority)
  * 4th advice (if any): "#0D47A1" (Medium priority)
  * 5th advice (if any): "#2E7D32" (Low priority)

**Confidence Scoring:**
- 90-100: High confidence, strong data support, clear evidence chain
- 80-89: Good confidence, supported by multiple indicators
- 70-79: Moderate confidence, some uncertainty in data
- 60-69: Lower confidence, limited data or conflicting signals

**Priority Assignment:**
- **Critical**: RUL <500 cycles, imminent safety risk, coil failure, SF6 leak
- **High**: RUL <1500 cycles, significant degradation, contact wear >200µΩ
- **High**: RUL <3000 cycles, preventive optimization, performance improvement
- **Medium**: RUL >3000 cycles, long-term enhancements, technology upgrades
- **Low**: RUL >5000 cycles, continuous improvement, best practices

**Strategic Advice Themes to Consider (Domain-Rich, Engineering-Focused):**

**IMPORTANT: Base your recommendations on the SPECIFIC maintenance actions and future faults provided in the input data.**

1. **Contact Bounce Mitigation** (if contact bounce detected in 55-65ms region):
   - "Optimize Contact Bounce Mitigation Strategy"
   - Description: "Address the detected contact bounce (55-65ms region) through operating mechanism calibration and spring tension adjustment. Calibrate damping dashpot to reduce impact velocity and minimize bounce amplitude, preventing accelerated contact erosion."
   - Short-term effects: "Immediate reduction in contact bounce amplitude by 60-70%", "Stabilized contact resistance during closing", "Enhanced closing operation smoothness", "Real-time bounce monitoring activation"
   - Long-term effects: "Extended contact life by 25-30%", "Reduced risk of contact welding by 40%", "Lower maintenance frequency and costs", "Improved breaker reliability over 5+ years"
   - Performance gains: "Improved contact resistance stability", "Enhanced operational reliability"
   - Risk mitigation: "Prevent accelerated contact erosion", "Reduce risk of contact failure"

2. **SF6 Gas Management** (if SF6 issues, insulation anomalies, or Phase 5 low confidence):
   - "Enhanced SF6 Gas Management Protocol"
   - Description: "Implement predictive SF6 pressure monitoring with automated alerts for gas quality degradation and leakage detection. Monitor moisture content, purity, and decomposition products to prevent insulation breakdown."
   - Short-term effects: "Real-time SF6 leak detection", "Enhanced gas purity monitoring accuracy", "Immediate identification of moisture ingress", "Automated alert system activation"
   - Long-term effects: "Prevented insulation failures saving $75,000-100,000", "Extended gas replacement intervals", "Reduced environmental SF6 emissions", "Improved arc quenching performance consistency"
   - Performance gains: "Maintained optimal arc quenching capability", "Enhanced dielectric strength stability"
   - Risk mitigation: "Prevent insulation breakdown and flashover events", "Reduce environmental compliance risks"

3. **Trip Coil Redundancy** (if Trip Coil 2 failed or coil current = 0.0A):
   - "Trip Coil Redundancy & Protection System Enhancement"
   - Description: "With Trip Coil 2 failed (0.0A current), establish continuous monitoring of Trip Coil 1 operational status. Implement backup trip mechanisms and redundancy verification protocols to ensure critical fault interruption capability."
   - Short-term effects: "Immediate trip coil status verification", "Backup protection activation", "Enhanced fault interruption assurance", "Real-time coil health monitoring"
   - Long-term effects: "Prevented protection system failures by 95%", "Enhanced grid reliability and safety", "Reduced risk of catastrophic grid faults", "Extended protection system lifespan"
   - Performance gains: "Guaranteed fault clearing capability", "Improved protection system reliability"
   - Risk mitigation: "Eliminate single-point-of-failure in trip system", "Prevent grid instability due to breaker trip failure"

4. **Contact Refurbishment Planning** (if DLRO >200µΩ or Main Contact Wear detected):
   - "Strategic Contact Refurbishment & Wear Tracking Program"
   - Description: "With DLRO at 300µΩ indicating severe contact wear, plan proactive contact refurbishment within 3 months. Implement contact wear tracking to optimize refurbishment timing and prevent contact welding or overheating failures."
   - Short-term effects: "Immediate contact resistance stability improvement", "Thermal hotspot elimination", "Reduced I²R losses and heating", "Enhanced current carrying capacity"
   - Long-term effects: "Extended circuit breaker service life by 2-3 years", "Reduced emergency outage costs by $50,000+", "Prevented contact welding incidents", "Optimized maintenance scheduling"
   - Performance gains: "Restored nominal contact resistance levels", "Improved current distribution across contacts"
   - Risk mitigation: "Prevent contact welding and breaker failure-to-open", "Eliminate overheating-induced insulation damage"

5. **Operating Mechanism Calibration** (if excessive travel, wipe, speed, or timing deviations):
   - "Operating Mechanism Calibration & Timing Optimization"
   - Description: "Recalibrate operating mechanism to address excessive contact travel, wipe, or speed anomalies. Optimize spring tension, lubrication, and linkage alignment to reduce mechanical stress and improve timing accuracy."
   - Short-term effects: "Normalized contact motion parameters", "Reduced mechanical vibration by 40%", "Improved timing consistency", "Enhanced mechanism smoothness"
   - Long-term effects: "Extended mechanism component life by 30%", "Improved operational reliability", "Reduced wear on linkages and bearings", "Lower maintenance intervention frequency"
   - Performance gains: "Optimized contact closing/opening velocities", "Enhanced mechanical timing precision"
   - Risk mitigation: "Prevent mechanism seizure or binding", "Reduce risk of contact damage due to excessive impact forces"

6. **Phase Anomaly Investigation** (if Phase 5 low confidence or electrical anomalies):
   - "Phase-Wise Diagnostic Enhancement & Anomaly Resolution"
   - Description: "Investigate Final Open State anomaly showing 15% confidence with abnormal resistance profile. Conduct insulation resistance tests (Megger, PI, DAR), SF6 analysis, and internal inspection to identify root cause of electrical anomaly."
   - Short-term effects: "Root cause identification of phase anomaly", "Enhanced diagnostic confidence", "Immediate safety hazard assessment", "Isolation capability verification"
   - Long-term effects: "Prevented insulation breakdown failures", "Optimized phase-wise health monitoring", "Improved diagnostic accuracy for future assessments", "Reduced uncertainty in condition assessments"
   - Performance gains: "Confirmed dielectric strength integrity", "Enhanced phase-wise analysis reliability"
   - Risk mitigation: "Prevent internal flashover or tracking failures", "Eliminate incomplete isolation hazards"

**Step 3: Tailor Advice to Current Situation (Analyze Maintenance Actions & Future Faults)**
- **CRITICAL**: Review the Priority 1, 2, 3+ maintenance actions in detail
- **CRITICAL**: Review the high, medium, low risk future faults in detail
- **CRITICAL**: Combine insights from both to create cohesive recommendations

**Mapping Guidelines:**
- If Priority 1 includes Trip Coil repair → Focus on trip coil redundancy and protection system
- If Priority 1 includes contact refurbishment → Focus on contact wear tracking and DLRO monitoring
- If Priority 2 includes mechanism overhaul → Focus on operating mechanism calibration and timing
- If Priority 3 includes insulation/SF6 investigation → Focus on SF6 gas management and phase diagnostics
- If future faults include "Main Contact Wear" (high probability) → Emphasize contact refurbishment urgency
- If future faults include "Operating Mechanism Malfunction" → Emphasize mechanism calibration
- If future faults include "Trip Coil Damage" → Emphasize coil monitoring and redundancy
- If RUL is low (<500 cycles) → Add Critical priority advice on immediate monitoring
- If DLRO >250µΩ → Add High priority advice on contact resistance mitigation
- If Phase 5 confidence <20% → Add advice on phase-wise diagnostic enhancement

**FINAL CHECKLIST BEFORE GENERATING OUTPUT:**
1. ✓ Total AI advice items: 3-5 (minimum 3, maximum 5)
2. ✓ Each advice item has all required fields: color, confidence, description, effectAnalysis, expectedImpact, id, priority, title
3. ✓ **CRITICAL JSON FIELD ORDER**: color → confidence → description → effectAnalysis → expectedImpact → id → priority → title
4. ✓ **effectAnalysis has all 4 required arrays**: longTerm (4 items), performanceGains (2 items), riskMitigation (2 items), shortTerm (4 items)
5. ✓ **HARDCODED Colors**: 1st=#B71C1C, 2nd=#F57F17, 3rd=#F57F17, 4th=#0D47A1, 5th=#2E7D32
6. ✓ Confidence values are realistic (60-100 range)
7. ✓ Descriptions are domain-rich, engineering-focused, and specific (under 250 chars)
8. ✓ Expected impacts are quantifiable benefits from circuit breaker domain (under 100 chars)
9. ✓ Titles are domain-specific and clear (under 80 chars)
10. ✓ All JSON properly formatted with double quotes, no smart quotes
11. ✓ Advice is based on SPECIFIC maintenance actions and future faults from input data
12. ✓ Recommendations reference actual KPI values (DLRO, coil current, timing, etc.)
13. ✓ Each advice item provides unique value focused on circuit breaker engineering
14. ✓ Short-term effects are immediate operational improvements (weeks to months)
15. ✓ Long-term effects are strategic benefits (years, cost savings, reliability)
16. ✓ Performance gains focus on operational improvements and capability enhancements
17. ✓ Risk mitigation items identify specific hazards being prevented
18. ✓ In any response dont mention like phase 1,2,3,4,5 etc..instead replace with their formal names like:
     phase 1: Pre Contact Travel
     phase 2: Arc Initiation
     phase 3: Main Contact Conduction
     phase 4: Main Contact Separation & Arc elongation
     phase 5: Final Open State
**Remember**: Your advice should demonstrate deep expertise in DCRM circuit breaker systems. Reference specific components, failure modes, and field maintenance practices. Avoid generic "AI" or "IoT" buzzwords - instead use domain-specific terms like "contact bounce mitigation", "SF6 purity monitoring", "trip coil redundancy", "damping dashpot calibration", "DLRO trending", etc.

Generate value-rich, engineering-grounded insights that maintenance teams can immediately understand and implement.
"""

# =========================
# MAIN FUNCTION
# =========================
def generate_ai_advice(rul_data, recommendations_data, kpis_data, cbhi_score, phase_analysis, ai_verdict):
    """
    Generate AI-driven strategic advice using Gemini AI.
    
    Args:
        rul_data (dict): RUL analysis with rulEstimate and uncertainty
        recommendations_data (dict): Combined dict with "maintenanceActions" and "futureFaultsPdf"
        kpis_data (dict): KPI data with structure {"kpis": [...]}
        cbhi_score (int): Circuit Breaker Health Index (0-100)
        phase_analysis (dict): Phase-wise analysis JSON
        ai_verdict (dict): AI fault detection verdict
    
    Returns:
        dict: JSON with aiAdvice array
    """
    
    # Extract maintenance and future faults from combined data
    maintenance_data = recommendations_data.get("maintenanceActions", [])
    future_faults_data = recommendations_data.get("futureFaultsPdf", [])
    
    # Format the prompt with actual data
    prompt = AI_ADVICE_PROMPT.format(
        rul_json=json.dumps(rul_data, indent=2, ensure_ascii=False),
        maintenance_json=json.dumps(maintenance_data, indent=2, ensure_ascii=False),
        future_faults_json=json.dumps(future_faults_data, indent=2, ensure_ascii=False),
        kpis_json=json.dumps(kpis_data, indent=2, ensure_ascii=False),
        cbhi_score=cbhi_score,
        phase_analysis_json=json.dumps(phase_analysis, indent=2, ensure_ascii=False),
        ai_verdict_json=json.dumps(ai_verdict, indent=2, ensure_ascii=False)
    )
    
    # Configure generation
    generation_config = {
        "temperature": 0.3,  # Lower temperature for more focused, consistent output
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 4096,
    }
    
    # Initialize model
    model = genai.GenerativeModel(
        model_name=MODEL_NAME,
        generation_config=generation_config
    )
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            if attempt == 0:
                print("🤖 Generating AI strategic advice...")
            else:
                print(f"🔄 Retry attempt {attempt}/{max_retries-1}...")
            
            # Add JSON validation instruction on retry
            if attempt > 0:
                prompt_with_retry = prompt + "\n\nIMPORTANT: Ensure all text in description, expectedImpact, and title fields has properly escaped quotes. Use single quotes within text or escape double quotes with backslash."
            else:
                prompt_with_retry = prompt
            
            # Generate response
            response = model.generate_content(prompt_with_retry)
            
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
            
            print("✅ Successfully generated AI strategic advice")
            return result
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing error (attempt {attempt+1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                print(f"Raw response excerpt:\n{response_text[:1000]}...")
                # Try to salvage what we can
                return {
                    "error": f"Failed to parse AI response after {max_retries} attempts",
                    "raw_response": response_text[:2000],
                    "aiAdvice": []
                }
            # Wait a bit before retry
            import time
            time.sleep(1)
            
        except Exception as e:
            print(f"❌ Error generating AI advice: {e}")
            return {
                "error": str(e),
                "aiAdvice": []
            }
    
    return {
        "error": "Max retries exceeded",
        "aiAdvice": []
    }
