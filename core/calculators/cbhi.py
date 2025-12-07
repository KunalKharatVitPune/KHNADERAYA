# Previous Name: analysis/cbhi_calculator.py
import json

def calculate_score(val, min_good, max_good, buffer):
    # Calculate the middle of the ideal range
    middle = (min_good + max_good) / 2.0
    half_range = (max_good - min_good) / 2.0
    
    if min_good <= val <= max_good:
        # Distance from the middle point
        distance_from_middle = abs(val - middle)
        # If exactly at middle, score is 100
        if distance_from_middle == 0:
            return 100.0
        # Scale from 90 to 100 based on how far from middle
        # Closer to middle = higher score (90-100%)
        score = 100.0 - (10.0 * (distance_from_middle / half_range))
        return max(90.0, score)  # Ensure minimum 90% within range
    
    # 2. Low Side: Value is lower than min_good
    if val < min_good:
        # Calculate how far off it is
        distance = min_good - val
        # If distance exceeds buffer, score is 0. Otherwise, scale it.
        if distance >= buffer:
            return 0.0
        return 90.0 * (1.0 - (distance / buffer))

    # 3. High Side: Value is higher than max_good
    if val > max_good:
        distance = val - max_good
        if distance >= buffer:
            return 0.0
        return 90.0 * (1.0 - (distance / buffer))
    
    return 0.0

WEIGHTS = {
    "peak_resistance": 0.15,
    "dlro": 0.10,
    "travel": 0.10,
    "speed": 0.10,
    "open_time": 0.15,
    "close_time": 0.15,
    "main_wipe": 0.05,
    "arc_wipe": 0.05,
    "coil_current": 0.10,  
    "temp": 0.05
}

# ==========================================
# NEW LOGIC: AI & PHASE ADJUSTMENTS
# ==========================================

def calculate_ai_penalty(ai_verdict_list):
    """
    Calculates penalty based on AI defects.
    High: -15, Medium: -12, Low: -10 (Scaled by Confidence)
    """
    total_penalty = 0.0
    
    # Base penalty values
    severity_map = {
        "High": 15.0,
        "Medium": 12.0,
        "Low": 10.0
    }

    # Ensure input is a list (handle single object or list of objects)
    if isinstance(ai_verdict_list, dict):
        ai_verdict_list = [ai_verdict_list]
    
    for defect in ai_verdict_list:
        label = defect.get("faultLabel", defect.get("defect_name", "Healthy"))
        
        # If explicitly Healthy, no penalty
        if label.lower() == "healthy":
            continue

        severity = defect.get("severity", "Low")
        confidence = float(defect.get("confidence", 0)) / 100.0
        
        # Dynamic Penalty: Base * Confidence
        # Example: High (15) * 0.90 = 13.5 deduction
        base_deduction = severity_map.get(severity, 10.0)
        actual_deduction = base_deduction * confidence
        
        total_penalty += actual_deduction

    return total_penalty

def calculate_phase_adjustment(phase_data):
    """
    Calculates adjustment based on 5 phases.
    Healthy: Add +2.0 to +3.0 (based on confidence)
    Not Healthy: Subtract -1.5 to -2.0 (based on confidence)
    """
    adjustment = 0.0
    
    for phase_name, data in phase_data.items():
        status = data.get("status", "Not Healthy").lower()
        confidence = float(data.get("confidence", 0)) / 100.0
        
        if status == "healthy":
            # REWARD: Range 2.0 to 3.0
            # If conf=100 -> +3.0, If conf=0 -> +2.0
            reward = 2.0 + (1.0 * confidence)
            adjustment += reward
        else:
            # PENALTY: Range 1.5 to 2.0
            # If conf=100 -> -2.0, If conf=0 -> -1.5
            penalty = 1.5 + (0.5 * confidence)
            adjustment -= penalty
            
    return adjustment

# ==========================================
# MAIN COMPUTE FUNCTION (UPDATED)
# ==========================================

def compute_cbhi(kpis_list, ai_data=None, phase_data=None):
    # 1. Parse KPIs into Dictionary
    kpi_dict = {item['name']: item['value'] for item in kpis_list}

    # 2. Compute Individual KPI Scores (Existing Logic)
    def get_val(name):
        v = kpi_dict.get(name)
        return v if v is not None else 0.0

    s_dlro = calculate_score(get_val("DLRO Value"), 20, 100, 50)
    s_peak = calculate_score(get_val("Peak Resistance"), 80, 150, 200)
    s_travel = calculate_score(get_val("Contact Travel Distance"), 150, 200, 30)
    s_speed = calculate_score(get_val("Contact Speed"), 2.0, 6.0, 1.5)
    s_open = calculate_score(get_val("Opening Time"), 20, 40, 20)
    s_close = calculate_score(get_val("Closing Time"), 70, 110, 20)
    s_mw = calculate_score(get_val("Main Wipe"), 10, 20, 5)
    s_aw = calculate_score(get_val("Arc Wipe"), 15, 25, 5)
    
    c_close = get_val("Peak Close Coil Current")
    c_trip = max(get_val("Peak Trip Coil 1 Current"), get_val("Peak Trip Coil 2 Current"))
    s_coil_c = calculate_score(c_close, 1.0, 7.0, 5.0)
    s_coil_t = calculate_score(c_trip, 1.0, 7.0, 5.0)
    s_coil = (s_coil_c + s_coil_t) / 2.0
    
    s_temp = calculate_score(get_val("Ambient Temperature"), 10, 40, 30)

    # 3. Store Scores
    scores = {
        "dlro": s_dlro,
        "peak_resistance": s_peak,
        "travel": s_travel,
        "speed": s_speed,
        "open_time": s_open,
        "close_time": s_close,
        "main_wipe": s_mw,
        "arc_wipe": s_aw,
        "coil_current": s_coil,
        "temp": s_temp
    }

    # 4. Calculate Base Weighted Score
    base_cbhi = sum(scores[key] * WEIGHTS[key] for key in scores)
    
    # 5. Calculate AI Verdict Penalty
    ai_penalty = 0.0
    if ai_data:
        # Extract verdicts from ai_data robustly
        defects = []
        if isinstance(ai_data, dict):
            if "aiVerdict" in ai_data:
                verdicts = ai_data["aiVerdict"]
                if isinstance(verdicts, dict):
                    defects = [verdicts]
                elif isinstance(verdicts, list):
                    defects = verdicts
            else:
                defects = [ai_data]
        elif isinstance(ai_data, list):
            defects = ai_data
        # Only penalize non-healthy verdicts
        non_healthy_defects = [d for d in defects if str(d.get("faultLabel", d.get("defect_name", "Healthy")).lower()) != "healthy"]
        if non_healthy_defects:
            ai_penalty = calculate_ai_penalty(non_healthy_defects)

    # 6. Calculate Phase Adjustments
    phase_adj = 0.0
    if phase_data:
        phase_adj = calculate_phase_adjustment(phase_data)

    # 7. Final Aggregation
    final_score = base_cbhi - ai_penalty + phase_adj
    
    # 8. Clamp to 0-100
    final_score = max(3.33, min(98.67, final_score))

    # Return Detailed Object (optional) or just the number
    return  int(round(final_score, 2))
