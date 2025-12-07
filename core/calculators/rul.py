# Previous Name: analysis/rul_calculator.py
import math
from typing import Dict, Any

def calculate_rul_and_uncertainty(kpis: Dict, cbhi_score: float, ai_verdict: Dict, phase_data: Dict) -> Dict:
    """
    Optimized RUL Calculator with tight bounds and defect-specific degradation.
    
    Returns:
        Dictionary with only 'rulEstimate' and 'uncertainty'
    """
    
    # =========================================================================
    # 1. INDUSTRIAL CONSTANTS (IEEE C37.10 / IEC 62271-100)
    # =========================================================================
    MAX_CYCLES = 10000      # Class M2 rated life
    HEALTHY_BASELINE = 8500 # Healthy breaker typical RUL
    MIN_CYCLES = 100        # Absolute floor
    
    # Defect Class Degradation Multipliers (1.0 = no impact, lower = worse)
    # These directly reduce RUL based on detected fault type
    DEFECT_MULTIPLIERS = {
        "healthy": 1.0,
        "main contact wear": 0.35,
        "arcing contact wear": 0.40,
        "main contact misalignment": 0.45,
        "arcing contact misalignment": 0.50,
        "operating mechanism malfunction": 0.55,
        "damping system fault": 0.60,
        "sf6 pressure leakage": 0.30,
        "linkage obstruction": 0.50,
        "fixed contact damage": 0.40,
        "close coil damage": 0.25,
        "trip coil damage": 0.20,
        "unknown": 0.70
    }
    
    # Severity-based additional reduction
    SEVERITY_REDUCTION = {
        "Critical": 0.50,
        "High": 0.70,
        "Medium": 0.85,
        "Low": 0.95,
        "None": 1.0
    }
    
    # KPI Critical Thresholds (beyond these = severe degradation)
    KPI_CRITICAL = {
        "DLRO Value": {"healthy_max": 70, "warning": 150, "critical": 250, "severe": 350},
        "Peak Resistance": {"healthy_max": 400, "warning": 600, "critical": 800, "severe": 1000},
        "Closing Time": {"healthy_max": 100, "warning": 120, "critical": 140, "severe": 160},
        "Opening Time": {"healthy_max": 45, "warning": 55, "critical": 70, "severe": 90},
        "Contact Speed": {"healthy_min": 4.0, "healthy_max": 6.5, "warning_dev": 1.0, "critical_dev": 2.0},
        "Peak Trip Coil 1 Current": {"healthy_min": 3.0, "critical_min": 1.5, "zero": 0.5},
        "Peak Trip Coil 2 Current": {"healthy_min": 3.0, "critical_min": 1.5, "zero": 0.5},
        "Peak Close Coil Current": {"healthy_min": 3.0, "critical_min": 1.5, "zero": 0.5},
    }
    
    # =========================================================================
    # 2. HELPER FUNCTIONS
    # =========================================================================
    
    def clamp(val: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
        return max(min_val, min(max_val, val))
    
    def safe_get(d: Any, *keys, default=None):
        for key in keys:
            if isinstance(d, dict):
                d = d.get(key, default)
            elif isinstance(d, list) and isinstance(key, int) and 0 <= key < len(d):
                d = d[key]
            else:
                return default
        return d if d is not None else default
    
    def get_defect_multiplier(fault_label: str) -> float:
        """Get degradation multiplier based on fault label."""
        label_lower = fault_label.lower().strip()
        for defect, multiplier in DEFECT_MULTIPLIERS.items():
            if defect in label_lower or label_lower in defect:
                return multiplier
        # Partial matches
        if "wear" in label_lower:
            return 0.38
        if "misalignment" in label_lower:
            return 0.48
        if "coil" in label_lower:
            return 0.22
        if "mechanism" in label_lower or "damping" in label_lower:
            return 0.55
        if "sf6" in label_lower or "pressure" in label_lower or "leak" in label_lower:
            return 0.30
        return DEFECT_MULTIPLIERS.get("unknown", 0.70)
    
    # =========================================================================
    # 3. EXTRACT DATA
    # =========================================================================
    
    # Extract KPIs
    kpis_list = kpis.get("kpis", kpis) if isinstance(kpis, dict) else kpis
    if isinstance(kpis_list, dict):
        kpis_list = [{"name": k, "value": v} for k, v in kpis_list.items()]
    kpi_map = {k.get('name', ''): float(k.get('value', 0)) for k in kpis_list if isinstance(k, dict)}
    
    # Extract AI Verdict
    ai_verdict_data = safe_get(ai_verdict, "aiVerdict", default={})
    fault_label = safe_get(ai_verdict_data, "faultLabel", default="Unknown")
    ai_confidence = float(safe_get(ai_verdict_data, "confidence", default=50)) / 100.0
    ai_severity = safe_get(ai_verdict_data, "severity", default="Low")
    severity_reason = safe_get(ai_verdict_data, "severityReason", default="")
    
    # Extract CBHI components
    cbhi_components = safe_get(ai_verdict, "cbhi", "overall_health_assessment", default={})
    
    # Extract Phase Analysis
    phases_list = safe_get(phase_data, "phaseWiseAnalysis", default=[])
    if not isinstance(phases_list, list):
        phases_list = []
    
    # Normalize CBHI
    cbhi_normalized = clamp(float(cbhi_score) / 100.0)
    
    # =========================================================================
    # 4. DETERMINE IF HEALTHY OR FAULTY
    # =========================================================================
    
    is_healthy = fault_label.lower().strip() == "healthy"
    
    # =========================================================================
    # 5. CALCULATE ELECTRICAL WEAR FACTOR (Contact & Resistance Issues)
    # =========================================================================
    
    electrical_penalties = []
    
    # DLRO Analysis (most critical for electrical wear)
    dlro = kpi_map.get("DLRO Value", 50)
    if dlro <= 70:
        electrical_penalties.append(0.0)  # Healthy
    elif dlro <= 150:
        electrical_penalties.append(0.15 + 0.15 * (dlro - 70) / 80)
    elif dlro <= 250:
        electrical_penalties.append(0.30 + 0.25 * (dlro - 150) / 100)
    elif dlro <= 350:
        electrical_penalties.append(0.55 + 0.25 * (dlro - 250) / 100)
    else:
        electrical_penalties.append(0.80 + 0.15 * min((dlro - 350) / 150, 1.0))
    
    # Peak Resistance Analysis
    peak_r = kpi_map.get("Peak Resistance", 300)
    if peak_r <= 400:
        electrical_penalties.append(0.0)
    elif peak_r <= 600:
        electrical_penalties.append(0.10 + 0.15 * (peak_r - 400) / 200)
    elif peak_r <= 800:
        electrical_penalties.append(0.25 + 0.20 * (peak_r - 600) / 200)
    else:
        electrical_penalties.append(0.45 + 0.25 * min((peak_r - 800) / 400, 1.0))
    
    # Coil Current Analysis (Trip Coil failures are CRITICAL)
    trip1 = kpi_map.get("Peak Trip Coil 1 Current", 5.0)
    trip2 = kpi_map.get("Peak Trip Coil 2 Current", 5.0)
    close_coil = kpi_map.get("Peak Close Coil Current", 5.0)
    
    # Trip Coil 1 penalty
    if trip1 < 0.5:
        electrical_penalties.append(0.70)  # Critical - coil failed
    elif trip1 < 1.5:
        electrical_penalties.append(0.50)
    elif trip1 < 3.0:
        electrical_penalties.append(0.25)
    else:
        electrical_penalties.append(0.0)
    
    # Trip Coil 2 penalty
    if trip2 < 0.5:
        electrical_penalties.append(0.60)  # Critical - redundancy lost
    elif trip2 < 1.5:
        electrical_penalties.append(0.40)
    elif trip2 < 3.0:
        electrical_penalties.append(0.20)
    else:
        electrical_penalties.append(0.0)
    
    # Close Coil penalty
    if close_coil < 0.5:
        electrical_penalties.append(0.50)
    elif close_coil < 1.5:
        electrical_penalties.append(0.30)
    elif close_coil < 3.0:
        electrical_penalties.append(0.15)
    else:
        electrical_penalties.append(0.0)
    
    # Weighted electrical wear (DLRO has highest weight)
    electrical_weights = [0.35, 0.20, 0.20, 0.15, 0.10]  # DLRO, Peak_R, Trip1, Trip2, Close
    electrical_wear = sum(p * w for p, w in zip(electrical_penalties, electrical_weights))
    
    # =========================================================================
    # 6. CALCULATE MECHANICAL WEAR FACTOR (Timing & Mechanism Issues)
    # =========================================================================
    
    mechanical_penalties = []
    
    # Closing Time Analysis
    close_time = kpi_map.get("Closing Time", 85)
    if close_time <= 100:
        mechanical_penalties.append(0.0)
    elif close_time <= 120:
        mechanical_penalties.append(0.15 + 0.15 * (close_time - 100) / 20)
    elif close_time <= 140:
        mechanical_penalties.append(0.30 + 0.25 * (close_time - 120) / 20)
    else:
        mechanical_penalties.append(0.55 + 0.25 * min((close_time - 140) / 40, 1.0))
    
    # Opening Time Analysis
    open_time = kpi_map.get("Opening Time", 35)
    if open_time <= 45:
        mechanical_penalties.append(0.0)
    elif open_time <= 55:
        mechanical_penalties.append(0.15 + 0.15 * (open_time - 45) / 10)
    elif open_time <= 70:
        mechanical_penalties.append(0.30 + 0.25 * (open_time - 55) / 15)
    else:
        mechanical_penalties.append(0.55 + 0.25 * min((open_time - 70) / 30, 1.0))
    
    # Contact Speed Analysis
    speed = kpi_map.get("Contact Speed", 5.0)
    if 4.0 <= speed <= 6.5:
        mechanical_penalties.append(0.0)
    elif 3.0 <= speed <= 7.5:
        deviation = max(abs(speed - 4.0), abs(speed - 6.5)) if speed < 4.0 or speed > 6.5 else 0
        mechanical_penalties.append(0.15 + 0.15 * deviation)
    else:
        mechanical_penalties.append(0.50)
    
    # Contact Travel Distance Analysis
    travel = kpi_map.get("Contact Travel Distance", 550)
    if 500 <= travel <= 600:
        mechanical_penalties.append(0.0)
    elif 450 <= travel <= 650:
        mechanical_penalties.append(0.15)
    else:
        mechanical_penalties.append(0.35)
    
    # Main/Arc Wipe Analysis
    main_wipe = kpi_map.get("Main Wipe", 150)
    arc_wipe = kpi_map.get("Arc Wipe", 15)
    
    if main_wipe < 100 or main_wipe > 200:
        mechanical_penalties.append(0.25)
    else:
        mechanical_penalties.append(0.0)
    
    if arc_wipe < 10 or arc_wipe > 25:
        mechanical_penalties.append(0.20)
    else:
        mechanical_penalties.append(0.0)
    
    # Weighted mechanical wear
    mech_weights = [0.25, 0.25, 0.20, 0.10, 0.10, 0.10]
    mechanical_wear = sum(p * w for p, w in zip(mechanical_penalties, mech_weights))
    
    # =========================================================================
    # 7. CALCULATE CBHI COMPONENT IMPACT
    # =========================================================================
    
    component_multipliers = {
        "Contacts (moving & arcing)": {"weight": 0.40, "High Risk": 0.35, "Medium Risk": 0.60, "Low Risk": 0.85, "Normal": 1.0},
        "SF6 Gas Chamber": {"weight": 0.25, "High Risk": 0.30, "Medium Risk": 0.55, "Low Risk": 0.80, "Normal": 1.0},
        "Operating Mechanism": {"weight": 0.20, "High Risk": 0.40, "Medium Risk": 0.65, "Low Risk": 0.85, "Normal": 1.0},
        "Coil": {"weight": 0.15, "High Risk": 0.25, "Medium Risk": 0.50, "Low Risk": 0.80, "Normal": 1.0}
    }
    
    cbhi_factor = 0.0
    total_weight = 0.0
    
    for component, status in cbhi_components.items():
        if component in component_multipliers:
            config = component_multipliers[component]
            weight = config["weight"]
            multiplier = config.get(status, 0.70)
            cbhi_factor += weight * multiplier
            total_weight += weight
    
    if total_weight > 0:
        cbhi_factor = cbhi_factor / total_weight
    else:
        cbhi_factor = cbhi_normalized
    
    # =========================================================================
    # 8. CALCULATE PHASE CONFIDENCE IMPACT
    # =========================================================================
    
    phase_weights = {
        "Pre-Contact Travel": 0.10,
        "Arcing Contact Engagement & Arc Initiation": 0.25,
        "Main Contact Conduction": 0.35,
        "Main Contact Parting & Arc Elongation": 0.20,
        "Final Open State": 0.10
    }
    
    phase_factor = 0.0
    phase_total_weight = 0.0
    critical_phase_failure = False
    
    for phase in phases_list:
        conf = float(safe_get(phase, "confidence", default=75))
        name = safe_get(phase, "name", default="")
        
        weight = 0.15  # Default weight
        for pname, pw in phase_weights.items():
            if pname.lower() in name.lower() or name.lower() in pname.lower():
                weight = pw
                break
        
        # Normalize confidence to factor (0-1)
        conf_factor = conf / 100.0
        
        # Critical phase failure detection
        if conf < 30:
            critical_phase_failure = True
            conf_factor *= 0.5  # Double penalty for critical failure
        elif conf < 50:
            conf_factor *= 0.75
        
        phase_factor += weight * conf_factor
        phase_total_weight += weight
    
    if phase_total_weight > 0:
        phase_factor = phase_factor / phase_total_weight
    else:
        phase_factor = 0.75
    
    # =========================================================================
    # 9. CALCULATE DEFECT-SPECIFIC DEGRADATION
    # =========================================================================
    
    # Get defect multiplier based on AI verdict
    defect_multiplier = get_defect_multiplier(fault_label)
    
    # Apply severity reduction
    severity_factor = SEVERITY_REDUCTION.get(ai_severity, 0.85)
    
    # Confidence-weighted defect impact
    # Higher confidence = more trust in the defect diagnosis
    confidence_weight = 0.3 + 0.7 * ai_confidence  # Range: 0.3-1.0
    
    if is_healthy:
        # Healthy: Use baseline with minor adjustments
        defect_impact = 1.0 - (1.0 - defect_multiplier) * (1.0 - ai_confidence) * 0.3
    else:
        # Faulty: Full defect impact weighted by confidence
        defect_impact = defect_multiplier * confidence_weight + (1.0 - confidence_weight) * 0.6
    
    defect_impact *= severity_factor
    
    # =========================================================================
    # 10. EXTRACT DEVIATION PERCENTAGE FROM SEVERITY REASON
    # =========================================================================
    
    import re
    deviation_penalty = 0.0
    if severity_reason:
        percentages = re.findall(r'(\d+)%', severity_reason)
        if percentages:
            max_dev = max(int(p) for p in percentages)
            if max_dev > 500:
                deviation_penalty = 0.40
            elif max_dev > 300:
                deviation_penalty = 0.30
            elif max_dev > 150:
                deviation_penalty = 0.20
            elif max_dev > 50:
                deviation_penalty = 0.10
    
    # =========================================================================
    # 11. COMBINE ALL FACTORS INTO FINAL RUL
    # =========================================================================
    
    # Combined wear factor (electrical has higher weight for contact issues)
    if "contact" in fault_label.lower() or "wear" in fault_label.lower():
        combined_wear = 0.65 * electrical_wear + 0.35 * mechanical_wear
    elif "mechanism" in fault_label.lower() or "damping" in fault_label.lower() or "linkage" in fault_label.lower():
        combined_wear = 0.35 * electrical_wear + 0.65 * mechanical_wear
    else:
        combined_wear = 0.50 * electrical_wear + 0.50 * mechanical_wear
    
    # Health factor combining all components
    # Weights: Defect (35%), CBHI (25%), Wear (25%), Phase (15%)
    health_factor = (
        0.35 * defect_impact +
        0.25 * cbhi_factor +
        0.25 * (1.0 - combined_wear) +
        0.15 * phase_factor
    )
    
    # Apply deviation penalty
    health_factor *= (1.0 - deviation_penalty)
    
    # Apply critical phase penalty
    if critical_phase_failure:
        health_factor *= 0.70
    
    # Clamp health factor
    health_factor = clamp(health_factor, 0.05, 1.0)
    
    # Calculate base RUL
    if is_healthy:
        base_rul = HEALTHY_BASELINE * health_factor
    else:
        # Non-healthy: Start from reduced baseline
        base_rul = (MAX_CYCLES * 0.6) * health_factor
    
    # Apply Weibull-like degradation curve for acceleration
    beta = 1.3 if is_healthy else 1.6  # Steeper curve for faulty
    final_rul = base_rul * math.pow(health_factor, beta - 1)
    
    # Ensure minimum floor
    final_rul = max(MIN_CYCLES, min(MAX_CYCLES, final_rul))
    
    # =========================================================================
    # 12. CALCULATE TIGHT UNCERTAINTY BOUNDS
    # =========================================================================
    
    # Base uncertainty factors
    uncertainty_components = []
    
    # Factor 1: AI Confidence uncertainty (inversely proportional)
    ai_uncertainty = (1.0 - ai_confidence) * 0.08
    uncertainty_components.append(ai_uncertainty)
    
    # Factor 2: Phase confidence variance
    phase_confs = [float(safe_get(p, "confidence", default=75)) for p in phases_list]
    if len(phase_confs) >= 2:
        phase_variance = (max(phase_confs) - min(phase_confs)) / 100.0
        uncertainty_components.append(phase_variance * 0.06)
    
    # Factor 3: CBHI vs AI discrepancy
    cbhi_health = cbhi_factor
    ai_health = defect_impact
    discrepancy = abs(cbhi_health - ai_health)
    uncertainty_components.append(discrepancy * 0.05)
    
    # Factor 4: KPI spread (how many KPIs are in warning/critical range)
    warning_count = 0
    for name, val in kpi_map.items():
        if name == "DLRO Value" and val > 100:
            warning_count += 1
        if name == "Closing Time" and val > 110:
            warning_count += 1
        if name == "Opening Time" and val > 50:
            warning_count += 1
        if name in ["Peak Trip Coil 1 Current", "Peak Trip Coil 2 Current"] and val < 2.0:
            warning_count += 1
    uncertainty_components.append(warning_count * 0.025)
    
    # Base irreducible uncertainty (5%)
    base_uncertainty_ratio = 0.05
    
    # Total uncertainty ratio
    total_uncertainty_ratio = base_uncertainty_ratio + sum(uncertainty_components)
    total_uncertainty_ratio = clamp(total_uncertainty_ratio, 0.08, 0.25)  # Tight bounds: 8-25%
    
    # Calculate uncertainty value
    uncertainty_value = final_rul * total_uncertainty_ratio
    
    # Round to sensible values
    uncertainty_value = int(round(uncertainty_value / 25) * 25)
    uncertainty_value = max(50, min(uncertainty_value, int(final_rul * 0.30)))
    
    # =========================================================================
    # 13. CALCULATE FINAL RANGE
    # =========================================================================
    
    estimated_cycles = int(round(final_rul / 10) * 10)
    low_range = max(MIN_CYCLES, estimated_cycles - uncertainty_value)
    high_range = min(MAX_CYCLES, estimated_cycles + uncertainty_value)
    
    # Ensure reasonable gap (minimum 10% of estimate or 100 cycles)
    min_gap = max(100, int(estimated_cycles * 0.10))
    if high_range - low_range < min_gap:
        mid = (low_range + high_range) // 2
        low_range = max(MIN_CYCLES, mid - min_gap // 2)
        high_range = min(MAX_CYCLES, mid + min_gap // 2)
    
    # Safety checks
    if low_range >= high_range:
        low_range = max(MIN_CYCLES, high_range - min_gap)
    
    # Ensure estimate is within range
    estimated_cycles = clamp(estimated_cycles, low_range, high_range)
    
    # =========================================================================
    # 14. RETURN SIMPLIFIED OUTPUT
    # =========================================================================
    
    return {
        "rulEstimate": f"{low_range}-{high_range} cycles",
        "uncertainty": f"±{uncertainty_value} cycles"
    }
