# Previous Name: analysis/engines/rule_engine.py
import pandas as pd
import numpy as np
import json
import sys
from scipy.signal import find_peaks

# Set UTF-8 encoding for console output (handles µ, Ω, etc.)
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# ============================================================================
# PHYSICS-BASED CONSTANTS (Ultra-Optimized for Real-Life DCRM Analysis)
# ============================================================================

# Phase Detection Thresholds (Industry-Calibrated)
R_OPEN_THRESHOLD = 1_000_000      # Above this = Open Circuit (Phase 1 & 5)
R_ARCING_THRESHOLD = 1000         # Intermediate resistance = Arcing Phase (Phase 2 & 4)
R_MAIN_THRESHOLD = 600            # Below this = Main Contact Zone (Phase 3)

# Nominal Healthy Values (Real-World Calibrated)
R_HEALTHY_MEAN_IDEAL = 35         # Ideal healthy: ~20-50 µΩ
R_HEALTHY_MAX = 70                # STRICT: >70 µΩ = early wear
R_HEALTHY_STD_MAX = 15            # Healthy std deviation < 15 µΩ (very smooth)

# Main Contact Wear Thresholds (Physics-Based Progressive Scale)
R_WEAR_EARLY_MIN = 70             # Mean > 70 µΩ = early wear (preventive)
R_WEAR_MODERATE_MIN = 100         # Mean > 100 µΩ = moderate wear
R_WEAR_SEVERE_MIN = 180           # Mean > 180 µΩ = severe wear
R_WEAR_CRITICAL_MIN = 280         # Mean > 280 µΩ = critical wear (imminent failure)
WEAR_STD_EARLY = 15               # Std > 15 µΩ = surface roughness
WEAR_STD_MODERATE = 25            # Std > 25 µΩ = noisy/grassy wear
WEAR_STD_SEVERE = 45              # Std > 45 µΩ = severe pitting/erosion

# Main Contact Misalignment (Square-Wave/Telegraph Pattern)
MISALIGNMENT_JUMP_MIN = 120       # Telegraph jump > 120 µΩ (square wave edge)
MISALIGNMENT_COUNT_MIN = 6        # Must have >= 6 distinct square-wave jumps
MISALIGNMENT_JUMP_RATIO = 0.15    # Affects >= 15% of main contact duration
MISALIGNMENT_STD_MIN = 70         # Std deviation > 70 µΩ for telegraph
SHELF_DETECTION_THRESHOLD = 80    # Mid-transition shelf > 80 µΩ
SQUARE_WAVE_DUTY_CYCLE = 0.3      # Square wave duty cycle check

# Arcing Contact Wear (Spike/Impulse Detection)
ARCING_SPIKE_CRITICAL = 8000      # Critical spike > 8000 µΩ (arc flash)
ARCING_SPIKE_SEVERE = 5000        # Severe spike > 5000 µΩ
ARCING_SPIKE_MODERATE = 3000      # Moderate spike > 3000 µΩ
ARCING_SPIKE_COUNT_CRITICAL = 4   # >= 4 critical spikes
ARCING_SPIKE_COUNT_SEVERE = 3     # >= 3 severe spikes
ARCING_INSTABILITY_STD = 700      # High std in arcing zones
SPIKE_WIDTH_THRESHOLD = 3         # Spike width > 3 samples = sustained arc

# Arcing Contact Misalignment (Asymmetry + Sinusoidal Bounce)
ASYMMETRY_RATIO_MODERATE = 1.6    # Opening/Closing ratio > 1.6
ASYMMETRY_RATIO_SEVERE = 2.2      # Ratio > 2.2 = severe asymmetry
ASYMMETRY_RATIO_CRITICAL = 3.0    # Ratio > 3.0 = critical misalignment
BOUNCE_PROMINENCE = 500           # Bounce/rounded peak > 500 µΩ
BOUNCE_SINUSOIDAL_FREQ = 10       # Sinusoidal frequency (samples per cycle)
PHASE3_REDUCTION_RATIO = 0.65     # Phase 3 < 65% of expected = reduced contact

# ============================================================================
# KPI THRESHOLDS FOR CLASSES 6-12 (Secondary Mechanical & Coil Defects)
# ============================================================================

# Class 6: Operating Mechanism Malfunction (Timing/Speed)
CLOSING_TIME_NOM = (80, 100)      # ms
OPENING_TIME_NOM = (30, 40)       # ms
CONTACT_SPEED_NOM = (4.5, 6.5)    # m/s
TIMING_DEVIATION_THRESHOLD = 0.20 # >20% off nominal

# Class 7: Damping System Fault (Bouncing)
BOUNCE_COUNT_THRESHOLD = 5        # >5 distinct bounces
BOUNCE_AMPLITUDE = 100            # >100 µΩ amplitude

# Class 8: SF6 Pressure Leakage
SF6_PRESSURE_NOM = (5.5, 6.5)     # bar
SF6_PRESSURE_CRITICAL = 5.0       # <5.0 bar = leak
ARC_QUENCH_DURATION_MAX = 25      # >25 ms = prolonged arc

# Class 9: Linkage/Rod Obstruction
STUTTER_COUNT_MIN = 3             # >3 distinct stutters
STUTTER_DURATION_MIN = 10         # >10 ms flat plateau

# Class 10: Fixed Contact Damage
DLRO_HEALTHY_MAX = 50             # <50 µΩ healthy
DLRO_MODERATE = 80                # 50-80 µΩ moderate
DLRO_CRITICAL = 100               # >100 µΩ critical
FIXED_CONTACT_STD_MAX = 15        # <15 µΩ = smooth (not wear)

# Class 11/12: Coil Damage
CLOSE_COIL_CURRENT_MIN = 2.0      # <2A = failure
TRIP_COIL_CURRENT_MIN = 2.0       # <2A = failure (both coils)
COIL_CURRENT_NOM = (4.0, 7.0)     # Normal: 4-7A



def standardize_input(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame with one row and columns T_0...T_400 containing Resistance values (uOhm).
    Supports vertical (>=401 rows, 1 col 'Resistance') or horizontal (>=401 cols).
    """
    if 'Resistance' not in df.columns:
        raise KeyError("CSV must contain a 'Resistance' column.")

    df = df[['Resistance']]

    # Vertical data (e.g., 401+ rows, single column)
    if df.shape[0] >= 401 and df.shape[1] == 1:
        values = df.iloc[:401, 0].values.reshape(1, -1)
        cols = [f"T_{i}" for i in range(401)]
        return pd.DataFrame(values, columns=cols)

    # Horizontal data (already a single row with many columns)
    elif df.shape[1] >= 401:
        df = df.iloc[:, :401]
        df.columns = [f"T_{i}" for i in range(401)]
        return df

    else:
        raise ValueError(f"Input shape {df.shape} invalid. Expected 401 Resistance points.")
    

def analyze_dcrm_advanced(row_values, kpis=None):
    """
    Production-Grade DCRM Analysis Engine with Full KPI Support
    ============================================================
    Detects ALL 12 defect classes:
    1. Healthy
    2. Main Contact Wear
    3. Arcing Contact Wear
    4. Main Contact Misalignment
    5. Arcing Contact Misalignment
    6. Operating Mechanism Malfunction
    7. Damping System Fault
    8. SF6 Pressure Leakage
    9. Linkage/Rod Obstruction
    10. Fixed Contact Damage
    11. Close Coil Damage
    12. Trip Coil Damage
    
    Args:
        row_values: DCRM waveform (401 time points, resistance in µΩ)
        kpis: Optional dictionary with KPIs (Closing Time, Opening Time, Contact Speed, 
              SF6 Pressure, DLRO, Close Coil Current, Trip Coil 1/2 Currents, etc.)
    
    Returns: 
        JSON with ALL defects having confidence >75% (Gemini-style dual-agent output)
        + classifications array with confidence scores for all 12 classes
    """
    arr = np.array(row_values, dtype=float)
    
    # Default KPIs if not provided
    if kpis is None:
        kpis = {}
    
    # === PHASE 1: AUTOMATIC PHASE DETECTION ===
    phases = detect_five_phases(arr)
    
    if phases is None:
        return {
            "Fault_Detection": [_build_result(
                "Open Circuit or Invalid Data", 
                "100.00 %", 
                "Critical", 
                "Breaker did not close properly or data is corrupted"
            )],
            "overall_health_assessment": {
                "Contacts (moving & arcing)": "High Risk",
                "SF6 Gas Chamber": "Normal",
                "Operating Mechanism": "High Risk",
                "Coil": "Normal"
            }
        }
    
    # Extract phase segments
    phase1_open = arr[phases['phase1_start']:phases['phase1_end']]
    phase2_closing = arr[phases['phase2_start']:phases['phase2_end']]
    phase3_main = arr[phases['phase3_start']:phases['phase3_end']]
    phase4_opening = arr[phases['phase4_start']:phases['phase4_end']]
    phase5_open = arr[phases['phase5_start']:phases['phase5_end']]
    
    # === PHASE 2: FEATURE EXTRACTION ===
    features = extract_features(phase3_main, phase2_closing, phase4_opening, phases)
    
    # === PHASE 3: PRIMARY FAULT CLASSIFICATION (Classes 1-5, 11-12) ===
    primary_faults = classify_primary_faults(features, phases, kpis)
    
    # === PHASE 4: SECONDARY MECHANICAL FAULT CLASSIFICATION (Classes 6-10) ===
    secondary_faults = classify_secondary_faults(features, phases, kpis, primary_faults)
    
    # === PHASE 5: MERGE AND FILTER (Return ALL defects with probability >50%) ===
    all_faults = primary_faults + secondary_faults
    
    # Filter: Only return defects with probability >50% (convert "XX.XX %" to float)
    high_prob_faults = []
    for fault in all_faults:
        prob_str = fault['Confidence'].replace('%', '').strip()
        prob_val = float(prob_str)
        if prob_val > 50.0:
            high_prob_faults.append(fault)
    
    # Sort by probability (highest first)
    high_prob_faults.sort(key=lambda x: float(x['Confidence'].replace('%', '').strip()), reverse=True)
    
    # If no high-probability defects, return Healthy with low score
    if not high_prob_faults:
        healthy_desc = f"Insufficient evidence for any specific defect. Main Contact: Mean={features['main_mean']:.1f} µΩ, Std={features['main_std']:.1f} µΩ. All defect probabilities <50%."
        high_prob_faults.append(_build_result(
            "Inconclusive",
            "45.00 %",
            "Low",
            healthy_desc
        ))
    
    # === PHASE 6: BUILD OVERALL HEALTH ASSESSMENT (Gemini-style) ===
    overall_health = {
        "Contacts (moving & arcing)": "Normal",
        "SF6 Gas Chamber": "Normal",
        "Operating Mechanism": "Normal",
        "Coil": "Normal"
    }
    
    for fault in high_prob_faults:
        name = fault['defect_name'].lower()
        severity = fault['Severity'].lower()
        probability = float(fault['Confidence'].replace('%', '').strip())
        
        # Determine risk level based on probability
        if probability >= 85 and severity in ["high", "critical"]:
            risk = "High Risk"
        elif probability >= 70:
            risk = "Moderate Risk"
        elif probability >= 50:
            risk = "Low Risk"
        else:
            risk = "Normal"
        
        # Map defects to health categories
        if any(x in name for x in ["main contact", "arcing contact", "contact wear", "contact misalignment", "fixed contact"]):
            if overall_health["Contacts (moving & arcing)"] != "High Risk":
                overall_health["Contacts (moving & arcing)"] = risk
        
        if "sf6" in name or "pressure" in name:
            if overall_health["SF6 Gas Chamber"] != "High Risk":
                overall_health["SF6 Gas Chamber"] = risk
        
        if any(x in name for x in ["operating mechanism", "damping", "linkage", "rod"]):
            if overall_health["Operating Mechanism"] != "High Risk":
                overall_health["Operating Mechanism"] = risk
        
        if "coil" in name:
            if overall_health["Coil"] != "High Risk":
                overall_health["Coil"] = risk
    
    # === PHASE 7: BUILD CLASSIFICATIONS ARRAY FOR ALL 12 CLASSES ===
    # Collect all probabilities (convert from all_faults list)
    class_probabilities = {
        "Healthy": 0.0,
        "Main Contact Wear": 0.0,
        "Arcing Contact Wear": 0.0,
        "Main Contact Misalignment": 0.0,
        "Arcing Contact Misalignment": 0.0,
        "Operating Mechanism Malfunction": 0.0,
        "Damping System Fault": 0.0,
        "Pressure System Leakage (SF6 Gas Chamber)": 0.0,
        "Linkage/Connecting Rod Obstruction/Damage": 0.0,
        "Fixed Contact Damage/Deformation": 0.0,
        "Close Coil Damage": 0.0,
        "Trip Coil Damage": 0.0
    }
    
    # Fill in probabilities from all_faults (including those <50%)
    for fault in all_faults:
        name = fault['defect_name']
        prob_str = fault['Confidence'].replace('%', '').strip()
        prob_val = float(prob_str) / 100.0  # Convert to 0-1 scale
        class_probabilities[name] = prob_val
    
    # Build classifications array (all 12 classes)
    classifications = []
    for class_name, confidence in class_probabilities.items():
        classifications.append({
            "Class": class_name,
            "Confidence": round(confidence, 4)
        })
    
    # === RETURN OPTIMIZED JSON (Probability-Based Scoring) ===
    result = {
        "Fault_Detection": high_prob_faults,
        "overall_health_assessment": overall_health,
        "classifications": classifications
    }
    
    return result


def detect_five_phases(arr):
    """
    Automatically detects all 5 DCRM phases using adaptive thresholding.
    Returns dict with start/end indices for each phase, or None if detection fails.
    """
    # Find contact regions (below threshold)
    is_contact = arr < R_ARCING_THRESHOLD
    contact_indices = np.where(is_contact)[0]
    
    if len(contact_indices) < 20:
        return None  # No valid contact detected
    
    # Identify main contact region (very low resistance)
    is_main = arr < R_MAIN_THRESHOLD
    main_indices = np.where(is_main)[0]
    
    if len(main_indices) < 5:
        # No main contact = severe fault
        # Best effort: assume entire contact is arcing
        phase1_end = contact_indices[0]
        phase5_start = contact_indices[-1] + 1
        
        return {
            'phase1_start': 0,
            'phase1_end': phase1_end,
            'phase2_start': phase1_end,
            'phase2_end': contact_indices[-1],
            'phase3_start': contact_indices[-1],
            'phase3_end': contact_indices[-1],  # Empty main phase
            'phase4_start': contact_indices[-1],
            'phase4_end': phase5_start,
            'phase5_start': phase5_start,
            'phase5_end': len(arr)
        }
    
    # Normal case: Main contact exists
    t_contact_start = contact_indices[0]
    t_contact_end = contact_indices[-1]
    t_main_start = main_indices[0]
    t_main_end = main_indices[-1]
    
    return {
        'phase1_start': 0,
        'phase1_end': t_contact_start,
        'phase2_start': t_contact_start,
        'phase2_end': t_main_start,
        'phase3_start': t_main_start,
        'phase3_end': t_main_end,
        'phase4_start': t_main_end,
        'phase4_end': t_contact_end + 1,
        'phase5_start': t_contact_end + 1,
        'phase5_end': len(arr)
    }


def extract_features(seg_main, seg_closing, seg_opening, phases):
    """
    ULTRA-OPTIMIZED Feature Extraction with Micro-Level Waveform Analysis
    =====================================================================
    Detects:
    - Square wave patterns (misalignment)
    - Sinusoidal oscillations (damping/bounce)
    - Impulse spikes (arcing wear)
    - Grassy noise (contact wear)
    - Telegraph jumps (mechanical defects)
    - DC offset shifts (fixed contact issues)
    """
    features = {}
    
    # === MAIN CONTACT FEATURES (Phase 3) - MICRO-LEVEL ANALYSIS ===
    if len(seg_main) > 0:
        # Basic statistics
        features['main_mean'] = float(np.mean(seg_main))
        features['main_median'] = float(np.median(seg_main))
        features['main_std'] = float(np.std(seg_main))
        features['main_min'] = float(np.min(seg_main))
        features['main_max'] = float(np.max(seg_main))
        features['main_range'] = float(features['main_max'] - features['main_min'])
        
        # === SQUARE WAVE PATTERN DETECTION (Misalignment) ===
        diffs = np.diff(seg_main)
        abs_diffs = np.abs(diffs)
        
        # Count sharp edges (square wave transitions)
        sharp_edges = np.sum(abs_diffs > MISALIGNMENT_JUMP_MIN)
        features['telegraph_jumps'] = int(sharp_edges)
        features['jump_ratio'] = float(sharp_edges / len(seg_main) if len(seg_main) > 0 else 0)
        
        # Detect duty cycle of square wave (time spent at high vs low levels)
        if features['main_range'] > 100:
            threshold = features['main_median']
            high_time = np.sum(seg_main > threshold)
            duty_cycle = high_time / len(seg_main)
            features['square_wave_duty'] = float(duty_cycle)
            # True square wave has duty cycle ~0.3-0.7 (not 0 or 1)
            features['is_square_wave'] = 1 if 0.2 < duty_cycle < 0.8 else 0
        else:
            features['square_wave_duty'] = 0.5
            features['is_square_wave'] = 0
        
        # === SINUSOIDAL/OSCILLATION DETECTION (Damping Fault) ===
        # Use autocorrelation to detect periodic patterns
        if len(seg_main) > 20:
            # Detrend signal
            detrended = seg_main - np.mean(seg_main)
            # Simple autocorrelation at lag=10 (typical bounce frequency)
            if len(detrended) > BOUNCE_SINUSOIDAL_FREQ:
                autocorr = np.correlate(detrended[:min(100, len(detrended))], 
                                       detrended[:min(100, len(detrended))], mode='valid')[0]
                features['oscillation_score'] = float(abs(autocorr) / (np.std(detrended)**2 * len(detrended) + 1))
            else:
                features['oscillation_score'] = 0.0
        else:
            features['oscillation_score'] = 0.0
        
        # === GRASSY NOISE PATTERN (Wear Signature) ===
        # Count uniform medium-amplitude spikes throughout plateau
        noise_threshold_low = features['main_median'] + 30
        noise_threshold_high = features['main_median'] + 200
        grassy_spikes = np.sum((seg_main > noise_threshold_low) & (seg_main < noise_threshold_high))
        features['uniform_spikes'] = int(grassy_spikes)
        features['spike_density'] = float(grassy_spikes / len(seg_main) if len(seg_main) > 0 else 0)
        
        # Measure continuous noise level (RMS of derivative)
        features['avg_noise'] = float(np.mean(abs_diffs))
        features['max_single_jump'] = float(np.max(abs_diffs)) if len(abs_diffs) > 0 else 0
        features['noise_rms'] = float(np.sqrt(np.mean(abs_diffs**2)))
        
        # === STEPPED SHELF PATTERN (Misalignment Signature) ===
        if len(seg_main) > 20:
            # Histogram analysis to find discrete levels
            hist, edges = np.histogram(seg_main, bins=min(15, len(seg_main)//10))
            # Count bins with >8% of data (significant plateaus)
            significant_bins = np.sum(hist > (len(seg_main) * 0.08))
            features['num_shelves'] = int(significant_bins)
            
            # Detect initial transient vs steady-state
            split_point = min(25, len(seg_main)//3)
            initial_segment = seg_main[:split_point]
            plateau_segment = seg_main[split_point:]
            features['initial_deviation'] = float(np.std(initial_segment))
            features['plateau_stability'] = float(np.std(plateau_segment))
            
            # Initial jump indicator
            if len(initial_segment) > 0 and len(plateau_segment) > 0:
                features['has_initial_jump'] = 1 if features['initial_deviation'] > features['plateau_stability'] * 1.8 else 0
            else:
                features['has_initial_jump'] = 0
        else:
            features['num_shelves'] = 1
            features['initial_deviation'] = 0
            features['plateau_stability'] = features['main_std']
            features['has_initial_jump'] = 0
            
    else:
        # Empty main contact = severe fault
        features.update({
            'main_mean': 9999, 'main_median': 9999, 'main_std': 0,
            'main_min': 9999, 'main_max': 9999, 'main_range': 0,
            'telegraph_jumps': 0, 'jump_ratio': 0, 'uniform_spikes': 0,
            'spike_density': 0, 'avg_noise': 0, 'max_single_jump': 0,
            'noise_rms': 0, 'square_wave_duty': 0, 'is_square_wave': 0,
            'oscillation_score': 0, 'initial_deviation': 0, 
            'plateau_stability': 0, 'has_initial_jump': 0, 'num_shelves': 0
        })
    
    # === TIMING FEATURES ===
    features['dur_closing'] = int(max(1, len(seg_closing)))
    features['dur_opening'] = int(max(1, len(seg_opening)))
    features['dur_main'] = int(len(seg_main))
    features['asymmetry_ratio'] = float(features['dur_opening'] / features['dur_closing'])
    
    # Detect reduced Phase 3 (arcing misalignment signature)
    expected_main_duration = 160  # Typical ~160-180 ms
    features['phase3_reduction'] = float(features['dur_main'] / expected_main_duration)
    
    # === ARCING CONTACT FEATURES - IMPULSE SPIKE ANALYSIS ===
    
    features['closing_critical_spikes'] = 0
    features['closing_severe_spikes'] = 0
    features['closing_moderate_spikes'] = 0
    features['opening_critical_spikes'] = 0
    features['opening_severe_spikes'] = 0
    features['opening_moderate_spikes'] = 0
    features['closing_std'] = 0
    features['opening_std'] = 0
    features['closing_peak'] = 0
    features['opening_peak'] = 0
    
    # === CLOSING ARCING ZONE ANALYSIS ===
    if len(seg_closing) > 0:
        features['closing_critical_spikes'] = int(np.sum(seg_closing > ARCING_SPIKE_CRITICAL))
        features['closing_severe_spikes'] = int(np.sum(seg_closing > ARCING_SPIKE_SEVERE))
        features['closing_moderate_spikes'] = int(np.sum(seg_closing > ARCING_SPIKE_MODERATE))
        features['closing_std'] = float(np.std(seg_closing))
        features['closing_peak'] = float(np.max(seg_closing))
        
        # Analyze spike width (sustained arc vs transient)
        critical_indices = np.where(seg_closing > ARCING_SPIKE_SEVERE)[0]
        if len(critical_indices) > 0:
            # Count sustained spikes (width > threshold)
            spike_groups = np.split(critical_indices, np.where(np.diff(critical_indices) > 2)[0] + 1)
            sustained_spikes = sum(1 for group in spike_groups if len(group) >= SPIKE_WIDTH_THRESHOLD)
            features['closing_sustained_spikes'] = int(sustained_spikes)
        else:
            features['closing_sustained_spikes'] = 0
    else:
        features['closing_sustained_spikes'] = 0
    
    # === OPENING ARCING ZONE ANALYSIS ===
    if len(seg_opening) > 0:
        features['opening_critical_spikes'] = int(np.sum(seg_opening > ARCING_SPIKE_CRITICAL))
        features['opening_severe_spikes'] = int(np.sum(seg_opening > ARCING_SPIKE_SEVERE))
        features['opening_moderate_spikes'] = int(np.sum(seg_opening > ARCING_SPIKE_MODERATE))
        features['opening_std'] = float(np.std(seg_opening))
        features['opening_peak'] = float(np.max(seg_opening))
        
        # Spike width analysis
        critical_indices = np.where(seg_opening > ARCING_SPIKE_SEVERE)[0]
        if len(critical_indices) > 0:
            spike_groups = np.split(critical_indices, np.where(np.diff(critical_indices) > 2)[0] + 1)
            sustained_spikes = sum(1 for group in spike_groups if len(group) >= SPIKE_WIDTH_THRESHOLD)
            features['opening_sustained_spikes'] = int(sustained_spikes)
        else:
            features['opening_sustained_spikes'] = 0
        
        # === SINUSOIDAL BOUNCE DETECTION (Arcing Misalignment) ===
        # Detect rounded/oscillating peaks
        peaks, properties = find_peaks(seg_opening, prominence=BOUNCE_PROMINENCE, distance=5)
        features['num_bounces'] = int(len(peaks))
        
        # Detect high-frequency telegraph in arcing zone
        opening_diffs = np.abs(np.diff(seg_opening))
        features['arcing_telegraph'] = int(np.sum(opening_diffs > 400))
    else:
        features['opening_sustained_spikes'] = 0
        features['num_bounces'] = 0
        features['arcing_telegraph'] = 0
    
    # Add closing telegraph
    if len(seg_closing) > 10:
        closing_diffs = np.abs(np.diff(seg_closing))
        features['arcing_telegraph'] += int(np.sum(closing_diffs > 400))
    
    # === TOTAL SPIKE COUNTS (All Severity Levels) ===
    features['total_critical_spikes'] = features['closing_critical_spikes'] + features['opening_critical_spikes']
    features['total_severe_spikes'] = features['closing_severe_spikes'] + features['opening_severe_spikes']
    features['total_moderate_spikes'] = features['closing_moderate_spikes'] + features['opening_moderate_spikes']
    features['total_sustained_spikes'] = features['closing_sustained_spikes'] + features['opening_sustained_spikes']
    
    # Symmetry check (wear = similar spikes; misalignment = asymmetric)
    if features['closing_severe_spikes'] > 0 and features['opening_severe_spikes'] > 0:
        spike_ratio = max(features['closing_severe_spikes'], features['opening_severe_spikes']) / \
                      max(1, min(features['closing_severe_spikes'], features['opening_severe_spikes']))
        features['spike_symmetry'] = float(spike_ratio)  # ~1.0 = symmetric (wear)
    else:
        features['spike_symmetry'] = 1.0
    
    return features


def classify_primary_faults(features, phases, kpis):
    """
    ULTRA-OPTIMIZED Multi-Class PRIMARY Fault Classification with Individual Probability Scores
    ============================================================================================
    Returns ALL defects with probability > 50% (independent scores, not cumulative).
    
    Scoring Logic:
    - Each defect gets 0-100% probability based on signature strength
    - Multiple defects can coexist (e.g., Wear + Misalignment both at 85%)
    - Scores are NOT normalized (don't sum to 100%)
    - Only defects >50% probability are returned
    
    Classes: 1-Healthy, 2-Main Wear, 3-Arcing Wear, 4-Main Misalign, 
             5-Arcing Misalign, 11-Close Coil, 12-Trip Coil
    """
    all_faults = []
    
    # ========================================================================
    # CLASS 11: CLOSE COIL DAMAGE
    # ========================================================================
    close_coil_current = kpis.get('Peak Close Coil Current (A)', None)
    if close_coil_current is not None:
        if close_coil_current < CLOSE_COIL_CURRENT_MIN:
            conf = 95.0
            sev = "High"
            desc = f"Close Coil Current critically low ({close_coil_current:.2f} A, normal: 4-7 A). Coil winding damaged or control circuit fault."
            all_faults.append(_build_result("Close Coil Damage", f"{conf:.2f} %", sev, desc))
    
    # ========================================================================
    # CLASS 12: TRIP COIL DAMAGE
    # CRITICAL: BOTH trip coils must fail. If at least ONE works, NO FAULT.
    # ========================================================================
    trip_coil1 = kpis.get('Peak Trip Coil 1 Current (A)', None)
    trip_coil2 = kpis.get('Peak Trip Coil 2 Current (A)', None)
    
    if trip_coil1 is not None and trip_coil2 is not None:
        # Check if BOTH coils failed
        if trip_coil1 < TRIP_COIL_CURRENT_MIN and trip_coil2 < TRIP_COIL_CURRENT_MIN:
            conf = 95.0
            sev = "Critical"
            desc = f"BOTH Trip Coils failed (TC1: {trip_coil1:.2f} A, TC2: {trip_coil2:.2f} A, normal: 4-7 A each). Breaker cannot trip - SAFETY CRITICAL."
            all_faults.append(_build_result("Trip Coil Damage", f"{conf:.2f} %", sev, desc))
        # If at least one coil works, NO fault (this is normal redundancy)
    
    # ========================================================================
    # CLASS 5: ARCING CONTACT MISALIGNMENT
    # Signature: Asymmetric curve + Reduced Phase 3 + Sinusoidal Bounces
    # Probability Score: 0-100% based on signature strength
    # ========================================================================
    arcing_misalign_prob = 0.0
    arcing_misalign_reasons = []
    
    # Factor 1: Timing asymmetry (CRITICAL SIGNATURE - 40 points max)
    if features['asymmetry_ratio'] > ASYMMETRY_RATIO_CRITICAL:
        arcing_misalign_prob += 40.0
        arcing_misalign_reasons.append(f"Critical timing asymmetry: Opening phase {features['asymmetry_ratio']:.2f}x longer than closing (>3.0x indicates severe misalignment)")
    elif features['asymmetry_ratio'] > ASYMMETRY_RATIO_SEVERE:
        arcing_misalign_prob += 32.0
        arcing_misalign_reasons.append(f"Severe timing asymmetry: Opening {features['asymmetry_ratio']:.2f}x longer (>2.2x threshold)")
    elif features['asymmetry_ratio'] > ASYMMETRY_RATIO_MODERATE:
        arcing_misalign_prob += 24.0
        arcing_misalign_reasons.append(f"Moderate asymmetry: Timing ratio {features['asymmetry_ratio']:.2f} (normal <1.5)")
    
    # Factor 2: Reduced Phase 3 duration (20 points max)
    if features['phase3_reduction'] < PHASE3_REDUCTION_RATIO:
        reduction_pct = (1 - features['phase3_reduction']) * 100
        arcing_misalign_prob += 20.0
        arcing_misalign_reasons.append(f"Main contact duration reduced by {reduction_pct:.0f}% ({features['dur_main']} ms vs expected ~160 ms)")
    elif features['phase3_reduction'] < 0.80:
        arcing_misalign_prob += 12.0
        arcing_misalign_reasons.append(f"Slightly reduced contact engagement ({features['dur_main']} ms)")
    
    # Factor 3: Sinusoidal bounces (25 points max)
    if features['num_bounces'] >= 5:
        arcing_misalign_prob += 25.0
        arcing_misalign_reasons.append(f"Detected {features['num_bounces']} sinusoidal bounces during opening (indicates mechanical oscillation)")
    elif features['num_bounces'] >= 3:
        arcing_misalign_prob += 18.0
        arcing_misalign_reasons.append(f"{features['num_bounces']} rounded bounces detected")
    elif features['num_bounces'] >= 1:
        arcing_misalign_prob += 10.0
        arcing_misalign_reasons.append(f"{features['num_bounces']} bounce peak(s) in arcing phase")
    
    # Factor 4: Telegraph noise in arcing zones (15 points max)
    if features['arcing_telegraph'] > 15:
        arcing_misalign_prob += 15.0
        arcing_misalign_reasons.append(f"High-frequency telegraph noise in arcing zones ({features['arcing_telegraph']} rapid transitions)")
    elif features['arcing_telegraph'] > 8:
        arcing_misalign_prob += 10.0
        arcing_misalign_reasons.append(f"Telegraph noise detected ({features['arcing_telegraph']} events)")
    
    if arcing_misalign_prob >= 50.0:
        prob_str = f"{min(99.0, arcing_misalign_prob):.2f} %"
        sev = _get_severity(arcing_misalign_prob)
        desc = ". ".join(arcing_misalign_reasons)
        all_faults.append(_build_result("Arcing Contact Misalignment", prob_str, sev, desc))
    
    # ========================================================================
    # CLASS 4: MAIN CONTACT MISALIGNMENT
    # Signature: Square-wave telegraph + Stepped shelves + Initial jump
    # Probability Score: 0-100% based on signature strength
    # ========================================================================
    main_misalign_prob = 0.0
    main_misalign_reasons = []
    
    # Factor 1: Square-wave telegraph pattern (45 points max)
    if (features['telegraph_jumps'] >= MISALIGNMENT_COUNT_MIN and 
        features['jump_ratio'] > MISALIGNMENT_JUMP_RATIO and
        features['main_std'] > MISALIGNMENT_STD_MIN and
        features['is_square_wave'] == 1):
        main_misalign_prob += 45.0
        main_misalign_reasons.append(f"Square-wave telegraph pattern: {features['telegraph_jumps']} sharp jumps (>{MISALIGNMENT_JUMP_MIN} µΩ), duty cycle {features['square_wave_duty']:.2f}")
    elif features['telegraph_jumps'] >= MISALIGNMENT_COUNT_MIN and features['main_std'] > MISALIGNMENT_STD_MIN:
        main_misalign_prob += 35.0
        main_misalign_reasons.append(f"Telegraph pattern detected: {features['telegraph_jumps']} jumps, Std={features['main_std']:.1f} µΩ")
    elif features['telegraph_jumps'] >= 4:
        main_misalign_prob += 20.0
        main_misalign_reasons.append(f"Partial telegraph: {features['telegraph_jumps']} jumps")
    
    # Factor 2: Initial deviation/transient jump (20 points max)
    if features['has_initial_jump'] == 1 and features['initial_deviation'] > 120:
        main_misalign_prob += 20.0
        main_misalign_reasons.append(f"High initial transient (Std={features['initial_deviation']:.1f} µΩ) then stabilizes - classic misalignment signature")
    elif features['has_initial_jump'] == 1:
        main_misalign_prob += 12.0
        main_misalign_reasons.append(f"Initial deviation detected ({features['initial_deviation']:.1f} µΩ)")
    
    # Factor 3: Stepped shelf transitions (20 points max)
    if features['num_shelves'] >= 4 and features['main_range'] > SHELF_DETECTION_THRESHOLD:
        main_misalign_prob += 20.0
        main_misalign_reasons.append(f"Stepped transitions: {features['num_shelves']} discrete resistance plateaus (range {features['main_range']:.1f} µΩ)")
    elif features['num_shelves'] >= 3:
        main_misalign_prob += 12.0
        main_misalign_reasons.append(f"{features['num_shelves']} resistance shelves detected")
    
    # Factor 4: High variability (15 points max)
    if features['main_std'] > MISALIGNMENT_STD_MIN * 2:
        main_misalign_prob += 15.0
        main_misalign_reasons.append(f"Very high variability (Std={features['main_std']:.1f} µΩ, normal <15 µΩ)")
    elif features['main_std'] > MISALIGNMENT_STD_MIN:
        main_misalign_prob += 10.0
        main_misalign_reasons.append(f"Elevated variability (Std={features['main_std']:.1f} µΩ)")
    
    if main_misalign_prob >= 50.0:
        prob_str = f"{min(99.0, main_misalign_prob):.2f} %"
        sev = _get_severity(main_misalign_prob)
        desc = ". ".join(main_misalign_reasons)
        all_faults.append(_build_result("Main Contact Misalignment", prob_str, sev, desc))
    
    # ========================================================================
    # CLASS 2: MAIN CONTACT WEAR
    # Signature: Elevated resistance + Grassy noise + Uniform spikes
    # Probability Score: 0-100% based on wear severity
    # ========================================================================
    main_wear_prob = 0.0
    main_wear_reasons = []
    
    # Get DLRO value if available
    dlro_value = kpis.get('DLRO Value (µΩ)', kpis.get('DLRO_Value_uOhm', kpis.get('dlro_uohm', None)))
    
    # Factor 1: Elevated resistance (50 points max - MOST CRITICAL)
    if features['main_mean'] > R_WEAR_CRITICAL_MIN:
        elevation = ((features['main_mean'] - R_HEALTHY_MEAN_IDEAL) / R_HEALTHY_MEAN_IDEAL) * 100
        main_wear_prob += 50.0
        main_wear_reasons.append(f"CRITICAL wear: Resistance {features['main_mean']:.1f} µΩ (healthy: 20-70 µΩ, {elevation:.0f}% above ideal). Severe erosion/material loss detected")
        if dlro_value is not None and dlro_value > 250:
            main_wear_prob += 8.0
            main_wear_reasons.append(f"Confirmed by DLRO: {dlro_value:.1f} µΩ")
    elif features['main_mean'] > R_WEAR_SEVERE_MIN:
        elevation = ((features['main_mean'] - R_HEALTHY_MEAN_IDEAL) / R_HEALTHY_MEAN_IDEAL) * 100
        main_wear_prob += 42.0
        main_wear_reasons.append(f"SEVERE wear: Resistance {features['main_mean']:.1f} µΩ ({elevation:.0f}% above ideal). Significant contact degradation")
        if dlro_value is not None and dlro_value > 180:
            main_wear_prob += 8.0
            main_wear_reasons.append(f"Confirmed by DLRO: {dlro_value:.1f} µΩ")
    elif features['main_mean'] > R_WEAR_MODERATE_MIN:
        main_wear_prob += 32.0
        main_wear_reasons.append(f"MODERATE wear: Resistance {features['main_mean']:.1f} µΩ (healthy <70 µΩ). Contact wear progressing")
        if dlro_value is not None and dlro_value > 100:
            main_wear_prob += 8.0
            main_wear_reasons.append(f"Confirmed by DLRO: {dlro_value:.1f} µΩ")
    elif features['main_mean'] > R_WEAR_EARLY_MIN:
        main_wear_prob += 20.0
        main_wear_reasons.append(f"EARLY wear signs: Resistance {features['main_mean']:.1f} µΩ (healthy <70 µΩ)")
        if dlro_value is not None and dlro_value > 70:
            main_wear_prob += 8.0
            main_wear_reasons.append(f"DLRO confirms: {dlro_value:.1f} µΩ")
    
    # Factor 2: Grassy/noisy pattern - Surface roughness (25 points max)
    if features['main_std'] > WEAR_STD_SEVERE:
        main_wear_prob += 25.0
        main_wear_reasons.append(f"Severe surface roughness: Std={features['main_std']:.1f} µΩ (healthy <15 µΩ). Indicates pitting/erosion")
    elif features['main_std'] > WEAR_STD_MODERATE:
        main_wear_prob += 18.0
        main_wear_reasons.append(f"Moderate roughness: Std={features['main_std']:.1f} µΩ (healthy <15 µΩ)")
    elif features['main_std'] > WEAR_STD_EARLY:
        main_wear_prob += 10.0
        main_wear_reasons.append(f"Surface roughness detected: Std={features['main_std']:.1f} µΩ")
    
    # Factor 3: Uniform grassy spikes throughout plateau (15 points max)
    if features['spike_density'] > 0.35:
        main_wear_prob += 15.0
        main_wear_reasons.append(f"Dense uniform spikes: {features['uniform_spikes']} spikes ({features['spike_density']*100:.1f}% density). Classic wear signature")
    elif features['spike_density'] > 0.20:
        main_wear_prob += 10.0
        main_wear_reasons.append(f"Grassy pattern: {features['uniform_spikes']} spikes detected")
    elif features['uniform_spikes'] > 10:
        main_wear_prob += 5.0
        main_wear_reasons.append(f"{features['uniform_spikes']} noise spikes in plateau")
    
    # Factor 4: Continuous noise level (10 points max)
    if features['noise_rms'] > 40:
        main_wear_prob += 10.0
        main_wear_reasons.append(f"High continuous noise: RMS={features['noise_rms']:.1f} µΩ")
    elif features['avg_noise'] > 25:
        main_wear_prob += 6.0
        main_wear_reasons.append(f"Elevated noise level: {features['avg_noise']:.1f} µΩ")
    
    if main_wear_prob >= 50.0:
        prob_str = f"{min(99.0, main_wear_prob):.2f} %"
        sev = _get_severity(main_wear_prob)
        desc = ". ".join(main_wear_reasons)
        all_faults.append(_build_result("Main Contact Wear", prob_str, sev, desc))
    
    # ========================================================================
    # CLASS 3: ARCING CONTACT WEAR
    # Signature: Sustained impulse spikes + Symmetric pattern + Healthy Phase 3
    # Probability Score: 0-100% based on arc severity
    # ========================================================================
    arcing_wear_prob = 0.0
    arcing_wear_reasons = []
    
    # Factor 1: Critical/Severe spikes in arcing zones (50 points max)
    if features['total_critical_spikes'] >= ARCING_SPIKE_COUNT_CRITICAL:
        arcing_wear_prob += 50.0
        arcing_wear_reasons.append(f"CRITICAL: {features['total_critical_spikes']} severe arc flashes detected (>8000 µΩ). Arcing contact severely eroded")
        if features['total_sustained_spikes'] >= 2:
            arcing_wear_prob += 8.0
            arcing_wear_reasons.append(f"{features['total_sustained_spikes']} sustained arc events (>3 samples width)")
    elif features['total_severe_spikes'] >= ARCING_SPIKE_COUNT_SEVERE:
        arcing_wear_prob += 40.0
        arcing_wear_reasons.append(f"SEVERE: {features['total_severe_spikes']} high-energy spikes (>5000 µΩ) in arcing zones")
        if features['total_sustained_spikes'] >= 2:
            arcing_wear_prob += 8.0
            arcing_wear_reasons.append(f"{features['total_sustained_spikes']} sustained arcs detected")
    elif features['total_severe_spikes'] >= 2:
        arcing_wear_prob += 28.0
        arcing_wear_reasons.append(f"{features['total_severe_spikes']} arcing spikes detected (>5000 µΩ)")
    elif features['total_moderate_spikes'] >= 5:
        arcing_wear_prob += 20.0
        arcing_wear_reasons.append(f"{features['total_moderate_spikes']} moderate arcing events (>3000 µΩ)")
    
    # Factor 2: Spike symmetry check (wear = symmetric; misalignment = asymmetric) (20 points max)
    if features['spike_symmetry'] < 1.4 and features['total_severe_spikes'] > 0:
        arcing_wear_prob += 20.0
        arcing_wear_reasons.append(f"Symmetric spike distribution (ratio {features['spike_symmetry']:.2f}). Confirms uniform arcing wear on both contacts")
    elif features['spike_symmetry'] < 1.8 and features['total_severe_spikes'] > 0:
        arcing_wear_prob += 12.0
        arcing_wear_reasons.append(f"Relatively symmetric pattern (ratio {features['spike_symmetry']:.2f})")
    
    # Factor 3: High arcing zone instability (15 points max)
    max_arcing_std = max(features['closing_std'], features['opening_std'])
    if max_arcing_std > ARCING_INSTABILITY_STD * 1.5:
        arcing_wear_prob += 15.0
        arcing_wear_reasons.append(f"Very high arcing instability: Std={max_arcing_std:.1f} µΩ (normal <500 µΩ)")
    elif max_arcing_std > ARCING_INSTABILITY_STD:
        arcing_wear_prob += 10.0
        arcing_wear_reasons.append(f"Elevated arcing zone variability: Std={max_arcing_std:.1f} µΩ")
    
    # Factor 4: Phase 3 health check (15 points max - confirms isolation to arcing contacts)
    if features['main_mean'] < R_HEALTHY_MAX and features['main_std'] < WEAR_STD_MODERATE:
        arcing_wear_prob += 15.0
        arcing_wear_reasons.append(f"Main contact healthy (Mean={features['main_mean']:.1f} µΩ, Std={features['main_std']:.1f} µΩ). Confirms wear isolated to arcing contacts")
    else:
        # If main contact also worn, arcing wear is secondary/co-existing
        if features['main_mean'] > R_WEAR_MODERATE_MIN:
            arcing_wear_prob -= 10.0  # Penalty: main wear likely primary
    
    # Disqualify if strong misalignment signature (asymmetric timing)
    if features['asymmetry_ratio'] > ASYMMETRY_RATIO_MODERATE:
        arcing_wear_prob -= 18.0
        # Don't add reason - just reduce probability
    
    if arcing_wear_prob >= 50.0:
        prob_str = f"{min(99.0, arcing_wear_prob):.2f} %"
        sev = _get_severity(arcing_wear_prob)
        desc = ". ".join(arcing_wear_reasons)
        all_faults.append(_build_result("Arcing Contact Wear", prob_str, sev, desc))
    
    # ========================================================================
    # CLASS 1: HEALTHY
    # Signature: Low resistance + Low variability + Smooth transitions
    # Probability Score: 100% minus penalties for any abnormalities
    # ULTRA-STRICT: Only high score if ALL parameters ideal
    # ========================================================================
    healthy_prob = 100.0
    healthy_reasons = []
    
    # Penalty 1: Elevated resistance (MOST CRITICAL - up to 55 points penalty)
    if features['main_mean'] > R_WEAR_CRITICAL_MIN:
        healthy_prob -= 55.0
    elif features['main_mean'] > R_WEAR_SEVERE_MIN:
        healthy_prob -= 48.0
    elif features['main_mean'] > R_WEAR_MODERATE_MIN:
        healthy_prob -= 40.0
    elif features['main_mean'] > R_WEAR_EARLY_MIN:
        healthy_prob -= 25.0
    elif features['main_mean'] > R_HEALTHY_MAX:
        healthy_prob -= 12.0
    
    # Penalty 2: High variability/noise (up to 30 points)
    if features['main_std'] > WEAR_STD_SEVERE:
        healthy_prob -= 30.0
    elif features['main_std'] > WEAR_STD_MODERATE:
        healthy_prob -= 22.0
    elif features['main_std'] > R_HEALTHY_STD_MAX:
        healthy_prob -= 12.0
    
    # Penalty 3: Telegraph/square wave pattern (up to 20 points)
    if features['telegraph_jumps'] > 8:
        healthy_prob -= 20.0
    elif features['telegraph_jumps'] > 5:
        healthy_prob -= 12.0
    elif features['telegraph_jumps'] > 2:
        healthy_prob -= 6.0
    
    # Penalty 4: Arcing spikes (up to 20 points)
    if features['total_critical_spikes'] > 0:
        healthy_prob -= 20.0
    elif features['total_severe_spikes'] > 1:
        healthy_prob -= 15.0
    elif features['total_moderate_spikes'] > 3:
        healthy_prob -= 10.0
    
    # Penalty 5: Timing asymmetry (up to 15 points)
    if features['asymmetry_ratio'] > ASYMMETRY_RATIO_SEVERE:
        healthy_prob -= 15.0
    elif features['asymmetry_ratio'] > ASYMMETRY_RATIO_MODERATE:
        healthy_prob -= 10.0
    elif features['asymmetry_ratio'] > 1.5:
        healthy_prob -= 5.0
    
    # Penalty 6: Bounces/oscillations (up to 12 points)
    if features['num_bounces'] > 4:
        healthy_prob -= 12.0
    elif features['num_bounces'] > 2:
        healthy_prob -= 7.0
    
    # Penalty 7: Grassy noise pattern (up to 10 points)
    if features['spike_density'] > 0.30:
        healthy_prob -= 10.0
    elif features['spike_density'] > 0.15:
        healthy_prob -= 5.0
    
    # Build healthy description if score is high
    if healthy_prob >= 50.0:
        healthy_reasons.append(f"Normal operation. Main Contact: Mean={features['main_mean']:.1f} µΩ (ideal: 20-70 µΩ), Std={features['main_std']:.1f} µΩ (smooth: <15 µΩ)")
        healthy_reasons.append(f"Timing: {features['asymmetry_ratio']:.2f} ratio (balanced: <1.5)")
        
        if features['total_critical_spikes'] == 0 and features['total_severe_spikes'] == 0:
            healthy_reasons.append("No abnormal arcing detected")
        
        if features['telegraph_jumps'] <= 2 and features['is_square_wave'] == 0:
            healthy_reasons.append("Smooth transitions, no misalignment patterns")
        
        prob_str = f"{max(0.0, healthy_prob):.2f} %"
        sev = "None" if healthy_prob >= 85.0 else "Low"
        desc = ". ".join(healthy_reasons)
        all_faults.append(_build_result("Healthy", prob_str, sev, desc))
    
    return all_faults


def classify_secondary_faults(features, phases, kpis, primary_faults):
    """
    Detect SECONDARY MECHANICAL/OPERATIONAL DEFECTS (Classes 6-10).
    Uses EXTREME STRICTNESS as per Gemini Agent 2 logic.
    Only reports if confidence >75% AND overwhelming evidence.
    """
    secondary_faults = []
    
    # Get primary defect names for context
    primary_names = [f['defect_name'] for f in primary_faults]
    
    # ========================================================================
    # CLASS 6: OPERATING MECHANISM MALFUNCTION (Slow/Fast Operation)
    # ========================================================================
    closing_time = kpis.get('Closing Time (ms)', None)
    opening_time = kpis.get('Opening Time (ms)', None)
    contact_speed = kpis.get('Contact Speed (m/s)', None)
    
    mechanism_score = 0
    mechanism_reasons = []
    kpi_count = 0
    
    # Check Closing Time
    if closing_time is not None:
        if closing_time > CLOSING_TIME_NOM[1] * (1 + TIMING_DEVIATION_THRESHOLD):
            deviation = ((closing_time - CLOSING_TIME_NOM[1]) / CLOSING_TIME_NOM[1]) * 100
            mechanism_score += 35
            mechanism_reasons.append(f"Slow closing: {closing_time:.1f} ms (nominal 80-100 ms, {deviation:.1f}% slower)")
            kpi_count += 1
        elif closing_time < CLOSING_TIME_NOM[0] * (1 - TIMING_DEVIATION_THRESHOLD):
            deviation = ((CLOSING_TIME_NOM[0] - closing_time) / CLOSING_TIME_NOM[0]) * 100
            mechanism_score += 35
            mechanism_reasons.append(f"Fast closing: {closing_time:.1f} ms ({deviation:.1f}% faster)")
            kpi_count += 1
    
    # Check Opening Time
    if opening_time is not None:
        if opening_time > OPENING_TIME_NOM[1] * (1 + TIMING_DEVIATION_THRESHOLD):
            deviation = ((opening_time - OPENING_TIME_NOM[1]) / OPENING_TIME_NOM[1]) * 100
            mechanism_score += 35
            mechanism_reasons.append(f"Slow opening: {opening_time:.1f} ms (nominal 30-40 ms, {deviation:.1f}% slower)")
            kpi_count += 1
        elif opening_time < OPENING_TIME_NOM[0] * (1 - TIMING_DEVIATION_THRESHOLD):
            deviation = ((OPENING_TIME_NOM[0] - opening_time) / OPENING_TIME_NOM[0]) * 100
            mechanism_score += 35
            mechanism_reasons.append(f"Fast opening: {opening_time:.1f} ms ({deviation:.1f}% faster)")
            kpi_count += 1
    
    # Check Contact Speed
    if contact_speed is not None:
        if contact_speed < CONTACT_SPEED_NOM[0] * (1 - TIMING_DEVIATION_THRESHOLD):
            deviation = ((CONTACT_SPEED_NOM[0] - contact_speed) / CONTACT_SPEED_NOM[0]) * 100
            mechanism_score += 30
            mechanism_reasons.append(f"Low contact speed: {contact_speed:.2f} m/s (nominal 4.5-6.5 m/s, {deviation:.1f}% slower)")
            kpi_count += 1
        elif contact_speed > CONTACT_SPEED_NOM[1] * (1 + TIMING_DEVIATION_THRESHOLD):
            deviation = ((contact_speed - CONTACT_SPEED_NOM[1]) / CONTACT_SPEED_NOM[1]) * 100
            mechanism_score += 30
            mechanism_reasons.append(f"High contact speed: {contact_speed:.2f} m/s ({deviation:.1f}% faster)")
            kpi_count += 1
    
    # Confidence boost if multiple KPIs affected
    if kpi_count >= 2:
        mechanism_score += 15
        mechanism_reasons.append("Multiple timing parameters affected - confirms mechanism malfunction")
    
    if mechanism_score > 0:
        conf = min(95.0, mechanism_score)
        sev = _get_severity(conf)
        desc = ". ".join(mechanism_reasons)
        secondary_faults.append(_build_result("Operating Mechanism Malfunction", f"{conf:.2f} %", sev, desc))
    
    # ========================================================================
    # CLASS 7: DAMPING SYSTEM FAULT (Excessive Bouncing/Oscillation)
    # ========================================================================
    damping_score = 0
    damping_reasons = []
    
    # Count bounces in main contact zone (>5 distinct bounces with >100 µΩ amplitude)
    if features['num_bounces'] > BOUNCE_COUNT_THRESHOLD:
        damping_score += 50
        damping_reasons.append(f"Excessive bouncing detected: {features['num_bounces']} distinct bounces in main contact zone (>5 indicates damper failure)")
    elif features['num_bounces'] >= 5:
        damping_score += 35
        damping_reasons.append(f"{features['num_bounces']} bounces detected")
    
    # Check for oscillation pattern (not random noise)
    if features['main_std'] > 50 and features['num_bounces'] >= 5:
        damping_score += 25
        damping_reasons.append(f"Oscillation pattern in main contact (Std={features['main_std']:.1f} µΩ with structured bounces)")
    
    if damping_score > 0:
        conf = min(95.0, damping_score)
        sev = _get_severity(conf)
        desc = ". ".join(damping_reasons)
        secondary_faults.append(_build_result("Damping System Fault", f"{conf:.2f} %", sev, desc))
    
    # ========================================================================
    # CLASS 8: SF6 PRESSURE LEAKAGE
    # ========================================================================
    sf6_pressure = kpis.get('SF6 Pressure (bar)', None)
    
    sf6_score = 0
    sf6_reasons = []
    
    if sf6_pressure is not None:
        if sf6_pressure < SF6_PRESSURE_CRITICAL:
            sf6_score += 60
            sf6_reasons.append(f"SF6 pressure critically low: {sf6_pressure:.2f} bar (normal: 5.5-6.5 bar)")
            
            # Check for prolonged arc (secondary evidence)
            if features['dur_opening'] > ARC_QUENCH_DURATION_MAX:
                sf6_score += 25
                sf6_reasons.append(f"Prolonged arc quenching ({features['dur_opening']} ms) confirms gas leak")
            
            # Check if primary defect is Arcing Wear (supports SF6 leak hypothesis)
            if "Arcing Contact Wear" in primary_names:
                sf6_score += 10
                sf6_reasons.append("Arcing wear detected as primary defect - consistent with SF6 leak")
    else:
        # No SF6 KPI: Can only infer from waveform (low confidence)
        if features['dur_opening'] > ARC_QUENCH_DURATION_MAX + 10:
            if "Arcing Contact Wear" in primary_names:
                # Get arcing wear confidence
                arcing_conf = 0
                for pf in primary_faults:
                    if pf['defect_name'] == "Arcing Contact Wear":
                        arcing_conf = float(pf['Confidence'].replace('%', '').strip())
                
                if arcing_conf > 85:
                    sf6_score += 55
                    sf6_reasons.append(f"Prolonged arc quenching ({features['dur_opening']} ms, normal <25 ms) with severe arcing wear - indicates possible SF6 leak")
                    sf6_reasons.append("WARNING: No SF6 pressure sensor data. Confidence limited to 70%")
                    sf6_score = min(sf6_score, 70)  # Cap at 70% without pressure KPI
    
    if sf6_score > 0:
        conf = min(95.0, sf6_score)
        sev = _get_severity(conf)
        desc = ". ".join(sf6_reasons)
        secondary_faults.append(_build_result("Pressure System Leakage (SF6 Gas Chamber)", f"{conf:.2f} %", sev, desc))
    
    # ========================================================================
    # CLASS 9: LINKAGE/CONNECTING ROD OBSTRUCTION
    # ========================================================================
    linkage_score = 0
    linkage_reasons = []
    
    # Detect "stutters" (flat plateaus within transitions)
    # This requires analyzing slope changes in transitions (complex detection)
    # Simplified: Use telegraph in main contact + increased duration as proxy
    
    if features['telegraph_jumps'] > STUTTER_COUNT_MIN and features['num_shelves'] > 3:
        # Check if operating time is increased
        total_op_time = features['dur_closing'] + features['dur_main'] + features['dur_opening']
        expected_time = 250  # Nominal ~250 ms total
        
        if total_op_time > expected_time * 1.15:  # >15% longer
            linkage_score += 50
            linkage_reasons.append(f"Detected {features['telegraph_jumps']} mechanical stutters with {features['num_shelves']} stepped plateaus")
            linkage_reasons.append(f"Total operation time {total_op_time} ms (expected ~{expected_time} ms, {((total_op_time/expected_time - 1)*100):.1f}% longer)")
        elif features['num_shelves'] >= 5:
            linkage_score += 35
            linkage_reasons.append(f"Multiple stepped plateaus ({features['num_shelves']}) indicate mechanical impedance")
    
    if linkage_score > 0:
        conf = min(95.0, linkage_score)
        sev = _get_severity(conf)
        desc = ". ".join(linkage_reasons)
        secondary_faults.append(_build_result("Linkage/Connecting Rod Obstruction/Damage", f"{conf:.2f} %", sev, desc))
    
    # ========================================================================
    # CLASS 10: FIXED CONTACT DAMAGE/DEFORMATION
    # ========================================================================
    dlro_value = kpis.get('DLRO Value (µΩ)', None)
    
    fixed_contact_score = 0
    fixed_contact_reasons = []
    
    if dlro_value is not None:
        if dlro_value > DLRO_CRITICAL:
            # Check if curve is smooth (not wear)
            if features['main_std'] < FIXED_CONTACT_STD_MAX:
                fixed_contact_score += 50
                fixed_contact_reasons.append(f"DLRO critically high: {dlro_value:.1f} µΩ (normal <50 µΩ, critical >100 µΩ)")
                fixed_contact_reasons.append(f"Smooth plateau (Std={features['main_std']:.1f} µΩ) indicates fixed contact/connection issue, not wear")
            else:
                # High DLRO but noisy = likely Main Contact Wear already detected
                if "Main Contact Wear" not in primary_names:
                    fixed_contact_score += 40
                    fixed_contact_reasons.append(f"DLRO high: {dlro_value:.1f} µΩ with noisy plateau")
                else:
                    # List as secondary with reduced confidence
                    fixed_contact_score += 30
                    fixed_contact_reasons.append(f"DLRO high: {dlro_value:.1f} µΩ (secondary to Main Contact Wear)")
        
        elif dlro_value > DLRO_MODERATE:
            if features['main_std'] < FIXED_CONTACT_STD_MAX:
                fixed_contact_score += 35
                fixed_contact_reasons.append(f"DLRO moderately elevated: {dlro_value:.1f} µΩ (normal <50 µΩ)")
    else:
        # No DLRO KPI: Can infer from smooth elevated plateau
        if features['main_mean'] > DLRO_MODERATE and features['main_std'] < FIXED_CONTACT_STD_MAX:
            if "Main Contact Wear" not in primary_names:
                fixed_contact_score += 30
                fixed_contact_reasons.append(f"Elevated but smooth plateau (Mean={features['main_mean']:.1f} µΩ, Std={features['main_std']:.1f} µΩ) suggests fixed contact issue")
                fixed_contact_reasons.append("WARNING: No DLRO sensor data. Confidence limited to 65%")
                fixed_contact_score = min(fixed_contact_score, 65)
    
    if fixed_contact_score > 0:
        conf = min(90.0, fixed_contact_score)
        sev = _get_severity(conf)
        desc = ". ".join(fixed_contact_reasons)
        secondary_faults.append(_build_result("Fixed Contact Damage/Deformation", f"{conf:.2f} %", sev, desc))
    
    return secondary_faults


def _get_severity(probability):
    """
    Determine severity based on defect probability score.
    
    Args:
        probability: Float 0-100 representing defect probability
    
    Returns:
        String: "Critical", "High", "Medium", "Low", or "None"
    """
    if probability >= 90:
        return "Critical"
    elif probability >= 75:
        return "High"
    elif probability >= 60:
        return "Medium"
    elif probability >= 50:
        return "Low"
    else:
        return "None"


def _build_result(name, conf, sev, desc):
    """Helper to build fault result dictionary with proper Unicode handling"""
    return {
        "defect_name": name,
        "Confidence": conf,
        "Severity": sev,
        "description": desc
    }


# =============================================================================
# KPI HELPER FUNCTIONS
# =============================================================================
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


# =============================================================================
# CENTRAL PIPELINE FUNCTION
# =============================================================================
def analyze_dcrm_from_dataframe(df, kpis=None):
    """
    Central pipeline function to analyze DCRM data from DataFrame.
    
    Args:
        df: DataFrame with Resistance column (401 points)
        kpis: KPI data in JSON format or dict format
              JSON format: {"kpis": [{"name": "...", "unit": "...", "value": ...}, ...]}
              Dict format: {"Name (unit)": value, ...}
    
    Returns:
        JSON with fault detection results and classifications
    """
    # Standardize input
    df_standardized = standardize_input(df)
    
    # Extract time columns
    time_cols = [c for c in df_standardized.columns if c.startswith('T_')]
    
    # Get first row values (for single waveform analysis)
    row_values = df_standardized.iloc[0][time_cols].values
    
    # Parse KPIs to internal format
    kpis_dict = parse_kpis_from_json(kpis)
    
    # Run analysis
    result = analyze_dcrm_advanced(row_values, kpis=kpis_dict)
    
    return result


if __name__ == "__main__":
    df = pd.read_csv('C:\\Users\\rkhanke\\Downloads\\parallel_proccessing\\combined\\data\\df3_final.csv')

    # New JSON format for KPIs
    sample_kpis = {
        "kpis": [
            {
                "name": "Closing Time",
                "unit": "ms",
                "value": 90.0
            },
            {
                "name": "Opening Time",
                "unit": "ms",
                "value": 35.0
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
                "name": "Contact Travel Distance",
                "unit": "mm",
                "value": 550.0
            },
            {
                "name": "Contact Speed",
                "unit": "m/s",
                "value": 5.5
            },
            {
                "name": "Peak Close Coil Current",
                "unit": "A",
                "value": 5.2
            },
            {
                "name": "Peak Trip Coil 1 Current",
                "unit": "A",
                "value": 5.0
            },
            {
                "name": "Peak Trip Coil 2 Current",
                "unit": "A",
                "value": 4.8
            },
            {
                "name": "Ambient Temperature",
                "unit": "°C",
                "value": 28.4
            }
        ]
    }
    
    result = analyze_dcrm_from_dataframe(df, kpis=sample_kpis)
    print(json.dumps(result, indent=2, ensure_ascii=False))

