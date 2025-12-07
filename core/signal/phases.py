# Previous Name: analysis/phase_analysis.py
import pandas as pd
import numpy as np
import json
from langchain_core.messages import HumanMessage

def get_coil_state(coil_series, active_threshold=0.1):
    """
    Determines the state of a coil (Close, Trip 1, Trip 2) during a phase.

    Args:
        coil_series (pd.Series): Series of coil values for the phase.
        active_threshold (float): Value above which a coil is considered active.

    Returns:
        str: 'Active' or 'Inactive'.
    """
    if coil_series.empty:
        return "Unknown"
    # A simple mean-based check is often sufficient for coil states in DCRM phases.
    # If the average value is above the threshold, consider it active during that phase.
    if coil_series.mean() > active_threshold:
        return "Active"
    return "Inactive"


def generate_llm_description_prompt(phase_data_summary, phase_info):
    """
    Generates an optimized prompt for Gemini to describe a DCRM phase
    and its diagnostic verdict based on numerical summary.

    Args:
        phase_data_summary (dict): A dictionary containing summarized numerical data.
        phase_info (dict): A dictionary containing static phase information.

    Returns:
        str: The optimized prompt string.
    """
    # DCRM Phase-Wise Diagnostic Reference
    dcrm_diagnostic_reference = """
    **DCRM Phase-Wise Diagnostic Criteria Reference:**
    
    Phase 1 - Pre-Contact Travel:
    Healthy: High/stable resistance (around open circuit baseline), zero/low current (around open circuit baseline), smooth increasing travel, clean close coil energization.
    Unhealthy: Premature drop in resistance or current rise (leakage/insulation breakdown), erratic/non-linear travel (mechanical binding), hesitations/plateaus (obstructions), slow/incomplete close coil activation.
    
    Phase 2 - Arcing Contact Engagement:
    Healthy: Rapid sharp resistance drop from high to low, rapid current rise from low to high, smooth travel with minimal bounce (<5% amplitude, <10ms duration).
    Unhealthy: Slow/incomplete resistance drop (contamination/erosion/misalignment), excessive/prolonged bounce (worn contacts/weak springs), current fails to reach expected level or rises slowly, abnormal current oscillations.
    
    Phase 3 - Main Contact Conduction:
    Healthy: Very low stable resistance (around closed circuit baseline, low StdDev), high constant current at test level (around closed circuit baseline, low StdDev), stable travel at maximum (fully closed, low StdDev).
    Unhealthy: Elevated resistance (> 50-100 μΩ above baseline - severe erosion/contamination), fluctuating resistance (poor pressure/loose connections), gradual resistance increase (progressive degradation), unstable travel (insufficient overtravel).
    
    Phase 4 - Contact Parting & Arc Elongation:
    Healthy: Smooth, near-linear resistance rise from low to high with moderate deviation (not excessive spikes), possible short intermediate plateau during main contact parting before arc elongation, steady current drop from high to low, smooth consistent travel decrease at expected opening speed, clean trip coil energization. Resistance should show controlled progressive increase without large erratic spikes.
    Unhealthy: Excessively spiked resistance with large deviations (severe arcing/welding), abrupt or erratic resistance changes, current drop too slow or fails to decrease steadily, erratic/non-linear travel (binding/friction), hesitations in travel, slow/inconsistent opening speed, prolonged arcing indicated by chaotic resistance pattern.
    
    Phase 5 - Final Open State:
    Healthy: High/stable resistance (around open circuit baseline), zero/low current (around open circuit baseline), stable travel at minimum (fully open position, no rebound/drift).
    Unhealthy: Resistance not reaching open circuit baseline or current not reaching open circuit baseline (leakage/contaminated insulation), travel rebound/drift (damping issues), travel not reaching full open position (obstructions).
    """
    
    # Reference key_characteristics for each phase
    reference_key_characteristics = {
        1: [
            "Contacts are physically separated",
            "No electrical conduction",
            "Travel distance increasing as contacts approach",
            "High resistance due to air gap"
        ],
        2: [
            "Sharp drop in resistance",
            "Current injection initiation",
            "Initial contact touch point",
            "Potential for contact bounce"
        ],
        3: [
            "Minimum contact resistance",
            "Maximum current flow",
            "No mechanical movement",
            "Trip coil energization initiated (towards end)"
        ],
        4: [
            "Physical contact separation",
            "Arc elongation",
            "Current interruption",
            "Rapid decrease in travel"
        ],
        5: [
            "Circuit fully isolated",
            "Contacts at rest position",
            "Arc fully extinguished",
            "Trip coil de-energization"
        ]
    }
    
    prompt = f"""
    You are an expert in Dynamic Contact Resistance Measurement (DCRM) analysis for circuit breakers.
    Your task is to generate FOUR specific outputs for a DCRM phase based on numerical data analysis:

    1. "key_characteristics" - Output 3-4 short, direct phrases describing the main physical/electrical events of this phase.
    2. "event_synopsis" - A concise 1-2 sentence summary of what actually occurred in this phase.
    3. "diagnostic_verdict_details" - A detailed 2-4 sentence analysis explaining the health verdict with specific numerical evidence from the "Numerical Data Summary" and justifying it against the "DCRM Phase-Wise Diagnostic Criteria Reference". Be explicit with numbers and thresholds.
    4. "confidence_score" - An integer from 0 to 100 indicating your confidence in the diagnostic verdict.

    **Phase Context:**
    - Phase Name: {phase_info['name']}
    - Operation Title: {phase_info['phaseTitle']}
    - Subheading: {phase_info['description']}
    - Time Interval: {phase_data_summary['time_interval']}
    - Phase ID: {phase_info['id']}

    **Numerical Data Summary for this Phase:**
    - Resistance: Min={phase_data_summary['resistance_min']:.2f} μΩ, Max={phase_data_summary['resistance_max']:.2f} μΩ, Avg={phase_data_summary['resistance_avg']:.2f} μΩ, StdDev={phase_data_summary['resistance_std']:.2f} μΩ
    - Current: Min={phase_data_summary['current_min']:.2f} A, Max={phase_data_summary['current_max']:.2f} A, Avg={phase_data_summary['current_avg']:.2f} A, StdDev={phase_data_summary['current_std']:.2f} A
    - Travel: Min={phase_data_summary['travel_min']:.2f} mm, Max={phase_data_summary['travel_max']:.2f} mm, Avg={phase_data_summary['travel_avg']:.2f} mm, StdDev={phase_data_summary['travel_std']:.2f} mm
    - Close Coil: State = '{phase_data_summary['close_coil_state']}' (Avg: {phase_data_summary['close_coil_avg']:.2f} A)
    - Trip Coil 1: State = '{phase_data_summary['trip_coil_1_state']}' (Avg: {phase_data_summary['trip_coil_1_avg']:.2f} A)
    - Trip Coil 2: State = '{phase_data_summary['trip_coil_2_state']}' (Avg: {phase_data_summary['trip_coil_2_avg']:.2f} A)

    **Programmatic Health Status:** {phase_data_summary['programmatic_health_status']}

    {dcrm_diagnostic_reference}

    **Reference key_characteristics for this phase:**
    {reference_key_characteristics.get(phase_info['phaseNumber'], [])}

    **Instructions:**
    - Analyze the numerical data against the diagnostic criteria reference above for the specific phase.
    - Focus the 'diagnostic_verdict_details' on explaining *why* the 'Programmatic Health Status' was assigned, using the specific measured values provided.
    - For 'key_characteristics', output 3-4 short, direct phrases describing the main physical/electrical events of this phase.
    - Provide specific numerical evidence in diagnostic_verdict_details.
    - For 'confidence_score', assess how well the measured data aligns with the expected patterns for the given health status. Higher alignment = higher confidence.
    - Output ONLY a valid JSON object with these exact keys:
      {{"key_characteristics": ["characteristic1", "characteristic2"], "event_synopsis": "synopsis text", "diagnostic_verdict_details": "details text", "confidence_score": 85}}
    """
    return prompt

# --- Core DCRM Analysis Function ---
def analyze_dcrm_data(df, llm=None):
    """
    Analyzes DCRM DataFrame to segment phases, determine health, and format JSON output.

    Args:
        df (pd.DataFrame): Input DataFrame with DCRM data.
                           Expected columns: Time_ms, Resistance, Current, Travel,
                           Close_Coil, Trip_Coil_1, Trip_Coil_2.
        llm (LangChain LLM object, optional): LLM to use for generating descriptions.

    Returns:
        dict: A dictionary representing the DCRM phase-wise interpretation in JSON format.
    """

    # Ensure Time_ms is the index for easier slicing
    if 'Time_ms' in df.columns:
        df = df.set_index('Time_ms')
    else:
        # If no Time_ms, assume index is time or create it
        if df.index.name != 'Time_ms':
             # Create Time_ms if not present, assuming 1ms steps
             df['Time_ms'] = range(len(df))
             df = df.set_index('Time_ms')

    # Convert columns to numeric, coercing errors to NaN
    numeric_cols = ['Resistance', 'Current', 'Travel', 'Close_Coil', 'Trip_Coil_1', 'Trip_Coil_2']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            # Handle missing columns gracefully
            df[col] = 0.0
            
    df = df.dropna() # Drop rows with any NaN created by coercion

    if df.empty:
        return {"error": "Input DataFrame is empty after cleaning."}

    # --- Optimized Dynamic Phase Boundary Detection Thresholds ---
    # Based on EHV DCRM domain knowledge and international standards (IEC, IEEE)
    
    # OPEN CIRCUIT THRESHOLDS (Phase 1 & 5)
    OPEN_CIRCUIT_RESISTANCE_IDEAL = 100000     # Ideal infinite resistance (≥ 10⁵ μΩ) for perfect open circuit
    OPEN_CIRCUIT_RESISTANCE_MIN = 750          # Minimum acceptable high resistance (straight line threshold for EHV)
    OPEN_CIRCUIT_VARIATION_THRESHOLD = 30      # ±30 μΩ flat variation is healthy for open circuit (straight line)
    
    # CLOSED CIRCUIT THRESHOLDS (Phase 3 - Main Contact Plateau)
    CLOSED_CIRCUIT_RESISTANCE_BASELINE = 200   # Typical ~200 μΩ for healthy main contact plateau in EHV
    CLOSED_CIRCUIT_RESISTANCE_MAX = 350        # Upper limit for acceptable main contact (allowing tolerance for EHV)
    CLOSED_CIRCUIT_VARIATION_THRESHOLD = 50    # ±50 μΩ deviation for healthy stable plateau
    
    # ARCING THRESHOLDS (Phase 2 & 4)
    ARCING_RESISTANCE_MIN = 300                # Arcing typically starts around 300-500 μΩ
    ARCING_RESISTANCE_MAX = 5000               # Arcing spikes can reach 3500-5000 μΩ in healthy operation
    ARCING_STDDEV_HEALTHY_MIN = 80             # Minimum StdDev for healthy arcing (indicates arcing activity)
    ARCING_SPIKE_FACTOR = 1.8                  # Max should be at least 1.8x Min for healthy arcing spikes
    
    # CURRENT & TRAVEL BASELINES (from observed data)
    OPEN_CIRCUIT_CURRENT_BASELINE = 230        # Typical open circuit current (leakage/capacitive)
    CLOSED_CIRCUIT_CURRENT_BASELINE = 715      # Test current injection level during conduction
    TRAVEL_MIN_POSITION = 200                  # Fully open position (minimum travel)
    TRAVEL_MAX_POSITION = 750                  # Fully closed position (maximum travel/overtravel)

    # PHASE DETECTION THRESHOLDS (for boundary identification)
    COIL_ACTIVE_THRESHOLD = 0.5                # Coil current above this indicates activation (A)
    GRADIENT_R_CHANGE = 50                     # Minimum resistance gradient for phase transition detection (μΩ/ms)
    GRADIENT_I_CHANGE = 50                     # Minimum current gradient for phase transition detection (A/ms)
    TRAVEL_MOVEMENT_RATE_THRESHOLD = 1.0       # Minimum travel rate to detect active movement (mm/ms)

    time_min = df.index.min()
    time_max = df.index.max()
    time_step = np.mean(np.diff(df.index)) if len(df.index) > 1 else 1.0 # Average time step, default to 1ms if only one point

    # Initialize phase start and end times
    phase_start_times = {1: time_min}
    phase_end_times = {5: time_max} # Phase 5 always extends to the end of the data

    # Helper to calculate rolling mean of differences (gradient)
    def rolling_gradient(series, window=5):
        if len(series) < window: # Handle cases where series is shorter than window
            return series.diff()
        return series.diff().rolling(window=window, min_periods=1).mean()

    # --- Detection Logic (Refined) ---

    # 1. Detect Contact Make (End of Phase 1 / Start of Phase 2)
    # Look for a sharp drop in Resistance OR sharp rise in Current, AFTER Close_Coil activation.
    close_coil_active_idx = df[df['Close_Coil'] > COIL_ACTIVE_THRESHOLD].index
    close_op_start = close_coil_active_idx.min() if not close_coil_active_idx.empty else time_min # Start from beginning if no clear close coil activation

    resistance_gradient_neg = rolling_gradient(df['Resistance'], window=3) # Smaller window for sharper detection
    current_gradient_pos = rolling_gradient(df['Current'], window=3)

    contact_make_candidates = df.loc[df.index >= close_op_start].index[
        (resistance_gradient_neg.loc[df.index >= close_op_start] < -GRADIENT_R_CHANGE) | # Significant drop in R
        (current_gradient_pos.loc[df.index >= close_op_start] > GRADIENT_I_CHANGE)     # Significant rise in I
    ]

    contact_make_time = time_max # Default to end
    if not contact_make_candidates.empty:
        contact_make_time = contact_make_candidates.min()
    else: # Fallback: first time current is significantly active after close op start AND resistance has dropped
        current_active_after_close_op = df.loc[df.index >= close_op_start].index[
            (df['Current'].loc[df.index >= close_op_start] > OPEN_CIRCUIT_CURRENT_BASELINE + 50) & # Current rising significantly
            (df['Resistance'].loc[df.index >= close_op_start] < OPEN_CIRCUIT_RESISTANCE_MIN) # Resistance has dropped significantly from open circuit
        ]
        if not current_active_after_close_op.empty:
            contact_make_time = current_active_after_close_op.min()
        else:
            contact_make_time = df.index.min() + (df.index.max() - df.index.min()) * 0.20 # Fallback to 20% into data if no clear event

    phase_end_times[1] = contact_make_time - time_step
    phase_start_times[2] = contact_make_time

    # 2. Detect Conduction Stabilization (End of Phase 2 / Start of Phase 3)
    # Look for Resistance near CLOSED_CIRCUIT_RESISTANCE_BASELINE AND Current near CLOSED_CIRCUIT_CURRENT_BASELINE AND both are stable.
    stable_conduction_candidates = df.loc[df.index >= phase_start_times[2]].index[
        (df['Resistance'].loc[df.index >= phase_start_times[2]].between(CLOSED_CIRCUIT_RESISTANCE_BASELINE - 100, CLOSED_CIRCUIT_RESISTANCE_MAX + 50)) &
        (df['Current'].loc[df.index >= phase_start_times[2]].between(CLOSED_CIRCUIT_CURRENT_BASELINE - 100, CLOSED_CIRCUIT_CURRENT_BASELINE + 50)) &
        (df['Resistance'].loc[df.index >= phase_start_times[2]].rolling(window=10, min_periods=1).std() < 30) & # Low std dev for stability (increased tolerance)
        (df['Current'].loc[df.index >= phase_start_times[2]].rolling(window=10, min_periods=1).std() < 30)
    ]

    conduction_start_time = time_max
    if not stable_conduction_candidates.empty:
        conduction_start_time = stable_conduction_candidates.min()
    else: # Fallback: assume conduction starts when resistance first drops very low and current is high
        low_res_high_curr_first_hit = df.loc[df.index >= phase_start_times[2]].index[
            (df['Resistance'].loc[df.index >= phase_start_times[2]] < CLOSED_CIRCUIT_RESISTANCE_MAX) & # Clearly dropped to conduction level
            (df['Current'].loc[df.index >= phase_start_times[2]] > CLOSED_CIRCUIT_CURRENT_BASELINE - 100) # Clearly risen to conduction level
        ]
        if not low_res_high_curr_first_hit.empty:
            conduction_start_time = low_res_high_curr_first_hit.min()
        else:
            conduction_start_time = phase_start_times[2] + 10.0 # Ensure a minimum 10ms duration for phase 2 if no clear event

    phase_end_times[2] = conduction_start_time - time_step
    phase_start_times[3] = conduction_start_time

    # 3. Detect Contact Break (End of Phase 3 / Start of Phase 4)
    # Look for Trip_Coil_1 activation OR a sharp rise in Resistance OR sharp drop in Current.
    trip_coil_active_idx = df[df['Trip_Coil_1'] > COIL_ACTIVE_THRESHOLD].index
    trip_op_start = trip_coil_active_idx.min() if not trip_coil_active_idx.empty else phase_start_times[3] + (df.index.max() - phase_start_times[3]) * 0.5 # Midpoint if no trip coil

    resistance_gradient_pos = rolling_gradient(df['Resistance'], window=3)
    current_gradient_neg = rolling_gradient(df['Current'], window=3)

    contact_break_candidates = df.loc[df.index >= trip_op_start].index[
        (resistance_gradient_pos.loc[df.index >= trip_op_start] > GRADIENT_R_CHANGE) | # Significant rise in R
        (current_gradient_neg.loc[df.index >= trip_op_start] < -GRADIENT_I_CHANGE)    # Significant drop in I
    ]

    contact_break_time = time_max
    if not contact_break_candidates.empty:
        contact_break_time = contact_break_candidates.min()
    else: # Fallback: first time current is low again and resistance is high after trip op start
        current_low_high_res_after_trip_op = df.loc[df.index >= trip_op_start].index[
            (df['Current'].loc[df.index >= trip_op_start] < CLOSED_CIRCUIT_CURRENT_BASELINE - 100) & # Current dropping significantly
            (df['Resistance'].loc[df.index >= trip_op_start] > CLOSED_CIRCUIT_RESISTANCE_BASELINE + 100) # Resistance rising significantly
        ]
        if not current_low_high_res_after_trip_op.empty:
            contact_break_time = current_low_high_res_after_trip_op.min()
        else:
            contact_break_time = phase_start_times[3] + (df.index.max() - phase_start_times[3]) * 0.75 # Default to later if no clear break

    phase_end_times[3] = contact_break_time - time_step
    phase_start_times[4] = contact_break_time

    # 4. Detect Final Open State Stabilization (End of Phase 4 / Start of Phase 5)
    # Look for Resistance high AND Current near OPEN_CIRCUIT_CURRENT_BASELINE AND Travel is stable at min position.
    final_open_stable_candidates = df.loc[df.index >= phase_start_times[4]].index[
        (df['Resistance'].loc[df.index >= phase_start_times[4]] > OPEN_CIRCUIT_RESISTANCE_MIN - 50) & # High resistance
        (df['Current'].loc[df.index >= phase_start_times[4]].between(OPEN_CIRCUIT_CURRENT_BASELINE - 50, OPEN_CIRCUIT_CURRENT_BASELINE + 50)) &
        (df['Travel'].loc[df.index >= phase_start_times[4]].rolling(window=10, min_periods=1).std() < 2) # Travel stability
    ]

    final_open_time = time_max
    if not final_open_stable_candidates.empty:
        final_open_time = final_open_stable_candidates.min()
    else: # Fallback: when travel stops moving after contact break
        travel_still_moving = df.loc[df.index >= phase_start_times[4]].index[rolling_gradient(df['Travel']).loc[df.index >= phase_start_times[4]].abs() > TRAVEL_MOVEMENT_RATE_THRESHOLD]
        if not travel_still_moving.empty:
            final_open_time = travel_still_moving.max() + 5 * time_step # End of significant travel movement
        else:
            final_open_time = df.index.max() - 20.0 # Default to near end if no clear stabilization

    phase_end_times[4] = final_open_time - time_step
    phase_start_times[5] = final_open_time

    # --- Ensure Phase Times are Valid and Sequential ---
    current_end = df.index.min() - time_step
    for p_id in sorted(phase_start_times.keys()):
        start_time_candidate = phase_start_times[p_id]
        end_time_candidate = phase_end_times[p_id]

        start_idx_loc = df.index.get_indexer([start_time_candidate], method='nearest')[0]
        end_idx_loc = df.index.get_indexer([end_time_candidate], method='nearest')[0]

        phase_start_times[p_id] = df.index[start_idx_loc]
        phase_end_times[p_id] = df.index[end_idx_loc]

        if phase_start_times[p_id] < current_end:
            phase_start_times[p_id] = current_end
            if phase_start_times[p_id] >= phase_end_times[p_id]:
                # Ensure minimum 5ms duration for a phase, or until max_time if near end
                phase_end_times[p_id] = min(phase_start_times[p_id] + time_step * 5, time_max) 
                # If still problematic, try to give it at least 1 point
                if phase_start_times[p_id] == phase_end_times[p_id] and phase_start_times[p_id] < time_max:
                    next_idx = df.index.get_indexer([phase_start_times[p_id] + time_step], method='nearest')[0]
                    if next_idx < len(df.index):
                        phase_end_times[p_id] = df.index[next_idx]

        current_end = phase_end_times[p_id]

    if 5 in phase_end_times:
         phase_end_times[5] = time_max # Ensure the last phase always goes to the end


    # --- JSON Structure Template (static definitions) ---
    json_output_template = {
      "phaseWiseAnalysis": [
          {
            "phaseNumber": 1,
            "id": "pre-contact-travel",
            "name": "Pre-Contact Travel",
            "phaseTitle": "Closing Operation — Pre-Contact Travel",
            "description": "Initial state before arc contact occurs",
            "color": "#ff9800"
          },
          {
            "phaseNumber": 2,
            "id": "arcing-contact-engagement-arc-initiation",
            "name": "Arcing Contact Engagement & Arc Initiation",
            "phaseTitle": "Closing Operation — Arcing Contact Engagement & Arc Initiation",
            "description": "Arcing contact engagement and arc initiation",
            "color": "#ff26bd"
          },
          {
            "phaseNumber": 3,
            "id": "main-contact-conduction",
            "name": "Main Contact Conduction",
            "phaseTitle": "Fully Closed State — Main Contact Conduction",
            "description": "Main contact conduction",
            "color": "#4caf50"
          },
          {
            "phaseNumber": 4,
            "id": "main-contact-parting-arc-elongation",
            "name": "Main Contact Parting & Arc Elongation",
            "phaseTitle": "Opening Operation — Main Contact Parting & Arc Elongation",
            "description": "Main contact parting and arc elongation",
            "color": "#2196f3"
          },
          {
            "phaseNumber": 5,
            "id": "final-open-state",
            "name": "Final Open State",
            "phaseTitle": "Post-Interruption — Final Open State",
            "description": "Final open state",
            "color": "#a629ff"
          }
        ]
    }

    final_segmented_phases = []

    # Populate the JSON structure dynamically
    for phase_data_template in json_output_template['phaseWiseAnalysis']:
        phase_id = phase_data_template['phaseNumber']
        start_time = phase_start_times.get(phase_id)
        end_time = phase_end_times.get(phase_id)

        if start_time is None or end_time is None or start_time >= end_time:
            continue

        phase_data = df.loc[(df.index >= start_time) & (df.index <= end_time)]

        if phase_data.empty:
            continue

        current_phase_entry = phase_data_template.copy()
        current_phase_entry['startTime'] = int(start_time)
        current_phase_entry['endTime'] = int(end_time)
        current_phase_entry['details'] = {
            "resistance": "",
            "current": "",
            "travel": "",
            "characteristics": []
        }
        current_phase_entry['waveformAnalysis'] = {
            "resistance": "",
            "current": "",
            "travel": "",
            "coilAnalysis": {
                "closeCoil": {"status": "", "description": "", "measuredValues": ""},
                "tripCoil1": {"status": "", "description": "", "measuredValues": ""},
                "tripCoil2": {"status": "", "description": "", "measuredValues": ""}
            }
        }
        current_phase_entry['diagnosticVerdict'] = ""
        current_phase_entry['eventSynopsis'] = ""
        current_phase_entry['confidence'] = 0
        current_phase_entry['status'] = "Unknown"

        current_data_summary = {
            'time_interval': f"{start_time:.1f}ms - {end_time:.1f}ms",
            'resistance_min': phase_data['Resistance'].min(),
            'resistance_max': phase_data['Resistance'].max(),
            'resistance_avg': phase_data['Resistance'].mean(),
            'resistance_std': phase_data['Resistance'].std() if len(phase_data) > 1 else 0,
            'current_min': phase_data['Current'].min(),
            'current_max': phase_data['Current'].max(),
            'current_avg': phase_data['Current'].mean(),
            'current_std': phase_data['Current'].std() if len(phase_data) > 1 else 0,
            'travel_min': phase_data['Travel'].min(),
            'travel_max': phase_data['Travel'].max(),
            'travel_avg': phase_data['Travel'].mean(),
            'travel_std': phase_data['Travel'].std() if len(phase_data) > 1 else 0,
            'close_coil_avg': phase_data['Close_Coil'].mean(),
            'trip_coil_1_avg': phase_data['Trip_Coil_1'].mean(),
            'trip_coil_2_avg': phase_data['Trip_Coil_2'].mean(),
            'close_coil_state': get_coil_state(phase_data['Close_Coil']),
            'trip_coil_1_state': get_coil_state(phase_data['Trip_Coil_1']),
            'trip_coil_2_state': get_coil_state(phase_data['Trip_Coil_2']),
            'programmatic_health_status': "Unknown",
            'resistance_status': "",
            'current_status': "",
            'travel_status': ""
        }

        # --- Health Verdict Logic (Optimized) ---
        # (Same logic as backup_DCRM2.py, ensuring it populates current_data_summary and current_phase_entry)
        
        if phase_id == 1: # PRE-CONTACT TRAVEL
            is_resistance_infinite = current_data_summary['resistance_avg'] >= OPEN_CIRCUIT_RESISTANCE_IDEAL
            is_resistance_high_and_flat = (current_data_summary['resistance_avg'] >= OPEN_CIRCUIT_RESISTANCE_MIN and 
                                           current_data_summary['resistance_std'] <= OPEN_CIRCUIT_VARIATION_THRESHOLD)
            
            if is_resistance_infinite or is_resistance_high_and_flat:
                current_data_summary['programmatic_health_status'] = "Healthy"
                current_data_summary['resistance_status'] = "High and Stable (Open Circuit)"
                if is_resistance_infinite:
                    resistance_desc = f"Resistance is infinite (Avg: {current_data_summary['resistance_avg']:.1f} μΩ ≥ {OPEN_CIRCUIT_RESISTANCE_IDEAL} μΩ) with minimal variation (StdDev: {current_data_summary['resistance_std']:.1f} μΩ), indicating proper open circuit with clean air gap and no premature contact."
                else:
                    resistance_desc = f"Resistance is high and stable (Avg: {current_data_summary['resistance_avg']:.1f} μΩ ≥ {OPEN_CIRCUIT_RESISTANCE_MIN} μΩ) with flat straight-line pattern (StdDev: {current_data_summary['resistance_std']:.1f} μΩ ≤ ±{OPEN_CIRCUIT_VARIATION_THRESHOLD} μΩ), indicating proper open circuit with clean separation and no premature contact."
                
                current_phase_entry['waveformAnalysis']['resistance'] = resistance_desc
                current_phase_entry['waveformAnalysis']['current'] = f"Current remains stable around its open circuit baseline of {OPEN_CIRCUIT_CURRENT_BASELINE:.1f} A (Avg: {current_data_summary['current_avg']:.1f}), reinforcing no premature electrical connection."
                current_phase_entry['waveformAnalysis']['travel'] = f"Travel shows a smooth, continuous increase from {current_data_summary['travel_min']:.1f} mm to {current_data_summary['travel_max']:.1f} mm, representing unimpeded mechanical movement."
                
                current_phase_entry['details']['resistance'] = f"High resistance (~{current_data_summary['resistance_avg']:.0f} µΩ) - contacts are open"
                current_phase_entry['details']['current'] = f"Low current (~{current_data_summary['current_avg']:.0f}) - no current flow"
                current_phase_entry['details']['travel'] = f"Gradual increase from {current_data_summary['travel_min']:.0f} to {current_data_summary['travel_max']:.0f} - contact approaching"
            else:
                current_data_summary['programmatic_health_status'] = "Unhealthy - Pre-Contact Issues"
                current_phase_entry['waveformAnalysis']['resistance'] = f"Resistance is too low (Avg: {current_data_summary['resistance_avg']:.1f} μΩ < {OPEN_CIRCUIT_RESISTANCE_MIN} μΩ) or shows excessive variation (StdDev: {current_data_summary['resistance_std']:.1f} μΩ > ±{OPEN_CIRCUIT_VARIATION_THRESHOLD} μΩ), suggesting premature contact, leakage, or insulation breakdown."
                current_phase_entry['waveformAnalysis']['current'] = f"Current was not consistently around {OPEN_CIRCUIT_CURRENT_BASELINE:.1f} A (Avg: {current_data_summary['current_avg']:.1f} A), indicating a premature electrical connection or abnormal current path."
                current_phase_entry['waveformAnalysis']['travel'] = f"Travel was not smooth, failed to ramp up sufficiently (Range: {current_data_summary['travel_min']:.1f} to {current_data_summary['travel_max']:.1f} mm), or showed erratic movement, indicating mechanical binding or obstruction."
                
                current_phase_entry['details']['resistance'] = f"Abnormal resistance (~{current_data_summary['resistance_avg']:.0f} µΩ) - premature contact"
                current_phase_entry['details']['current'] = f"Abnormal current (~{current_data_summary['current_avg']:.0f}) - unexpected flow"
                current_phase_entry['details']['travel'] = f"Erratic travel from {current_data_summary['travel_min']:.0f} to {current_data_summary['travel_max']:.0f} - binding detected"

        elif phase_id == 2: # Arcing Contact Engagement
            resistance_range = current_data_summary['resistance_max'] - current_data_summary['resistance_min']
            current_rise_magnitude = current_data_summary['current_max'] - current_data_summary['current_min']
            
            resistance_drops_significantly = (current_data_summary['resistance_max'] > 500 and 
                                             current_data_summary['resistance_min'] < CLOSED_CIRCUIT_RESISTANCE_MAX + 100 and
                                             resistance_range > 200)
            
            has_healthy_arcing_spikes = (current_data_summary['resistance_max'] > ARCING_RESISTANCE_MIN * 2 and
                                         current_data_summary['resistance_std'] > 80 and
                                         current_data_summary['resistance_max'] / max(current_data_summary['resistance_min'], 1) > 1.8)
            
            current_rises_properly = (current_rise_magnitude > 250 and
                                     current_data_summary['current_max'] > CLOSED_CIRCUIT_CURRENT_BASELINE * 0.6)
            
            travel_increasing = (current_data_summary['travel_max'] - current_data_summary['travel_min']) > 100
            
            if resistance_drops_significantly and has_healthy_arcing_spikes and current_rises_properly and travel_increasing:
                current_data_summary['programmatic_health_status'] = "Healthy"
                current_phase_entry['waveformAnalysis']['resistance'] = f"Resistance exhibits healthy arcing behavior: sharp drop from {current_data_summary['resistance_max']:.1f} μΩ to {current_data_summary['resistance_min']:.1f} μΩ (Range: {resistance_range:.1f} μΩ) with prominent high-frequency arcing spikes (StdDev: {current_data_summary['resistance_std']:.1f} μΩ > 80), indicating proper arc initiation as arcing contacts engage."
                current_phase_entry['waveformAnalysis']['current'] = f"Current rises sharply from {current_data_summary['current_min']:.1f} A to {current_data_summary['current_max']:.1f} A (Rise: {current_rise_magnitude:.1f} A), establishing conduction path."
                current_phase_entry['waveformAnalysis']['travel'] = f"Travel increases smoothly from {current_data_summary['travel_min']:.1f} mm to {current_data_summary['travel_max']:.1f} mm."
                
                current_phase_entry['details']['resistance'] = f"Highly variable ({current_data_summary['resistance_min']:.0f}-{current_data_summary['resistance_max']:.0f} µΩ) - arcing occurs"
                current_phase_entry['details']['current'] = f"Sharp rise to ~{current_data_summary['current_max']:.0f} - current flow begins"
                current_phase_entry['details']['travel'] = f"Continues to increase to ~{current_data_summary['travel_max']:.0f} - contacts closing"
            else:
                current_data_summary['programmatic_health_status'] = "Unhealthy - Arcing/Contact Issue"
                if not has_healthy_arcing_spikes:
                    current_phase_entry['waveformAnalysis']['resistance'] = f"CRITICAL ISSUE: Resistance pattern lacks expected high-frequency arcing spikes (Max: {current_data_summary['resistance_max']:.1f} μΩ, StdDev: {current_data_summary['resistance_std']:.1f} μΩ << 80 μΩ expected). This indicates SEVERELY WORN arcing contacts."
                elif not resistance_drops_significantly:
                    current_phase_entry['waveformAnalysis']['resistance'] = f"CRITICAL ISSUE: Resistance fails to drop adequately (Max: {current_data_summary['resistance_max']:.1f} μΩ, Min: {current_data_summary['resistance_min']:.1f} μΩ). This indicates CONTACT MISALIGNMENT or mechanical binding."
                else:
                    current_phase_entry['waveformAnalysis']['resistance'] = f"Resistance pattern shows abnormal arcing behavior (Range: {current_data_summary['resistance_min']:.1f} to {current_data_summary['resistance_max']:.1f} μΩ)."
                
                current_phase_entry['waveformAnalysis']['current'] = f"Current rise is inadequate or abnormal (Rise: {current_rise_magnitude:.1f} A)."
                current_phase_entry['waveformAnalysis']['travel'] = f"Travel pattern: {current_data_summary['travel_min']:.1f} to {current_data_summary['travel_max']:.1f} mm."
                
                current_phase_entry['details']['resistance'] = f"Abnormal ({current_data_summary['resistance_min']:.0f}-{current_data_summary['resistance_max']:.0f} µΩ) - contact issue"
                current_phase_entry['details']['current'] = f"Poor rise to ~{current_data_summary['current_max']:.0f} - impedance detected"
                current_phase_entry['details']['travel'] = f"Range to ~{current_data_summary['travel_max']:.0f} - issue detected"

        elif phase_id == 3: # Main Contact Conduction
            is_resistance_low = current_data_summary['resistance_avg'] <= CLOSED_CIRCUIT_RESISTANCE_MAX
            is_resistance_flat_plateau = current_data_summary['resistance_std'] <= CLOSED_CIRCUIT_VARIATION_THRESHOLD
            has_abnormal_spikes = current_data_summary['resistance_max'] > CLOSED_CIRCUIT_RESISTANCE_MAX + 100
            is_current_stable = abs(current_data_summary['current_avg'] - CLOSED_CIRCUIT_CURRENT_BASELINE) < 50 and current_data_summary['current_std'] < 30
            is_travel_stable = current_data_summary['travel_std'] < 5 and abs(current_data_summary['travel_avg'] - TRAVEL_MAX_POSITION) < 20

            if is_resistance_low and is_resistance_flat_plateau and not has_abnormal_spikes and is_current_stable and is_travel_stable:
                current_data_summary['programmatic_health_status'] = "Healthy"
                current_phase_entry['waveformAnalysis']['resistance'] = f"Resistance maintains a healthy flat plateau at low values (Avg: {current_data_summary['resistance_avg']:.1f} μΩ) with minimal deviation."
                current_phase_entry['waveformAnalysis']['current'] = f"Current remains high and stable, around {CLOSED_CIRCUIT_CURRENT_BASELINE:.1f} A (Avg: {current_data_summary['current_avg']:.1f})."
                current_phase_entry['waveformAnalysis']['travel'] = f"Travel remains perfectly stable at its maximum stroke of approximately {TRAVEL_MAX_POSITION:.1f} mm."
                
                current_phase_entry['details']['resistance'] = f"Stable low resistance (~{current_data_summary['resistance_avg']:.0f} µΩ) - good contact"
                current_phase_entry['details']['current'] = f"Gradual decline from {current_data_summary['current_max']:.0f} to {current_data_summary['current_min']:.0f} - stable flow"
                current_phase_entry['details']['travel'] = f"Constant at ~{current_data_summary['travel_avg']:.0f} - contacts fully closed"
            else:
                current_data_summary['programmatic_health_status'] = "Unhealthy - Contact Degradation"
                current_phase_entry['waveformAnalysis']['resistance'] = f"Resistance is unhealthy: either elevated (Avg: {current_data_summary['resistance_avg']:.1f} μΩ), shows high abnormal spikes, or lacks flat plateau."
                current_phase_entry['waveformAnalysis']['current'] = f"Current was unstable or lower than expected (Avg: {current_data_summary['current_avg']:.1f} A)."
                current_phase_entry['waveformAnalysis']['travel'] = f"Travel was not stable at its maximum (Avg: {current_data_summary['travel_avg']:.1f} mm)."
                
                current_phase_entry['details']['resistance'] = f"Elevated resistance (~{current_data_summary['resistance_avg']:.0f} µΩ) - degradation"
                current_phase_entry['details']['current'] = f"Unstable current (~{current_data_summary['current_avg']:.0f}) - poor conduction"
                current_phase_entry['details']['travel'] = f"Unstable at ~{current_data_summary['travel_avg']:.0f} - insufficient overtravel"

        elif phase_id == 4: # Main Contact Parting
            resistance_rise_magnitude = current_data_summary['resistance_max'] - current_data_summary['resistance_min']
            current_drop_magnitude = current_data_summary['current_max'] - current_data_summary['current_min']
            travel_drop_magnitude = current_data_summary['travel_max'] - current_data_summary['travel_min']
            
            resistance_rises_smoothly = (current_data_summary['resistance_min'] < CLOSED_CIRCUIT_RESISTANCE_MAX + 100 and
                                        resistance_rise_magnitude > 300)
            current_drops_steadily = current_drop_magnitude > 200
            travel_decreases_properly = travel_drop_magnitude > 400
            has_smooth_progressive_rise = current_data_summary['resistance_std'] < 250
            
            if resistance_rises_smoothly and current_drops_steadily and travel_decreases_properly and has_smooth_progressive_rise:
                current_data_summary['programmatic_health_status'] = "Healthy"
                current_phase_entry['waveformAnalysis']['resistance'] = f"Resistance rises smoothly and progressively from {current_data_summary['resistance_min']:.1f} μΩ to {current_data_summary['resistance_max']:.1f} μΩ."
                current_phase_entry['waveformAnalysis']['current'] = f"Current drops steadily from {current_data_summary['current_max']:.1f} A to {current_data_summary['current_min']:.1f} A."
                current_phase_entry['waveformAnalysis']['travel'] = f"Travel decreases smoothly from {current_data_summary['travel_max']:.1f} mm to {current_data_summary['travel_min']:.1f} mm."
                
                current_phase_entry['details']['resistance'] = f"Rising from {current_data_summary['resistance_min']:.0f} to {current_data_summary['resistance_max']:.0f} µΩ - contacts separating"
                current_phase_entry['details']['current'] = f"Sharp drop from {current_data_summary['current_max']:.0f} to {current_data_summary['current_min']:.0f} - current flow stops"
                current_phase_entry['details']['travel'] = f"Sharp drop from {current_data_summary['travel_max']:.0f} to {current_data_summary['travel_min']:.0f} - contacts opening"
            else:
                current_data_summary['programmatic_health_status'] = "Unhealthy - Opening/Interruption Issue"
                current_phase_entry['waveformAnalysis']['resistance'] = f"Resistance rise pattern is abnormal (Rise: {resistance_rise_magnitude:.1f} μΩ, StdDev: {current_data_summary['resistance_std']:.1f} μΩ)."
                current_phase_entry['waveformAnalysis']['current'] = f"Current drop is insufficient or erratic (Drop: {current_drop_magnitude:.1f} A)."
                current_phase_entry['waveformAnalysis']['travel'] = f"Travel decrease is insufficient or erratic (Drop: {travel_drop_magnitude:.1f} mm)."
                
                current_phase_entry['details']['resistance'] = f"Abnormal rise {current_data_summary['resistance_min']:.0f} to {current_data_summary['resistance_max']:.0f} µΩ - issue detected"
                current_phase_entry['details']['current'] = f"Erratic drop {current_data_summary['current_max']:.0f} to {current_data_summary['current_min']:.0f} - interruption problem"
                current_phase_entry['details']['travel'] = f"Abnormal drop {current_data_summary['travel_max']:.0f} to {current_data_summary['travel_min']:.0f} - mechanical issue"

        elif phase_id == 5: # Final Open State
            is_resistance_infinite = current_data_summary['resistance_avg'] >= OPEN_CIRCUIT_RESISTANCE_IDEAL
            is_resistance_high_and_flat = (current_data_summary['resistance_avg'] >= OPEN_CIRCUIT_RESISTANCE_MIN and 
                                           current_data_summary['resistance_std'] <= OPEN_CIRCUIT_VARIATION_THRESHOLD)
            is_current_healthy = abs(current_data_summary['current_avg'] - OPEN_CIRCUIT_CURRENT_BASELINE) < 20 and current_data_summary['current_std'] < 20
            is_travel_low_stable = current_data_summary['travel_std'] < 5 and abs(current_data_summary['travel_avg'] - TRAVEL_MIN_POSITION) < 20

            if (is_resistance_infinite or is_resistance_high_and_flat) and is_current_healthy and is_travel_low_stable:
                current_data_summary['programmatic_health_status'] = "Healthy"
                current_phase_entry['waveformAnalysis']['resistance'] = f"Resistance remains high and stable (Avg: {current_data_summary['resistance_avg']:.1f} μΩ)."
                current_phase_entry['waveformAnalysis']['current'] = f"Current remains stable around its open circuit baseline."
                current_phase_entry['waveformAnalysis']['travel'] = f"Travel remains stable at the fully open position."
                
                current_phase_entry['details']['resistance'] = f"High resistance (~{current_data_summary['resistance_avg']:.0f} µΩ) - contacts fully open"
                current_phase_entry['details']['current'] = f"Low current (~{current_data_summary['current_avg']:.0f}) - no current flow"
                current_phase_entry['details']['travel'] = f"Constant at ~{current_data_summary['travel_avg']:.0f} - contacts fully separated"
            else:
                current_data_summary['programmatic_health_status'] = "Unhealthy - Final State Issue"
                current_phase_entry['waveformAnalysis']['resistance'] = f"Resistance did not return to its expected high value or was unstable."
                current_phase_entry['waveformAnalysis']['current'] = f"Current was not stable around its open circuit baseline."
                current_phase_entry['waveformAnalysis']['travel'] = f"Travel was unstable or did not reach the full open position."
                
                current_phase_entry['details']['resistance'] = f"Abnormal resistance (~{current_data_summary['resistance_avg']:.0f} µΩ) - leakage detected"
                current_phase_entry['details']['current'] = f"Abnormal current (~{current_data_summary['current_avg']:.0f}) - residual flow"
                current_phase_entry['details']['travel'] = f"Unstable at ~{current_data_summary['travel_avg']:.0f} - rebound/drift"

        # Update Coil Analysis with descriptions
        current_phase_entry['waveformAnalysis']['coilAnalysis']['closeCoil']['status'] = current_data_summary['close_coil_state']
        current_phase_entry['waveformAnalysis']['coilAnalysis']['closeCoil']['measuredValues'] = f"Avg: {current_data_summary['close_coil_avg']:.2f} A"
        if current_data_summary['close_coil_state'] == 'Active':
            current_phase_entry['waveformAnalysis']['coilAnalysis']['closeCoil']['description'] = "Close coil is energized, driving contacts to closed position"
        else:
            current_phase_entry['waveformAnalysis']['coilAnalysis']['closeCoil']['description'] = "Close coil is de-energized"
            
        current_phase_entry['waveformAnalysis']['coilAnalysis']['tripCoil1']['status'] = current_data_summary['trip_coil_1_state']
        current_phase_entry['waveformAnalysis']['coilAnalysis']['tripCoil1']['measuredValues'] = f"Avg: {current_data_summary['trip_coil_1_avg']:.2f} A"
        if current_data_summary['trip_coil_1_state'] == 'Active':
            current_phase_entry['waveformAnalysis']['coilAnalysis']['tripCoil1']['description'] = "Trip coil 1 is energized, initiating opening operation"
        else:
            current_phase_entry['waveformAnalysis']['coilAnalysis']['tripCoil1']['description'] = "Trip coil 1 is de-energized"
            
        current_phase_entry['waveformAnalysis']['coilAnalysis']['tripCoil2']['status'] = current_data_summary['trip_coil_2_state']
        current_phase_entry['waveformAnalysis']['coilAnalysis']['tripCoil2']['measuredValues'] = f"Avg: {current_data_summary['trip_coil_2_avg']:.2f} A"
        if current_data_summary['trip_coil_2_state'] == 'Active':
            current_phase_entry['waveformAnalysis']['coilAnalysis']['tripCoil2']['description'] = "Trip coil 2 is energized, providing redundant trip capability"
        else:
            # Check if it's failed or just inactive
            if current_data_summary['trip_coil_2_avg'] == 0.0 and current_data_summary['trip_coil_1_state'] == 'Active':
                current_phase_entry['waveformAnalysis']['coilAnalysis']['tripCoil2']['description'] = "Trip coil 2 appears non-functional - potential redundancy loss"
            else:
                current_phase_entry['waveformAnalysis']['coilAnalysis']['tripCoil2']['description'] = "Trip coil 2 is de-energized"

        # --- LLM Enhancement ---
        if llm:
            try:
                llm_prompt = generate_llm_description_prompt(current_data_summary, current_phase_entry)
                
                # Invoke LLM
                response = llm.invoke([HumanMessage(content=llm_prompt)])
                content = response.content.replace("```json", "").replace("```", "").strip()
                llm_output = json.loads(content)
                
                current_phase_entry['details']['characteristics'] = llm_output.get('key_characteristics', [])
                current_phase_entry['eventSynopsis'] = llm_output.get('event_synopsis', "")
                current_phase_entry['diagnosticVerdict'] = llm_output.get('diagnostic_verdict_details', "")
                
                # Extract confidence score
                llm_confidence = llm_output.get('confidence_score', 0)
                if isinstance(llm_confidence, (int, float)):
                    current_phase_entry['confidence'] = max(0, min(100, int(llm_confidence)))
                else:
                    current_phase_entry['confidence'] = 50
                    
            except Exception as e:
                print(f"Error calling LLM for phase {phase_id}: {e}")
                # Fallback is already set by programmatic logic or defaults
                current_phase_entry['diagnosticVerdict'] = f"Analysis based on programmatic data: {current_data_summary['programmatic_health_status']}. (LLM enhancement failed)"
                current_phase_entry['confidence'] = 50

        # Assign status for CBHI calculation
        current_phase_entry['status'] = current_data_summary.get('programmatic_health_status', "Unknown")

        final_segmented_phases.append(current_phase_entry)

    json_output_template['phaseWiseAnalysis'] = final_segmented_phases
    return json_output_template
