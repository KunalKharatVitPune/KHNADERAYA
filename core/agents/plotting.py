# Previous Name: analysis/agents/dcrm_analysis.py
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
import json
from langchain_core.messages import HumanMessage
import streamlit as st

def get_dcrm_prompt(data_str):
    return f"""
    I have extracted data from a DCRM (Dynamic Contact Resistance Measurement) graph.
    Data (Sampled): {data_str}
    
    The columns are:
    - 'time': Time in milliseconds.
    - 'curr': Current signal amplitude (Blue curve) - represents the test current flowing through the contacts.
    - 'res': Dynamic Resistance amplitude (Green curve) - represents the contact resistance in micro-ohms (µΩ).
    - 'travel': Travel signal amplitude (Red curve) - represents the mechanical position/displacement of the moving contact.

    IMPORTANT: Higher values mean the signal is HIGHER on the graph.
    
    I have also provided the image of the graph. Use the visual information from the image to cross-reference with the data.

    === HEALTHY DCRM SIGNATURE REFERENCE ===
    
    **Resistance (Green) - Healthy Characteristics:**
    - Pre-contact: Infinite/Very High (off-scale or flat at top)
    - Arcing engagement: Drops sharply with moderate spikes (arcing activity), typically 100-500 µΩ
    - Main conduction: LOW and STABLE (30-80 µΩ for healthy contacts), minimal oscillation (<10 µΩ variance)
    - Parting: Sharp rise with spikes (arcing during separation)
    - Final open: Returns to infinite/very high (off-scale)
    
    **Current (Blue) - Healthy Characteristics:**
    - Pre-contact: Near zero baseline
    - Arcing engagement: Begins rising as circuit closes
    - Main conduction: Stable at test current level (plateau)
    - Parting: Maintained until final separation
    - Final open: Drops to zero
    
    **Travel (Red) - Healthy Characteristics:**
    - Pre-contact: Increasing linearly (contacts approaching)
    - Arcing engagement: Continues increasing
    - Main conduction: Reaches MAXIMUM and plateaus (fully closed position)
    - Parting: Decreases linearly (contacts separating)
    - Final open: Stabilizes at minimum (fully open position)

    === TASK: SEGMENT INTO 5 KINEMATIC ZONES ===
    
    Use ALL THREE curves together for accurate boundary detection. Each zone represents a distinct physical state of the circuit breaker.
    
    **Zone 1: Pre-Contact Travel (Initial Closing Motion)**
    *   **Physical Meaning**: The moving contact is traveling toward the stationary contact but has NOT yet made electrical contact. This is pure mechanical motion with no current flow.
    *   **Start**: time = 0 ms
    *   **End Boundary**: Detect when CURRENT (blue) FIRST starts rising significantly from baseline.
        *   Cross-reference: Resistance (green) should still be very high/infinite
        *   Cross-reference: Travel (red) should be steadily increasing
        *   **Typical Duration**: 80-120 ms
        *   **Detection Logic**: Find the point where 'curr' rises above baseline noise (e.g., >5% of max current)
    
    **Zone 2: Arcing Contact Engagement (Initial Electrical Contact)**
    *   **Physical Meaning**: The arcing contacts (W-Cu tips) make first contact and establish an electrical path. Current begins flowing through a small contact area, causing arcing and resistance fluctuations. This is the "make" transition.
    *   **Start**: End of Zone 1
    *   **End Boundary**: Detect when resistance SETTLES after initial spike activity.
        *   Primary indicator: Resistance (green) drops from high values, exhibits spikes, then STABILIZES to low plateau
        *   Cross-reference: Current (blue) should be rising/stabilizing
        *   Cross-reference: Travel (red) continues increasing toward maximum
        *   **Typical Duration**: 20-40 ms (Zone 2 typically ends around 110-150 ms total time)
        *   **Detection Logic**: Find where 'res' completes its descent and spike activity, settling into a stable low range
    
    **Zone 3: Main Contact Conduction (Fully Closed State)**
    *   **Physical Meaning**: The main contacts (Ag-plated) are fully engaged, providing a large, stable contact area. This is the "healthy contact" signature zone - resistance should be at its MINIMUM and STABLE. The breaker is in its fully closed, current-carrying state.
    *   **Start**: End of Zone 2
    *   **End Boundary**: Detect when the breaker begins OPENING (travel reverses direction).
        *   Primary indicator: Travel (red) reaches MAXIMUM and starts to DESCEND
        *   Cross-reference: Resistance (green) should remain low and stable throughout this zone
        *   Cross-reference: Current (blue) should be stable at test level
        *   **Typical Duration**: 100-200 ms (this is the longest zone, representing the dwell time)
        *   **Detection Logic**: Find the peak of 'travel' curve and the point where it starts decreasing
    
    **Zone 4: Main Contact Parting (Breaking/Opening Transition)**
    *   **Physical Meaning**: The main contacts are separating. As the contact area decreases, resistance rises sharply. Arcing occurs during the final separation of the arcing contacts. This is the "break" transition - the most critical phase for fault detection.
    *   **Start**: End of Zone 3
    *   **End Boundary**: Detect when resistance STABILIZES at high value after parting spikes.
        *   Primary indicator: Resistance (green) shoots UP, exhibits parting spikes, then STABILIZES at high/infinite value
        *   Cross-reference: Travel (red) should be decreasing (opening motion)
        *   Cross-reference: Current (blue) may drop or fluctuate during final arc extinction
        *   **Typical Duration**: 40-80 ms (Zone 4 typically ends around 280-340 ms total time)
        *   **Detection Logic**: Find where 'res' completes its rise and spike activity, becoming constant at high value
        *   **CRITICAL**: Do NOT extend this zone too long - end AS SOON AS resistance stabilizes
    
    **Zone 5: Final Open State (Fully Open)**
    *   **Physical Meaning**: The contacts are fully separated with an air gap. No current flows, resistance is infinite. The breaker is in its fully open, non-conducting state.
    *   **Start**: End of Zone 4
    *   **End**: The last time point in the dataset
    *   **Characteristics**: 
        *   Resistance (green): Very high/infinite (flat line at top)
        *   Current (blue): Zero or near-zero
        *   Travel (red): Stable at minimum (fully open position)

    **MULTI-CURVE ANALYSIS STRATEGY:**
    1. Use Current (blue) to identify Zone 1 → Zone 2 transition (first current rise)
    2. Use Resistance (green) to identify Zone 2 → Zone 3 transition (resistance settles to low plateau)
    3. Use Travel (red) to identify Zone 3 → Zone 4 transition (travel peak and reversal)
    4. Use Resistance (green) to identify Zone 4 → Zone 5 transition (resistance stabilizes at high value)
    5. Always cross-validate boundaries using all three curves for consistency

    **OUTPUT FORMAT (Strict JSON)**
    Return ONLY this JSON object:
    {{
      "zones": {{
        "zone_1_pre_contact": {{ "start_ms": float, "end_ms": float, "justification": "string (explain which curve indicators were used)" }},
        "zone_2_arcing_engagement": {{ "start_ms": float, "end_ms": float, "justification": "string (explain which curve indicators were used)" }},
        "zone_3_main_conduction": {{ "start_ms": float, "end_ms": float, "justification": "string (explain which curve indicators were used)" }},
        "zone_4_parting": {{ "start_ms": float, "end_ms": float, "justification": "string (explain which curve indicators were used)" }},
        "zone_5_final_open": {{ "start_ms": float, "end_ms": float, "justification": "string (explain which curve indicators were used)" }}
      }},
      "report_card": {{
        "opening_speed": {{ "status": "Pass"|"Warning"|"Fail", "comment": "Assessment of travel curve steepness" }},
        "contact_wear": {{ "status": "Pass"|"Warning"|"Fail", "comment": "Based on resistance fluctuations in Zone 2/4" }},
        "timing_consistency": {{ "status": "Pass"|"Warning"|"Fail", "comment": "Are phases within expected ranges?" }},
        "overall_health": {{ "status": "Healthy"|"Needs Review"|"Critical", "comment": "Overall summary" }}
      }},
      "detailed_analysis": "Provide a comprehensive technical analysis (in Markdown)..."
    }}
    """

def create_dcrm_plot(df, zones):
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add Traces
    fig.add_trace(go.Scatter(x=df['Time_ms'], y=df['Current'], name="Current (A)", line=dict(color='#2980b9', width=2)), secondary_y=False)
    fig.add_trace(go.Scatter(x=df['Time_ms'], y=df['Resistance'], name="Resistance (uOhm)", line=dict(color='#27ae60', width=2)), secondary_y=False)
    fig.add_trace(go.Scatter(x=df['Time_ms'], y=df['Travel'], name="Travel (mm)", line=dict(color='#c0392b', width=2)), secondary_y=True)

    # Zone Colors
    zone_colors = {
        "zone_1_pre_contact": "rgba(52, 152, 219, 0.1)",
        "zone_2_arcing_engagement": "rgba(231, 76, 60, 0.1)",
        "zone_3_main_conduction": "rgba(46, 204, 113, 0.1)",
        "zone_4_parting": "rgba(155, 89, 182, 0.1)",
        "zone_5_final_open": "rgba(149, 165, 166, 0.1)"
    }

    # Add Zone Rectangles
    for zone_name, details in zones.items():
        start = details.get("start_ms")
        end = details.get("end_ms")
        color = zone_colors.get(zone_name, "rgba(0,0,0,0)")
        
        if start is not None and end is not None:
            fig.add_vrect(
                x0=start, x1=end,
                fillcolor=color, opacity=1,
                layer="below", line_width=0,
                annotation_text=zone_name.split('_')[1].upper(),
                annotation_position="top left",
                annotation_font_color="#7f8c8d"
            )

    fig.update_layout(
        title_text="<b>Main Signals & Zones</b>",
        height=500,
        hovermode="x unified",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Segoe UI, sans-serif"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=60, b=20)
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
    fig.update_yaxes(title_text="Current / Resistance", secondary_y=False, showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
    fig.update_yaxes(title_text="Travel", secondary_y=True, showgrid=False)

    return fig

def create_velocity_plot(df):
    # Calculate Velocity (Derivative of Travel)
    # V = d(Travel) / d(Time)
    # Units: mm/ms = m/s
    df['Velocity'] = df['Travel'].diff() / df['Time_ms'].diff()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Time_ms'], y=df['Velocity'], name="Velocity (m/s)", line=dict(color='#e67e22', width=2), fill='tozeroy'))
    
    fig.update_layout(
        title_text="<b>Contact Velocity Profile</b>",
        height=300,
        hovermode="x unified",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Segoe UI, sans-serif"),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
    fig.update_yaxes(title_text="Velocity (m/s)", showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
    return fig

def create_resistance_zoom_plot(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Time_ms'], y=df['Resistance'], name="Resistance", line=dict(color='#27ae60', width=2)))
    
    fig.update_layout(
        title_text="<b>Detailed Resistance (Log Scale)</b>",
        height=300,
        hovermode="x unified",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Segoe UI, sans-serif"),
        yaxis_type="log", # Log scale to see details
        margin=dict(l=20, r=20, t=40, b=20)
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
    fig.update_yaxes(title_text="Resistance (uOhm)", showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
    return fig

def enrich_data_with_zones(df, llm):
    """
    Uses the LLM to identify zones and adds a 'Zone' column to the DataFrame.
    """
    try:
        # 1. Prepare Text Data
        data_str = df.to_string(index=False)
        prompt_text = get_dcrm_prompt(data_str)

        # 2. Prepare Image Data
        # Create a simplified plot for the LLM to "see"
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=df['Time_ms'], y=df['Current'], name="Current", line=dict(color='blue')), secondary_y=False)
        fig.add_trace(go.Scatter(x=df['Time_ms'], y=df['Resistance'], name="Resistance", line=dict(color='green')), secondary_y=False)
        fig.add_trace(go.Scatter(x=df['Time_ms'], y=df['Travel'], name="Travel", line=dict(color='red')), secondary_y=True)
        fig.update_layout(title="DCRM Graph", showlegend=True)
        
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
        result = json.loads(content)
        zones = result.get("zones", {})
        
        # Initialize Zone column
        df['Zone'] = "Unknown"
        
        for zone_name, details in zones.items():
            start = details.get("start_ms")
            end = details.get("end_ms")
            if start is not None and end is not None:
                # Map zone name to a simpler label (e.g., "Zone 1")
                short_name = zone_name.split('_')[1] # "1", "2", etc.
                mask = (df['Time_ms'] >= start) & (df['Time_ms'] <= end)
                df.loc[mask, 'Zone'] = f"Zone {short_name}"
        
        return df, zones
    except Exception as e:
        st.error(f"Enrichment failed: {str(e)}")
        return df, {}
