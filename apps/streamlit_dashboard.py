# Previous Name: streamlit_app_v2.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import sys

# Add project root to sys.path to allow importing from core
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Ensure API key is set
if not os.getenv("GOOGLE_API_KEY"):
    st.error("GOOGLE_API_KEY not found in environment variables. Please check your .env file.")
    st.stop()

from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
import io
from langchain_core.messages import HumanMessage
from core.agents.plotting import create_dcrm_plot, create_velocity_plot, create_resistance_zoom_plot, get_dcrm_prompt
from core.calculators.kpi import calculate_kpis
from core.calculators.cbhi import compute_cbhi
from core.signal.phases import analyze_dcrm_data
from core.engines.rules import analyze_dcrm_advanced
from core.agents.diagnosis import detect_fault, standardize_input
from core.utils.report_generator import generate_dcrm_json
from core.agents.recommendation import generate_recommendations

# Optional ViT Model (requires PyTorch compatibility)
try:
    from core.models.vit_classifier import predict_dcrm_image, plot_resistance_for_vit
    VIT_AVAILABLE = True
except Exception as e:
    print(f"ViT Model not available: {e}")
    VIT_AVAILABLE = False
    predict_dcrm_image = None
    plot_resistance_for_vit = None

# --- Configuration & CSS ---
st.set_page_config(page_title="DCRM Analyzer Pro", page_icon="⚡", layout="wide")

def load_css():
    st.markdown("""
        <style>
        /* Main Font */
        html, body, [class*="css"] {
            font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        }
        /* Headers */
        h1, h2, h3 {
            color: #2c3e50;
            font-weight: 600;
        }
        /* Metrics */
        [data-testid="stMetricValue"] {
            font-size: 2rem;
            color: #2980b9;
        }
        /* Cards/Containers */
        .stExpander {
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        /* Sidebar */
        section[data-testid="stSidebar"] {
            background-color: #f8f9fa;
        }
        /* Status Indicators */
        .status-pass { color: #27ae60; font-weight: bold; }
        .status-warning { color: #f39c12; font-weight: bold; }
        .status-fail { color: #c0392b; font-weight: bold; }
        </style>
    """, unsafe_allow_html=True)

def main():
    load_css()
    
    with st.sidebar:
        st.title("⚡ DCRM Analyzer")
        st.markdown("---")
        mode = st.radio("Select Module", ["General Chat", "DCRM Analysis"], index=1)
        
        # Segmentation method selector removed - using robust signal processing by default
        
        st.markdown("---")
        st.caption("Powered by Gemini 1.5 Flash")

    st.header(f"{mode}")

    uploaded_file = st.file_uploader("Upload DCRM Data (CSV)", type="csv")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

            if mode == "General Chat":
                st.info("💡 **Tip:** You can ask for graphs! Try: 'Plot the resistance in Zone 2'")
                
                # Zone Enrichment Option
                if st.checkbox("✨ Enrich Data with Zones"):
                    with st.spinner("Detecting Zones..."):
                        df, zones = enrich_data_with_zones(df, llm)
                        st.success("Data enriched with 'Zone' column!")
                
                with st.expander("View Data"):
                    st.dataframe(df.head())
                
                # Agent with Plotting Capabilities
                agent = create_pandas_dataframe_agent(
                    llm, 
                    df, 
                    verbose=True, 
                    allow_dangerous_code=True, 
                    handle_parsing_errors=True,
                    prefix="""
                    You are a data analysis agent. 
                    If the user asks for a plot or graph:
                    1. Use `plotly.graph_objects` (import as go) or `plotly.express` (import as px).
                    2. Create the figure object `fig`.
                    3. Display it using `st.plotly_chart(fig)`.
                    4. Do NOT use matplotlib.
                    5. Ensure you import streamlit as st inside the code execution.
                    """
                )
                
                st.divider()
                user_question = st.chat_input("Ask a question or request a plot...")
                
                if user_question:
                    st.chat_message("user").write(user_question)
                    with st.spinner("Thinking..."):
                        try:
                            # We use the agent directly for everything now, as it handles both plotting and QA
                            response = agent.run(user_question)
                            
                            with st.chat_message("assistant"):
                                st.write(response)
                                
                        except Exception as e:
                            st.error(f"An error occurred: {str(e)}")

            elif mode == "DCRM Analysis":
                if st.button("🚀 Run Advanced Analysis", type="primary", width='stretch'):
                    with st.spinner("Performing Kinematic Segmentation & Diagnostics..."):
                        try:
                            # 1. Deterministic KPI Calculation
                            kpi_results = calculate_kpis(df)
                            kpis = kpi_results['kpis']
                            # cbhi_score will be calculated later after AI and Phase analysis
                            
                            # 1.5. Phase Segmentation (using AI Agent)
                            with st.spinner("Running AI-Based Phase Segmentation..."):
                                # Use analyze_dcrm_data for robust segmentation and status
                                phase_analysis_result = analyze_dcrm_data(df, llm)
                                
                                # Extract zones for plotting
                                phase_to_zone_map = {
                                    1: "zone_1_pre_contact",
                                    2: "zone_2_arcing_engagement",
                                    3: "zone_3_main_conduction",
                                    4: "zone_4_parting",
                                    5: "zone_5_final_open"
                                }
                                zones = {}
                                phase_timings = []
                                
                                if 'phaseWiseAnalysis' in phase_analysis_result:
                                    st.success("✓ AI Segmentation complete!")
                                    for phase in phase_analysis_result['phaseWiseAnalysis']:
                                        p_num = phase.get('phaseNumber')
                                        zone_key = phase_to_zone_map.get(p_num)
                                        start_time = phase.get('startTime', 0)
                                        end_time = phase.get('endTime', 0)
                                        duration = end_time - start_time
                                        
                                        if zone_key:
                                            zones[zone_key] = {
                                                'start_ms': start_time,
                                                'end_ms': end_time
                                            }
                                        
                                        phase_timings.append({
                                            "Phase": phase.get('name'),
                                            "Start (ms)": start_time,
                                            "End (ms)": end_time,
                                            "Duration (ms)": duration
                                        })
                                else:
                                    st.warning("⚠️ AI Segmentation returned no zones.")
                            
                            # 2. Rule Engine & AI Agent Analysis
                            # Prepare KPIs for Raj modules (needs specific format)
                            # kpis dict from calculate_kpis_and_score is flat: {'closing_time': 45.15, ...}
                            # Raj modules expect: {'kpis': [{'name': 'Closing Time', 'value': ...}, ...]} or dict {'Closing Time (ms)': ...}
                            # Let's construct the dict format expected by raj_rule_engine (it handles dicts loosely but prefers specific keys)
                            
                            # Mapping current KPIs to Raj expected keys
                            raj_kpis = {
                                "Closing Time (ms)": kpis.get('closing_time'),
                                "Opening Time (ms)": kpis.get('opening_time'),
                                "Contact Speed (m/s)": kpis.get('contact_speed'),
                                "DLRO Value (µΩ)": kpis.get('dlro'),
                                "Peak Resistance (µΩ)": kpis.get('peak_resistance'),
                                "Peak Close Coil Current (A)": kpis.get('peak_close_coil'),
                                "Peak Trip Coil 1 Current (A)": kpis.get('peak_trip_coil_1'),
                                "Peak Trip Coil 2 Current (A)": kpis.get('peak_trip_coil_2'),
                                "SF6 Pressure (bar)": kpis.get('sf6_pressure'),
                                "Ambient Temperature (°C)": kpis.get('ambient_temp'),
                                "Main Wipe (mm)": kpis.get('main_wipe'),
                                "Arc Wipe (mm)": kpis.get('arc_wipe'),
                                "Contact Travel Distance (mm)": kpis.get('contact_travel')
                            }
                            
                            # Also construct the list format for raj_ai_agent if needed, but detect_fault takes sample_kpis dict
                            # detect_fault expects: {'kpis': [{'name': '...', 'value': ...}]}
                            raj_ai_kpis = {
                                "kpis": [
                                    {"name": "Closing Time", "unit": "ms", "value": kpis.get('closing_time')},
                                    {"name": "Opening Time", "unit": "ms", "value": kpis.get('opening_time')},
                                    {"name": "DLRO Value", "unit": "µΩ", "value": kpis.get('dlro')},
                                    {"name": "Peak Resistance", "unit": "µΩ", "value": kpis.get('peak_resistance')},
                                    {"name": "Contact Speed", "unit": "m/s", "value": kpis.get('contact_speed')},
                                    {"name": "Peak Close Coil Current", "unit": "A", "value": kpis.get('peak_close_coil')},
                                    {"name": "Peak Trip Coil 1 Current", "unit": "A", "value": kpis.get('peak_trip_coil_1')},
                                    {"name": "Peak Trip Coil 2 Current", "unit": "A", "value": kpis.get('peak_trip_coil_2')},
                                    {"name": "SF6 Pressure", "unit": "bar", "value": kpis.get('sf6_pressure')},
                                    {"name": "Ambient Temperature", "unit": "°C", "value": kpis.get('ambient_temp')}
                                ]
                            }

                            # Prepare row values for Rule Engine
                            # We need 401 points. If df has > 401, take first 401. If < 401, pad?
                            # Assuming standard 400ms data at 1ms sample rate.
                            # If Time_ms is present, we can try to resample or just take the Resistance column values.
                            # raj_rule_engine.standardize_input handles this.
                            
                            # Create a temp df for standardize_input
                            temp_df = df[['Resistance']].copy()
                            # Ensure we have enough data or handle it
                            if len(temp_df) < 401:
                                # Pad with last value
                                last_val = temp_df.iloc[-1, 0]
                                padding = pd.DataFrame({'Resistance': [last_val] * (401 - len(temp_df))})
                                temp_df = pd.concat([temp_df, padding], ignore_index=True)
                            
                            std_df = standardize_input(temp_df)
                            row_values = std_df.iloc[0].values.tolist()

                            # --- Run Rule Engine ---
                            rule_engine_result = analyze_dcrm_advanced(row_values, raj_kpis)
                            
                            # --- Run AI Agent ---
                            # detect_fault takes (df, sample_kpis)
                            # We can pass the original df, it calls standardize_input internally
                            ai_agent_result = detect_fault(df, raj_ai_kpis)

                            # --- Run ViT Model (Visual Inspection) ---
                            vit_result = None
                            # --- Run ViT Model (Visual Inspection) ---
                            vit_result = None
                            vit_plot_path = "temp_vit_plot.png"
                            
                            # Always generate plot (doesn't require torch)
                            plot_generated = False
                            try:
                                if plot_resistance_for_vit(df, vit_plot_path):
                                    plot_generated = True
                            except Exception as e:
                                print(f"ViT Plot generation failed: {e}")

                            # Try prediction if plot exists and VIT is available
                            if plot_generated and VIT_AVAILABLE:
                                try:
                                    vit_class, vit_conf, vit_details = predict_dcrm_image(vit_plot_path)
                                    if vit_class: # Ensure we got a result
                                        vit_result = {
                                            "class": vit_class,
                                            "confidence": vit_conf,
                                            "details": vit_details
                                        }
                                except Exception as e:
                                    print(f"ViT Prediction failed: {e}")



                            # ============================================
                            # SECTION 1: CBHI SCORE & KPIs
                            # ============================================
                            
                            # Calculate CBHI using new logic
                            # Prepare phase_data for CBHI
                            cbhi_phase_data = {}
                            if 'phaseWiseAnalysis' in phase_analysis_result:
                                for phase in phase_analysis_result['phaseWiseAnalysis']:
                                    p_name = f"Phase {phase.get('phaseNumber')}"
                                    cbhi_phase_data[p_name] = {
                                        "status": phase.get('status', 'Unknown'),
                                        "confidence": phase.get('confidence', 0)
                                    }
                            
                            # Compute Score
                            cbhi_score = compute_cbhi(raj_ai_kpis['kpis'], ai_agent_result, cbhi_phase_data)

                            # --- Run Recommendation Agent ---
                            with st.spinner("Generating Maintenance Recommendations..."):
                                recommendations = generate_recommendations(
                                    kpis=kpis,
                                    cbhi_score=cbhi_score,
                                    rule_faults=rule_engine_result.get("Fault_Detection", []),
                                    ai_faults=ai_agent_result.get("Fault_Detection", []),
                                    llm=llm
                                )
                            
                            st.markdown("## 🏆 Composite Breaker Health Index (CBHI)")
                            
                            # CBHI Score Display
                            cbhi_col1, cbhi_col2, cbhi_col3 = st.columns([1, 2, 1])
                            
                            with cbhi_col1:
                                st.markdown("")  # Spacer
                            
                            with cbhi_col2:
                                # Large centered score
                                if cbhi_score >= 90:
                                    score_color = "#27ae60"  # Green
                                    status_text = "✅ Excellent Condition"
                                    status_color = "success"
                                    gradient = "linear-gradient(135deg, #11998e 0%, #38ef7d 100%)"
                                elif cbhi_score >= 75:
                                    score_color = "#f39c12"  # Orange
                                    status_text = "⚠️ Good - Minor Review Needed"
                                    status_color = "warning"
                                    gradient = "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)"
                                else:
                                    score_color = "#e74c3c"  # Red
                                    status_text = "🚨 Critical Attention Required"
                                    status_color = "error"
                                    gradient = "linear-gradient(135deg, #fa709a 0%, #fee140 100%)"
                                
                                st.markdown(f"""
                                <div style='text-align: center; padding: 30px; background: {gradient}; border-radius: 20px; box-shadow: 0 8px 16px rgba(0,0,0,0.2);'>
                                    <h1 style='color: white; font-size: 5rem; margin: 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);'>{cbhi_score}</h1>
                                    <p style='color: white; font-size: 1.4rem; margin: 10px 0; font-weight: 500;'>out of 100</p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                st.markdown("<br>", unsafe_allow_html=True)
                                
                                if status_color == "success":
                                    st.success(status_text)
                                elif status_color == "warning":
                                    st.warning(status_text)
                                else:
                                    st.error(status_text)
                            
                            with cbhi_col3:
                                st.markdown("")  # Spacer
                            
                            st.markdown("---")
                            
                            # KPIs Section
                            st.markdown("### 📊 Key Performance Indicators")
                            st.caption("Measured parameters from DCRM test data")
                            
                            # Timing & Motion
                            with st.container():
                                st.markdown("**⏱️ Timing & Motion**")
                                k1, k2, k3, k4 = st.columns(4)
                                k1.metric("Closing Time", f"{kpis['closing_time']} ms", help="Time taken for contacts to close")
                                k2.metric("Opening Time", f"{kpis['opening_time']} ms", help="Time taken for contacts to open")
                                k3.metric("Contact Speed", f"{kpis['contact_speed']} m/s", help="Average contact movement speed")
                                k4.metric("Contact Travel", f"{kpis['contact_travel']} mm", help="Total mechanical travel distance")
                            
                            st.markdown("")
                            
                            # Contact Health
                            with st.container():
                                st.markdown("**🔌 Contact Health**")
                                k5, k6, k7, k8 = st.columns(4)
                                k5.metric("DLRO", f"{kpis['dlro']} µΩ", help="Dynamic Low Resistance - contact quality indicator")
                                k6.metric("Peak Resistance", f"{kpis['peak_resistance']} µΩ", help="Maximum resistance during operation")
                                k7.metric("Main Wipe", f"{kpis['main_wipe']} mm", help="Main contact wipe distance")
                                k8.metric("Arc Wipe", f"{kpis['arc_wipe']} mm", help="Arcing contact wipe distance")
                            
                            st.markdown("")
                            
                            # Electrical & Environment
                            with st.container():
                                st.markdown("**⚡ Electrical & Environment**")
                                k9, k10, k11, k12, k13 = st.columns(5)
                                k9.metric("Close Coil", f"{kpis['peak_close_coil']} A", help="Peak closing coil current")
                                k10.metric("Trip Coil 1", f"{kpis['peak_trip_coil_1']} A", help="Peak trip coil 1 current")
                                k11.metric("Trip Coil 2", f"{kpis['peak_trip_coil_2']} A", help="Peak trip coil 2 current")
                                k12.metric("Temperature", f"{kpis['ambient_temp']} °C", help="Ambient temperature during test")
                                k13.metric("SF6 Pressure", f"{kpis['sf6_pressure']} bar", help="SF6 gas pressure")

                            st.markdown("---")
                            
                            # ============================================
                            # SECTION 2: SEGMENTED GRAPH WITH ZONES
                            # ============================================
                            
                            st.markdown("## 📈 DCRM Waveforms with Phase Segmentation")
                            st.caption("Programmatic phase detection using resistance, current, and travel thresholds")
                            
                            # Generate the full JSON report first to get phase boundaries
                            with st.spinner("Analyzing phases and generating report..."):
                                full_report = generate_dcrm_json(
                                    df=df,
                                    kpis=kpis,
                                    cbhi_score=cbhi_score,
                                    rule_result=rule_engine_result,
                                    ai_result=ai_agent_result,
                                    llm=llm,
                                    vit_result=vit_result,
                                    phase_analysis_result=phase_analysis_result,
                                    recommendations=recommendations
                                )
                            
                            # Get phase boundaries and create zone timing table
                            phase_to_zone_map = {
                                "Pre-Contact Travel": "zone_1_pre_contact",
                                "Arcing Contact Engagement & Arc Initiation": "zone_2_arcing_engagement",
                                "Main Contact Conduction": "zone_3_main_conduction",
                                "Main Contact Parting & Arc Elongation": "zone_4_parting",
                                "Final Open State": "zone_5_final_open"
                            }
                            
                            zones = {}
                            phase_timings = []
                            
                            if 'phaseWiseAnalysis' in full_report:
                                for phase in full_report['phaseWiseAnalysis']:
                                    phase_name = phase.get('name', '')
                                    zone_key = phase_to_zone_map.get(phase_name)
                                    start_time = phase.get('startTime', 0)
                                    end_time = phase.get('endTime', 0)
                                    duration = end_time - start_time
                                    
                                    if zone_key:
                                        zones[zone_key] = {
                                            'start_ms': start_time,
                                            'end_ms': end_time
                                        }
                                    
                                    phase_timings.append({
                                        "Phase": phase_name,
                                        "Start (ms)": start_time,
                                        "End (ms)": end_time,
                                        "Duration (ms)": duration
                                    })
                            
                            # Display graph
                            fig_main = create_dcrm_plot(df, zones)
                            st.plotly_chart(fig_main, width='stretch')
                            
                            # Display phase timing table
                            if phase_timings:
                                st.markdown("#### ⏱️ Phase Timing Breakdown")
                                timing_df = pd.DataFrame(phase_timings)
                                st.dataframe(timing_df, width='stretch', hide_index=True)

                            st.markdown("---")
                            
                            # ============================================
                            # SECTION 3: PHASE-WISE DETAILED ANALYSIS
                            # ============================================
                            
                            st.markdown("## 🔍 Phase-Wise Detailed Analysis")
                            st.caption("AI-enhanced analysis of each operational phase with diagnostic verdicts")
                            
                            if 'phaseWiseAnalysis' in full_report:
                                for phase in full_report['phaseWiseAnalysis']:
                                    phase_num = phase.get('phaseNumber', 0)
                                    phase_name = phase.get('name', 'Unknown Phase')
                                    phase_title = phase.get('phaseTitle', phase_name)
                                    phase_desc = phase.get('description', '')
                                    confidence = phase.get('confidence', 0)
                                    
                                    # Color code by phase
                                    phase_colors = {
                                        1: "#ff9800",  # Orange
                                        2: "#ff26bd",  # Pink
                                        3: "#4caf50",  # Green
                                        4: "#2196f3",  # Blue
                                        5: "#a629ff"   # Purple
                                    }
                                    phase_color = phase_colors.get(phase_num, "#cccccc")
                                    
                                    with st.expander(f"**Phase {phase_num}: {phase_name}** | Confidence: {confidence}%", expanded=False):
                                        # Phase header
                                        st.markdown(f"""
                                        <div style='padding: 15px; background-color: {phase_color}22; border-left: 5px solid {phase_color}; border-radius: 5px; margin-bottom: 15px;'>
                                            <h4 style='margin: 0; color: {phase_color};'>{phase_title}</h4>
                                            <p style='margin: 5px 0 0 0; color: #666;'>{phase_desc}</p>
                                        </div>
                                        """, unsafe_allow_html=True)
                                        
                                        # Event Synopsis
                                        event_synopsis = phase.get('eventSynopsis', '')
                                        if event_synopsis:
                                            st.markdown("**📋 Event Synopsis**")
                                            st.info(event_synopsis)
                                        
                                        # Key Characteristics
                                        characteristics = phase.get('details', {}).get('characteristics', [])
                                        if characteristics:
                                            st.markdown("**🔑 Key Characteristics**")
                                            for char in characteristics:
                                                st.markdown(f"- {char}")
                                        
                                        # Waveform Analysis
                                        waveform = phase.get('waveformAnalysis', {})
                                        if waveform:
                                            st.markdown("**📊 Waveform Analysis**")
                                            
                                            wave_col1, wave_col2, wave_col3 = st.columns(3)
                                            
                                            with wave_col1:
                                                st.markdown("**Resistance**")
                                                st.caption(waveform.get('resistance', 'N/A'))
                                            
                                            with wave_col2:
                                                st.markdown("**Current**")
                                                st.caption(waveform.get('current', 'N/A'))
                                            
                                            with wave_col3:
                                                st.markdown("**Travel**")
                                                st.caption(waveform.get('travel', 'N/A'))
                                        
                                        # Diagnostic Verdict
                                        verdict = phase.get('diagnosticVerdict', '')
                                        if verdict:
                                            st.markdown("**🩺 Diagnostic Verdict & Justification**")
                                            st.success(verdict)

                            st.markdown("---")
                            
                            # ============================================
                            # SECTION 4: AI AGENT ANALYSIS
                            # ============================================
                            
                            st.markdown("## 🤖 AI Agent Analysis (Generative)")
                            st.caption("Deep learning-based fault detection using physics-informed signatures")
                            
                            ai_faults = ai_agent_result.get("Fault_Detection", [])
                            if not ai_faults:
                                st.success("✅ **No AI-detected faults found.** System appears healthy based on generative AI analysis.")
                            else:
                                for idx, fault in enumerate(ai_faults):
                                    name = fault.get("defect_name", "Unknown")
                                    conf = fault.get("Confidence", "0%")
                                    sev = fault.get("Severity", "Low")
                                    desc = fault.get("description", "")
                                    
                                    # Check if this is "No Secondary Defect Detected"
                                    if "no secondary defect" in name.lower() or "no defect" in name.lower():
                                        st.success(f"✅ **{name}**")
                                    else:
                                        # Color code by severity
                                        if sev == "Critical":
                                            icon = "🔴"
                                            border_color = "#e74c3c"
                                        elif sev == "High":
                                            icon = "🟠"
                                            border_color = "#f39c12"
                                        else:
                                            icon = "🟡"
                                            border_color = "#f1c40f"
                                        
                                        with st.expander(f"{icon} **{name}** | Confidence: {conf} | Severity: {sev}", expanded=(sev in ["Critical", "High"])):
                                            st.markdown(f"""
                                            <div style='padding: 10px; background-color: {border_color}11; border-left: 4px solid {border_color}; border-radius: 5px;'>
                                                <p style='margin: 0;'><strong>AI Reasoning:</strong> {desc}</p>
                                            </div>
                                            """, unsafe_allow_html=True)

                            st.markdown("---")
                            
                            # ============================================
                            # SECTION 5: RULE ENGINE ANALYSIS
                            # ============================================
                            
                            st.markdown("## ⚙️ Rule-Based Analysis (Deterministic)")
                            st.caption("Threshold-based fault detection using industry standards and expert knowledge")
                            
                            re_faults = rule_engine_result.get("Fault_Detection", [])
                            if not re_faults:
                                st.success("✅ **No rule-based faults detected.** All parameters within acceptable ranges per industry standards.")
                            else:
                                for idx, fault in enumerate(re_faults):
                                    name = fault.get("defect_name", "Unknown")
                                    conf = fault.get("Confidence", "0%")
                                    sev = fault.get("Severity", "Low")
                                    desc = fault.get("description", "")
                                    
                                    # Color code by severity
                                    if sev == "Critical":
                                        icon = "🔴"
                                        border_color = "#e74c3c"
                                    elif sev == "High":
                                        icon = "🟠"
                                        border_color = "#f39c12"
                                    else:
                                        icon = "🟡"
                                        border_color = "#f1c40f"
                                    
                                    with st.expander(f"{icon} **{name}** | Confidence: {conf} | Severity: {sev}", expanded=(sev in ["Critical", "High"])):
                                        st.markdown(f"""
                                        <div style='padding: 10px; background-color: {border_color}11; border-left: 4px solid {border_color}; border-radius: 5px;'>
                                            <p style='margin: 0;'><strong>Rule Engine Reasoning:</strong> {desc}</p>
                                        </div>
                                        """, unsafe_allow_html=True)

                            st.markdown("---")
                            
                            # ============================================
                            # SECTION 6: ViT MODEL ANALYSIS
                            # ============================================
                            
                            st.markdown("## 📸 Visual Pattern Recognition (ViT Model)")
                            st.caption("Computer vision-based classification using Vision Transformer neural network")
                            
                            vit_col1, vit_col2 = st.columns([1, 2])
                            
                            with vit_col1:
                                if plot_generated:
                                    st.image(vit_plot_path, caption="Resistance Curve Input", width='stretch')
                                else:
                                    st.warning("⚠️ Could not generate visualization.")
                                    
                            with vit_col2:
                                if vit_result:
                                    pred_class = vit_result['class']
                                    pred_conf = vit_result['confidence'] * 100
                                    
                                    st.markdown("### 🎯 Prediction Results (ViT + Gemini Ensemble)")
                                    
                                    metric_col1, metric_col2 = st.columns(2)
                                    with metric_col1:
                                        st.metric("Predicted Condition", pred_class)
                                    with metric_col2:
                                        st.metric("Ensemble Confidence", f"{pred_conf:.1f}%")
                                    
                                    st.markdown("---")
                                    
                                    # Show Breakdown if available
                                    details = vit_result.get("details", {})
                                    if details and details.get("gemini_probs"):
                                        with st.expander("📊 Detailed Ensemble Breakdown", expanded=True):
                                            st.caption("Comparison of ViT Model and Gemini Expert Analysis")
                                            
                                            # Prepare data for chart
                                            vit_probs = details.get("vit_probs", {})
                                            gemini_probs = details.get("gemini_probs", {})
                                            ensemble_scores = details.get("ensemble_scores", {})
                                            
                                            chart_data = []
                                            for cls, score in ensemble_scores.items():
                                                chart_data.append({
                                                    "Class": cls,
                                                    "ViT": vit_probs.get(cls, 0),
                                                    "Gemini": gemini_probs.get(cls, 0),
                                                    "Ensemble": score
                                                })
                                            
                                            if chart_data:
                                                st.bar_chart(
                                                    pd.DataFrame(chart_data).set_index("Class")[["ViT", "Gemini", "Ensemble"]]
                                                )
                                    
                                    if pred_class == "Healthy":
                                        st.success("✅ **Visual pattern matches healthy signature.**")
                                    else:
                                        st.warning(f"⚠️ **Visual pattern suggests: {pred_class}**")
                                        
                                elif not VIT_AVAILABLE:
                                    st.info("ℹ️ **ViT Model Unavailable**\n\nThe Vision Transformer model requires PyTorch.")
                                else:
                                    st.warning("⚠️ **Prediction failed.** Model may not be loaded correctly or input image is invalid.")

                            st.markdown("---")
                            
                            # ============================================
                            # SECTION 7: ADDITIONAL DIAGNOSTICS
                            # ============================================
                            
                            st.markdown("## 🔬 Additional Diagnostic Plots")
                            st.caption("Supplementary visualizations for detailed analysis")
                            
                            diag_col1, diag_col2 = st.columns(2)
                            
                            with diag_col1:
                                st.markdown("### Contact Velocity Profile")
                                st.caption("Derivative of travel - indicates mechanical performance")
                                fig_vel = create_velocity_plot(df)
                                st.plotly_chart(fig_vel, width='stretch')
                                
                            with diag_col2:
                                st.markdown("### Resistance Detail (Log Scale)")
                                st.caption("Logarithmic view reveals subtle resistance variations")
                                fig_res = create_resistance_zoom_plot(df)
                                st.plotly_chart(fig_res, width='stretch')

                            st.markdown("---")
                            st.markdown("---")
                            
                            # ============================================
                            # SECTION 8: RECOMMENDATIONS & PREDICTIONS
                            # ============================================
                            
                            st.markdown("## 🛠️ Recommendations & Future Fault Predictions")
                            st.caption("AI-generated maintenance actions and predictive failure analysis")
                            
                            # Maintenance Actions
                            if recommendations.get("maintenanceActions"):
                                st.markdown("### 🔧 Recommended Maintenance Actions")
                                for group in recommendations["maintenanceActions"]:
                                    priority = group.get("priority", "Priority")
                                    color = group.get("color", "#333")
                                    bg_color = group.get("bgColor", "#eee")
                                    
                                    st.markdown(f"""
                                    <div style='background-color: {bg_color}; padding: 10px; border-radius: 5px; border-left: 5px solid {color}; margin-bottom: 10px;'>
                                        <h4 style='color: {color}; margin: 0;'>{priority}</h4>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    for action in group.get("actions", []):
                                        with st.expander(f"**{action.get('title')}**"):
                                            st.markdown(f"**Justification:** {action.get('justification')}")
                                            st.markdown(f"**Timeline:** {action.get('timeline')}")
                                            if action.get("whatToLookFor"):
                                                st.markdown("**What to Look For:**")
                                                for item in action["whatToLookFor"]:
                                                    st.markdown(f"- {item}")

                            st.markdown("---")

                            # Future Faults
                            if recommendations.get("futureFaultsPdf"):
                                st.markdown("### 🔮 Future Fault Predictions")
                                
                                cols = st.columns(len(recommendations["futureFaultsPdf"]))
                                for idx, fault in enumerate(recommendations["futureFaultsPdf"]):
                                    with cols[idx % 4]: # Wrap around if many
                                        prob = fault.get("probability", 0)
                                        color = fault.get("color", "#333")
                                        
                                        st.markdown(f"""
                                        <div style='text-align: center; border: 1px solid #ddd; border-radius: 10px; padding: 15px; height: 100%;'>
                                            <h3 style='color: {color};'>{prob}%</h3>
                                            <p style='font-weight: bold;'>{fault.get('fault')}</p>
                                            <p style='font-size: 0.9em; color: #666;'>{fault.get('timeline')}</p>
                                            <hr>
                                            <p style='font-size: 0.8em; text-align: left;'>{fault.get('evidence')}</p>
                                        </div>
                                        """, unsafe_allow_html=True)

                            st.markdown("---")

                            # RUL Estimation
                            ai_verdict = full_report.get("aiVerdict", {})
                            rul_est = ai_verdict.get("rulEstimate")
                            uncertainty = ai_verdict.get("uncertainty")
                            
                            if rul_est:
                                st.markdown("### ⏳ Remaining Useful Life (RUL) Estimation")
                                st.caption("Estimated remaining operations based on current degradation")
                                
                                rul_col1, rul_col2 = st.columns(2)
                                with rul_col1:
                                    st.metric("RUL Estimate", rul_est, delta=uncertainty, delta_color="off")
                                with rul_col2:
                                    st.info(f"**Uncertainty:** {uncertainty}\n\nThis estimate considers contact wear, timing deviations, and coil health.")
                                
                                st.markdown("---")

                            # AI Strategic Advice
                            ai_advice = ai_verdict.get("aiAdvice", [])
                            if ai_advice:
                                st.markdown("## 🧠 AI Strategic Advisory")
                                st.caption("Expert recommendations based on engineering analysis")
                                
                                for advice in ai_advice:
                                    color = advice.get("color", "#333")
                                    title = advice.get("title", "Recommendation")
                                    desc = advice.get("description", "")
                                    impact = advice.get("expectedImpact", "")
                                    priority = advice.get("priority", "Medium")
                                    
                                    with st.expander(f"**{priority}**: {title}", expanded=(priority=="Critical")):
                                        st.markdown(f"""
                                        <div style='border-left: 5px solid {color}; padding-left: 15px;'>
                                            <p><strong>Rationale:</strong> {desc}</p>
                                            <p><strong>Expected Impact:</strong> {impact}</p>
                                        </div>
                                        """, unsafe_allow_html=True)
                                        
                                        effects = advice.get("effectAnalysis", {})
                                        if effects:
                                            e_col1, e_col2 = st.columns(2)
                                            with e_col1:
                                                if effects.get("shortTerm"):
                                                    st.markdown("**Short Term Benefits:**")
                                                    for item in effects["shortTerm"]:
                                                        st.markdown(f"- {item}")
                                            with e_col2:
                                                if effects.get("longTerm"):
                                                    st.markdown("**Long Term Benefits:**")
                                                    for item in effects["longTerm"]:
                                                        st.markdown(f"- {item}")
                                
                                st.markdown("---")
                            
                            # ============================================
                            # SECTION 9: DOWNLOAD REPORT
                            # ============================================
                            
                            st.markdown("## 📥 Export Analysis Report")
                            st.caption("Download complete JSON report with all metrics, phase data, and AI insights")
                            
                            col_download1, col_download2, col_download3 = st.columns([1, 2, 1])
                            
                            with col_download1:
                                st.markdown("")  # Spacer
                            
                            with col_download2:
                                st.download_button(
                                    label="📄 Download Full Report (JSON)",
                                    data=json.dumps(full_report, indent=2),
                                    file_name=f"dcrm_analysis_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
                                    mime="application/json",
                                    width='stretch'
                                )
                                st.caption("✅ Includes: KPIs, CBHI score, phase analysis, AI verdicts, rule engine results, and ViT predictions")
                            
                            with col_download3:
                                st.markdown("")  # Spacer


                            
                        except Exception as e:
                            st.error(f"Analysis failed: {str(e)}")
                            st.expander("Debug Info").write(content if 'content' in locals() else "No content")
                else:
                    st.info("Click 'Run Advanced Analysis' to start.")
                    with st.expander("Preview Raw Data"):
                        st.dataframe(df.head())

        except Exception as e:
            st.error(f"Error reading CSV file: {str(e)}")

if __name__ == "__main__":
    main()
