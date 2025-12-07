# Previous Name: analysis/agents/arcing_segmentation.py
import pandas as pd
import json
import matplotlib.pyplot as plt
import io
import base64
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

def get_arcing_prompt(data_str):
    return f"""
    I have extracted data from a DCRM (Dynamic Contact Resistance Measurement) graph.
    Data (Sampled): {data_str}
    
    The columns are:
    - 'Time_ms': Time in milliseconds.
    - 'Resistance': Contact resistance in micro-ohms (µΩ).
    - 'Travel': Mechanical position/displacement (mm).
    
    I have also provided the image of the graph.
    
    === TASK: IDENTIFY ARCING CONTACT EVENTS (T0-T4) ===
    
    Your goal is to identify 5 specific timepoints (T0, T1, T2, T3, T4) that define the arcing contact behavior.
    
    **T0 (Breaker Closed)**:
    - State: Main Contacts are fully closed.
    - Signature: Resistance is LOW and STABLE (DLRO value, typically 30-60 µΩ).
    - Time: t=0 or very early.
    
    **T1 (Motion Starts)**:
    - Event: The breaker mechanism begins to move.
    - Signature: First significant deviation in the TRAVEL curve (Red). Resistance might rise slightly due to vibration.
    
    **T2 (Main Contact Separation)**:
    - Event: The Main Contacts (Silver) separate, forcing current to the Arcing Contacts (Tungsten).
    - **VISUAL SIGNATURE**: Look for a sharp "STEP" increase in Resistance.
    - **DATA SIGNATURE**: Resistance jumps from Low (~40 µΩ) to Medium (~150-300 µΩ).
    - This is the START of the "Arcing Plateau".
    
    **T3 (Arcing Contact Wipe Zone)**:
    - Event: Only Arcing Contacts are touching.
    - Signature: The "Plateau" phase between T2 and T4.
    - Healthy: Flat plateau. Unhealthy: Noisy/Spiky.
    - **Action**: Return the approximate MIDPOINT time of this plateau.
    
    **T4 (Arcing Contact Separation)**:
    - Event: The Arcing Contacts part.
    - **VISUAL SIGNATURE**: Resistance shoots VERTICALLY to INFINITY (Open Circuit).
    - **DATA SIGNATURE**: Resistance jumps from Medium (~200 µΩ) to High/Infinity (>2000 µΩ).
    - This is the END of the electrical connection.
    
    === IMPORTANT: JSON FORMATTING ===
    - You must return valid JSON.
    - Do NOT include markdown formatting (like ```json).
    - If you cannot find a point with certainty, return null.
    
    === OUTPUT FORMAT (Strict JSON) ===
    {{
      "events": {{
        "T0_breaker_closed": float,
        "T1_motion_start": float,
        "T2_main_separation": float,
        "T3_arcing_plateau_mid": float,
        "T4_arcing_separation": float
      }},
      "confidence": "High"|"Medium"|"Low",
      "reasoning": "Brief explanation of how you found T2 and T4"
    }}
    """

def identify_arcing_events(df, llm):
    """
    Uses the LLM to identify T0-T4 events from the dataframe.
    """
    try:
        # 1. Prepare Data Sample (downsample for token limit)
        # We focus on the first 200ms where action happens
        df_subset = df[df['Time_ms'] < 200].copy()
        step = max(1, len(df_subset) // 100)
        data_str = df_subset.iloc[::step][['Time_ms', 'Resistance', 'Travel']].to_string(index=False)
        
        # 2. Generate Plot Image
        plt.figure(figsize=(10, 6))
        
        # Plot Travel (Red)
        ax1 = plt.gca()
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('Travel (mm)', color='tab:red')
        ax1.plot(df_subset['Time_ms'], df_subset['Travel'], color='tab:red', label='Travel')
        ax1.tick_params(axis='y', labelcolor='tab:red')
        
        # Plot Resistance (Green) on secondary axis
        ax2 = ax1.twinx()
        ax2.set_ylabel('Resistance (µΩ)', color='tab:green')
        ax2.plot(df_subset['Time_ms'], df_subset['Resistance'], color='tab:green', label='Resistance')
        ax2.tick_params(axis='y', labelcolor='tab:green')
        # Limit Resistance view to see the "step" (ignore infinity spike)
        ax2.set_ylim(0, 2000) 
        
        plt.title("DCRM Signature: Identify T0, T1, T2, T4")
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image_data = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        # 3. Call LLM
        prompt = get_arcing_prompt(data_str)
        
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}}
            ]
        )
        
        response = llm.invoke([message])
        content = response.content.strip()
        
        # Clean code blocks if present
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].strip()
            
        result = json.loads(content)
        return result
        
    except Exception as e:
        return {"error": str(e)}
