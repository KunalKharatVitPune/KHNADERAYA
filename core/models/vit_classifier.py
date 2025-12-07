# Previous Name: analysis/models/vit_classifier.py
import os
import matplotlib
matplotlib.use('Agg')  # Set backend to non-interactive
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import requests
import google.generativeai as genai
import json

# Parameters
IMG_HEIGHT = 224
IMG_WIDTH = 224

# Define classes (must match training - sorted alphabetically)
CLASSES = sorted([
    "Healthy",
    "Arcing_Contact_Misalignment",
    "Arcing_Contact_Wear",
    "Main Contact Misalignment",
    "main_contact_wear"
])

# Deployed ViT Model URL
DEPLOYED_VIT_URL = "http://143.110.244.235/predict"

def plot_resistance_for_vit(df, save_path="temp_vit_plot.png"):
    """
    Generates and saves the resistance plot in the format expected by the ViT model.
    Plots all three curves: Green (Resistance), Blue (Current), Red (Travel)
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot based on column availability
    # Green: Resistance, Blue: Current, Red: Travel
    if 'Resistance' in df.columns:
        ax.plot(df['Resistance'], color='green', label='Resistance')
    if 'Current' in df.columns:
        ax.plot(df['Current'], color='blue', label='Current')
    if 'Travel' in df.columns:
        ax.plot(df['Travel'], color='red', label='Travel')
        
    ax.legend()
    ax.set_title("DCRM Trace")
    ax.set_xlabel("Time (Samples)")
    ax.set_ylabel("Value")
    ax.grid(True)
    
    # Save to file
    plt.savefig(save_path)
    plt.close(fig)
    return True

def get_remote_vit_probabilities(image_path):
    """
    Get probability distribution for all classes from the deployed ViT API.
    Returns: dict of {class_name: probability}
    """
    try:
        if not os.path.exists(image_path):
            print(f"Error: Image file not found at {image_path}")
            return {}

        with open(image_path, "rb") as f:
            # Explicitly set the filename and MIME type
            files = {"file": (os.path.basename(image_path), f, "image/jpeg")}
            # Add headers
            headers = {"accept": "application/json"}
            
            print(f"Sending request to {DEPLOYED_VIT_URL}...")
            response = requests.post(DEPLOYED_VIT_URL, headers=headers, files=files, timeout=10)
            
        if response.status_code != 200:
            print(f"Error: API returned status code {response.status_code}")
            print(f"Response: {response.text}")
            return {}

        data = response.json()
        
        if "probabilities" in data:
            return data["probabilities"]
        else:
            print(f"Error: 'probabilities' key not found in response: {data}")
            return {}

    except requests.exceptions.RequestException as e:
        print(f"Request Error: {e}")
        return {}
    except Exception as e:
        print(f"Error processing image with remote API: {e}")
        return {}


def get_gemini_prediction(image, api_key=None):
    """
    Get Gemini's expert analysis of the DCRM trace.
    Returns: (probabilities_dict, error_message)
    """
    if not api_key:
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    
    if not api_key:
        return None, "API Key missing"
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')
        prompt = f"""
        System Role: Principal DCRM & Kinematic Analyst
        Role:
        You are an expert High-Voltage Circuit Breaker Diagnostician. Your task is to interpret Dynamic Contact Resistance (DCRM) traces to detect specific electrical and mechanical faults.
        
        The input image contains 3 line charts:
        - Green: Resistance profile
        - Blue: Current profile 
        - Red: Travel profile
        
        1. Diagnostic Heuristics & Defect Taxonomy
        Map the visual DCRM trace to ONLY the following defect types. Use the specific Visual Heuristics to confirm detection.
        
        Defect Type | Visual Heuristic (The "Hint") | Mechanical Significance (Root Cause)
        --- | --- | ---
        Main Contact Issue (Corrosion/Oxidation) | "The Significant Grass"<br>In the fully closed plateau, look for pronounced, erratic instability. <br>• Ignore: Uniform, low-amplitude fuzz (sensor noise).<br>• Flag: Jagged, irregular peaks/valleys with significant amplitude (e.g., > 15–20 μΩ variance). The trace looks like a "rough rocky road," not just a "gravel path." | Surface Pathology: The Silver (Ag) plating is compromised (fretting corrosion) or heavy oxidation has occurred. The current path is constantly shifting through microscopic non-conductive spots.
        Arcing Contact Wear | "Big Spikes & Short Wipe"<br>Resistance spikes are frequent and significantly large (high amplitude). Crucially, the duration of the arcing zone (the time between first touch and main contact touch) is noticeably shorter than expected. | Ablation: The Tungsten-Copper (W-Cu) tips are heavily eroded. The contact length has physically diminished, risking failure to commutate current during opening.
        Misalignment (Main) | "The Struggle to Settle"<br>There are significant, high-amplitude peaks just before the trace tries to settle into the stable plateau. These are not bounces; they are "struggles" to mate that persist longer than 3-5ms. | Mechanical Centering: The moving contact pin is hitting the side or edge of the stationary rosette fingers before forcing its way in. Caused by loose nuts, kinematic play, or guide ring failure.
        Misalignment (Arcing) | "Rough Entry"<br>Erratic resistance spikes occurring specifically during the initial entry (commutation), well before the main contacts engage. | Tip Eccentricity: The arcing pin is not entering the nozzle concentrically. It is scraping the nozzle throat or hitting the side, indicating a bent rod or skewed interrupter.
        Slow Mechanism | "Stretched Time"<br>The entire resistance profile is elongated along the X-axis. Events happen later than normal. | Energy Starvation: Low spring charge, hydraulic pressure loss, or high friction due to hardened grease in the linkage.
        
        2. Analysis Logic (The "Signal-to-Noise" Filter)
        Before declaring a defect, run these logic checks:
        The "Noise Floor" Test (For Main Contacts):
        Is the plateau variance uniform and small (< 10 μΩ)? -> Classify as Healthy (Sensor/Manufacturing artifact).
        Is the variance erratic, jagged, and large (> 15 μΩ)? -> Classify as Corrosion/Oxidation.
        The "Duration" Test (For Misalignment):
        Are the pre-plateau peaks < 2ms? -> Ignore (Benign Bounce).
        Do the peaks persist > 3-5ms before settling? -> Classify as Misalignment.
        The "Combination" Check:
        Does the trace show both "Rough Entry" AND "Stretched Time"? -> Report Both (Misalignment + Slow Mechanism).
        
        3. Output Structure
        You must return a JSON object containing the probability (confidence score between 0.0 and 1.0) for EACH of the following classes. The sum of probabilities should ideally be close to 1.0.
        
        Classes: {CLASSES}
        
        Example Output Format:
        {{
            "Healthy": 0.1,
            "Arcing_Contact_Misalignment": 0.05,
            "Arcing_Contact_Wear": 0.8,
            "Main Contact Misalignment": 0.02,
            "main_contact_wear": 0.03
        }}
        
        RETURN ONLY THE JSON OBJECT. NO MARKDOWN. NO EXPLANATION.
        """
        response = model.generate_content([prompt, image])
        
        # Clean response to ensure valid JSON
        text = response.text.strip()
        # Remove markdown code blocks if present
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        
        probs = json.loads(text)
        return probs, None
    except Exception as e:
        return None, str(e)


def predict_dcrm_image(image_path, model_path=None, api_key=None):
    """
    Predicts the class of the DCRM image using Deployed ViT + Gemini Ensemble.
    Returns: (predicted_class, confidence_score, details_dict)
    
    The details_dict contains:
    - vit_probs: Dictionary of ViT probabilities for each class (from deployed API)
    - gemini_probs: Dictionary of Gemini probabilities for each class
    - ensemble_scores: Dictionary of combined scores for each class
    """
    try:
        # Get API key from environment if not provided
        if not api_key:
            api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        
        # Load image for Gemini
        image = Image.open(image_path).convert('RGB')
        
        # 1. ViT Prediction (Remote)
        vit_probs = get_remote_vit_probabilities(image_path)
        
        if not vit_probs:
            print("Warning: Failed to get ViT probabilities from remote API.")
            # We can continue if we want to rely on Gemini, or return failure.
            # For now, let's continue but note the failure.
        
        # 2. Gemini Prediction
        gemini_probs, error = get_gemini_prediction(image, api_key)
        
        # Ensemble Logic
        ensemble_scores = {}
        
        # Initialize with 0.0 for all classes
        for cls in CLASSES:
            ensemble_scores[cls] = 0.0
            
        # Add ViT scores
        if vit_probs:
            for cls, prob in vit_probs.items():
                if cls in ensemble_scores:
                    ensemble_scores[cls] += prob
                    
        # Add Gemini scores
        if gemini_probs:
            for cls, prob in gemini_probs.items():
                if cls in ensemble_scores:
                    ensemble_scores[cls] += prob
                    
        # If both failed, return error
        if not vit_probs and not gemini_probs:
            return None, 0.0, {}

        # Sort classes by score (descending)
        sorted_classes = sorted(ensemble_scores.items(), key=lambda item: item[1], reverse=True)
        
        best_class = sorted_classes[0][0]
        best_score = sorted_classes[0][1]
        
        # Conditional Logic - Use fallback only when Healthy is at top with low score
        if best_score < 1.0 and best_class == "Healthy" and best_score < 0.8:
            if len(sorted_classes) > 1:
                best_class = sorted_classes[1][0]
                best_score = sorted_classes[1][1]
        
        # Normalize score
        # If we had both models, max score is 2.0. If only one, max is 1.0.
        divisor = 0.0
        if vit_probs: divisor += 1.0
        if gemini_probs: divisor += 1.0
        
        if divisor > 0:
            normalized_confidence = best_score / divisor
        else:
            normalized_confidence = 0.0
        
        return best_class, normalized_confidence, {
            "vit_probs": vit_probs,
            "gemini_probs": gemini_probs,
            "ensemble_scores": ensemble_scores
        }
            
    except Exception as e:
        print(f"Error in predict_dcrm_image: {e}")
        import traceback
        traceback.print_exc()
        return None, 0.0, {}
