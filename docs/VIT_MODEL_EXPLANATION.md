# ViT Model in DCRM Pipeline - Complete Explanation

## What is `vitResult`?

The `vitResult` is the output from a **Vision Transformer (ViT) + Gemini AI Ensemble Model** that analyzes the DCRM resistance plot image to classify circuit breaker defects.

---

## 📊 Complete Flow (Step-by-Step)

### **Step 1: Generate Resistance Plot**
**File**: `core/models/vit_classifier.py` → `plot_resistance_for_vit()`

```python
# Creates a plot with 3 lines:
# - Green line: Resistance profile
# - Blue line: Current profile  
# - Red line: Travel profile

# Saves as temporary PNG file: temp_vit_plot_{phase}_{uuid}.png
```

**Example**: `temp_vit_plot_r_a3f8d2b1.png`

---

### **Step 2: ViT Model Analysis** (Remote API)
**File**: `core/models/vit_classifier.py` → `get_remote_vit_probabilities()`

```python
# Sends image to deployed ViT model API
DEPLOYED_VIT_URL = "http://143.110.244.235/predict"

# ViT is trained on DCRM images to detect 5 defect classes:
CLASSES = [
    "Healthy",
    "Arcing_Contact_Misalignment",
    "Arcing_Contact_Wear",
    "Main Contact Misalignment",
    "main_contact_wear"
]

# Returns probability distribution for each class
vit_probs = {
    "Healthy": 0.507,
    "Arcing_Contact_Misalignment": 0.120,
    "Arcing_Contact_Wear": 0.044,
    "Main Contact Misalignment": 0.142,
    "main_contact_wear": 0.186
}
```

**How ViT Works**:
- ViT (Vision Transformer) is a deep learning model trained on DCRM plot images
- It learned visual patterns from thousands of circuit breaker test plots
- Analyzes waveform shapes, spikes, plateaus, and transitions
- Outputs probability for each defect type

---

### **Step 3: Gemini AI Analysis**
**File**: `core/models/vit_classifier.py` → `get_gemini_prediction()`

```python
# Sends same image to Google Gemini 2.0 Flash
# Uses expert prompt with diagnostic heuristics:

Diagnostic Rules:
1. "The Significant Grass" → Main Contact Corrosion
   - Jagged, irregular resistance plateau (> 15-20μΩ variance)
   
2. "Big Spikes & Short Wipe" → Arcing Contact Wear
   - Large amplitude spikes, shortened arcing zone
   
3. "The Struggle to Settle" → Main Misalignment
   - High-amplitude peaks before plateau (> 3-5ms)
   
4. "Rough Entry" → Arcing Misalignment
   - Erratic spikes during initial entry
   
5. "Stretched Time" → Slow Mechanism
   - Elongated resistance profile on X-axis

# Returns probability distribution
gemini_probs = {
    "Healthy": 0.05,
    "Arcing_Contact_Misalignment": 0.02,
    "Arcing_Contact_Wear": 0.01,
    "Main Contact Misalignment": 0.02,
    "main_contact_wear": 0.90  # High confidence!
}
```

---

### **Step 4: Ensemble Prediction**
**File**: `core/models/vit_classifier.py` → `predict_dcrm_image()`

```python
# Combines ViT + Gemini predictions
# ensemble_score = vit_prob + gemini_prob

ensemble_scores = {
    "Healthy": 0.507 + 0.05 = 0.557,
    "Arcing_Contact_Misalignment": 0.120 + 0.02 = 0.140,
    "Arcing_Contact_Wear": 0.044 + 0.01 = 0.054,
    "Main Contact Misalignment": 0.142 + 0.02 = 0.162,
    "main_contact_wear": 0.186 + 0.90 = 1.086  # ✅ HIGHEST!
}

# Selects class with highest ensemble score
predicted_class = "main_contact_wear"
confidence = 0.543  # Normalized confidence
```

---

### **Step 5: Integration into Pipeline**
**File**: `apps/flask_server.py` → `process_single_phase_csv()`

```python
# Lines 155-183
vit_result = None
vit_plot_path = f"temp_vit_plot_{phase_name}_{uuid.uuid4().hex[:8]}.png"

# Generate plot
if plot_resistance_for_vit(df, vit_plot_path):
    # Get prediction
    vit_class, vit_conf, vit_details = predict_dcrm_image(vit_plot_path, api_key=api_key)
    
    vit_result = {
        "class": vit_class,           # "main_contact_wear"
        "confidence": vit_conf,       # 0.5429375439882278
        "details": vit_details        # Full breakdown below
    }

# Cleanup temp file
os.remove(vit_plot_path)
```

---

## 📦 vitResult Structure Breakdown

```json
{
  "class": "main_contact_wear",           // ✅ FINAL PREDICTION
  "confidence": 0.5429375439882278,       // ✅ NORMALIZED CONFIDENCE
  "details": {
    "vit_probs": {                        // 🤖 Vision Transformer probabilities
      "Healthy": 0.5076556205749512,
      "Arcing_Contact_Misalignment": 0.12034504860639572,
      "Arcing_Contact_Wear": 0.04370640590786934,
      "Main Contact Misalignment": 0.1424178034067154,
      "main_contact_wear": 0.1858750879764557
    },
    "gemini_probs": {                     // 🧠 Gemini AI probabilities
      "Healthy": 0.05,
      "Arcing_Contact_Misalignment": 0.02,
      "Arcing_Contact_Wear": 0.01,
      "Main Contact Misalignment": 0.02,
      "main_contact_wear": 0.9            // Gemini is very confident!
    },
    "ensemble_scores": {                  // 🎯 COMBINED SCORES
      "Healthy": 0.5576556205749512,
      "Arcing_Contact_Misalignment": 0.1403450486063957,
      "Arcing_Contact_Wear": 0.05370640590786934,
      "Main Contact Misalignment": 0.16241780340671538,
      "main_contact_wear": 1.0858750879764556  // ✅ HIGHEST → WINNER
    }
  }
}
```

---

## 🔍 Why Two Models?

| Model | Strengths | Weaknesses |
|-------|-----------|------------|
| **ViT** | - Trained on real DCRM data<br>- Fast inference<br>- Consistent | - May overfit to training data<br>- Limited to visual patterns |
| **Gemini** | - Expert reasoning<br>- Contextual understanding<br>- Adapts to new cases | - May hallucinate<br>- Slower<br>- Requires API calls |
| **Ensemble** | ✅ **Best of both worlds**<br>- ViT provides baseline<br>- Gemini adds expertise | - Slightly higher computational cost |

---

## 🎯 How It's Used in the Pipeline

The `vitResult` is:

1. **Generated** in `flask_server.py` (lines 155-183)
2. **Passed to** `report_generator.py` 
3. **Included in** final JSON output under each phase (r, y, b)
4. **Referenced** in fault summaries for LLM context

**Example Usage**:
```python
# In report_generator.py
if vit_result:
    faults_summary += f"\nViT Model Prediction:\n- Class: {vit_result.get('class', 'Unknown')}\n- Confidence: {vit_result.get('confidence', 0)*100:.2f}%\n"
```

---

## 📊 Visual Flow Diagram

```
Input CSV Data
      ↓
Extract Resistance, Current, Travel
      ↓
Generate Plot (matplotlib)
      ↓  
  temp_vit_plot.png
      ↓
      ├──→ [ViT API]      → vit_probs
      └──→ [Gemini AI]    → gemini_probs
            ↓
      Ensemble Combination
            ↓
      ensemble_scores
            ↓
   Select MAX score → predicted_class
            ↓
      vitResult JSON
            ↓
  Included in final report
```

---

## 🛠️ Configuration

**ViT API Endpoint**:
```python
DEPLOYED_VIT_URL = "http://143.110.244.235/predict"
```

**Gemini Model**:
```python
model = genai.GenerativeModel('gemini-2.0-flash')
```

**API Key** (from environment):
```python
GOOGLE_API_KEY  # Main key
GOOGLE_API_KEY_1  # For R phase
GOOGLE_API_KEY_2  # For Y phase  
GOOGLE_API_KEY_3  # For B phase
```

---

## 🚨 Error Handling

If ViT or Gemini fails:
```python
if not vit_result:
    # Pipeline continues without ViT analysis
    # Other components (Rule Engine, AI Agent) still work
    print("ViT prediction unavailable, continuing with other analyses...")
```

The pipeline is **resilient** - if ViT fails, analysis still completes using Rule Engine + AI Agent.

---

## 📝 Summary

**vitResult provides**:
- ✅ Image-based defect classification
- ✅ Visual pattern recognition (ViT)
- ✅ Expert reasoning (Gemini)
- ✅ Ensemble confidence scoring
- ✅ Detailed probability breakdown
- ✅ Complements KPI-based and time-series analysis

It's a **3rd independent diagnostic method** alongside:
1. Rule Engine (deterministic thresholds)
2. AI Agent (LLM-based fault detection)
3. **ViT Model (image classification)** ← This one!

All three methods are combined to provide comprehensive, multi-faceted circuit breaker diagnostics.
