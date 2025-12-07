# Previous Name: analysis/engines/advanced_rule_engine.py
import numpy as np
import pandas as pd
import json
import sys
from scipy.signal import savgol_filter, find_peaks, welch
from scipy.stats import skew, kurtosis
from scipy.integrate import simpson

# Set UTF-8 encoding for console output
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

class AdvancedDCRMEngine:
    """
    Top-Notch Advanced Rule-Based DCRM Engine
    =========================================
    Combines Physics-Based Signal Processing (Scipy) with Expert Heuristic Logic.
    
    Features:
    - 12-Class Defect Detection (Primary + Secondary)
    - Evidence-Based Scoring (0-100% Confidence)
    - Advanced Signal Processing:
        * Savitzky-Golay Filtering (Noise reduction without edge blurring)
        * FFT (Mechanical Chatter detection)
        * Peak Finding (Spike counting, Bounce detection)
        * Derivative Analysis (Jerk/Stutter detection)
        * Energy Integration (Arcing Ablation)
    """

    def __init__(self):
        # --- CONFIGURATION & THRESHOLDS ---
        
        # 1. Signal Processing Config
        self.SAVGOL_WINDOW = 11       # Window length for smoothing (must be odd)
        self.SAVGOL_POLYORDER = 2     # Polynomial order
        
        # 2. Physics Thresholds (Strict - Industry Standard)
        self.R_OPEN_THRESHOLD = 1_000_000 # µΩ
        self.R_MAIN_MAX_HEALTHY = 50.0    # µΩ
        self.R_MAIN_WARNING = 80.0        # µΩ
        self.R_MAIN_CRITICAL = 150.0      # µΩ
        
        # 3. Wear & Arcing Thresholds
        self.ARCING_SPIKE_CRITICAL = 8000 # µΩ
        self.ARCING_SPIKE_SEVERE = 5000   # µΩ
        self.ARCING_ENERGY_CRITICAL = 2000.0 # Joules
        
        # 4. Mechanical Thresholds
        self.BOUNCE_PROMINENCE = 500      # µΩ
        self.JERK_THRESHOLD = 500.0       # µΩ/ms^2
        self.FFT_CHATTER_POWER_THRESHOLD = 100.0

    def analyze(self, df: pd.DataFrame, segments: dict, kpis: dict = None) -> dict:
        """
        Main entry point for analysis.
        
        Args:
            df: DataFrame with 'Resistance' column.
            segments: Dictionary containing phase start/end indices.
                      Expected keys: 'phase2_start', 'phase2_end', 'phase3_start', 
                      'phase3_end', 'phase4_start', 'phase4_end'.
            kpis: Optional dictionary of Key Performance Indicators.
        """
        if kpis is None: kpis = {}
            
        # 1. Standardize Input
        df_std = self._standardize_input(df)
        resistance = df_std['Resistance'].values
        
        # 2. Signal Preprocessing (Scipy)
        try:
            resistance_smooth = savgol_filter(resistance, self.SAVGOL_WINDOW, self.SAVGOL_POLYORDER)
        except Exception:
            resistance_smooth = resistance
            
        # Derivatives
        res_velocity = np.gradient(resistance_smooth)
        res_acceleration = np.gradient(res_velocity)
        
        # 3. Advanced Feature Extraction (Using provided segments)
        features = self._extract_advanced_features(
            resistance, resistance_smooth, res_velocity, res_acceleration, segments, kpis
        )
        
        # 4. Heuristic Defect Detection (12 Classes)
        defects = self._detect_defects(features, kpis)
        
        # 5. Construct Report
        report = self._build_report(features, defects)
        
        return report

    def _standardize_input(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'Resistance' not in df.columns:
            cols = [c for c in df.columns if 'res' in c.lower() or 'uohm' in c.lower()]
            if cols: df = df.rename(columns={cols[0]: 'Resistance'})
            else:
                non_time = [c for c in df.columns if not c.lower().startswith('t')]
                if non_time: df = df.rename(columns={non_time[0]: 'Resistance'})
                else: raise ValueError("Could not identify Resistance column")
        
        if df.shape[1] > 100 and df.shape[0] < 10:
             vals = df.iloc[0].values
             return pd.DataFrame({'Resistance': vals})
             
        return df[['Resistance']].reset_index(drop=True)

    def _extract_advanced_features(self, r_raw, r_smooth, r_vel, r_acc, seg, kpis):
        """
        Extracts comprehensive features for heuristic scoring.
        """
        features = {}
        if not seg['valid']:
            features['valid_data'] = False
            return features
        features['valid_data'] = True
        
        # --- A. MAIN CONTACT (Phase 3) ---
        p3_slice = slice(seg['phase3_start'], seg['phase3_end'])
        r_p3 = r_raw[p3_slice]
        
        if len(r_p3) > 0:
            features['main_mean'] = float(np.mean(r_p3))
            features['main_std'] = float(np.std(r_p3))
            features['main_min'] = float(np.min(r_p3))
            features['main_max'] = float(np.max(r_p3))
            features['main_range'] = float(features['main_max'] - features['main_min'])
            
            # Detrend for roughness analysis
            x = np.arange(len(r_p3))
            p = np.polyfit(x, r_p3, 1)
            detrended = r_p3 - np.polyval(p, x)
            
            features['roughness_rms'] = float(np.sqrt(np.mean(detrended**2)))
            features['roughness_skew'] = float(skew(detrended)) if len(detrended) > 2 else 0
            features['roughness_kurtosis'] = float(kurtosis(detrended)) if len(detrended) > 2 else 0
            
            # FFT for Chatter (50-300Hz)
            if len(detrended) > 32:
                freqs, psd = welch(detrended, fs=1000)
                chatter_band = (freqs >= 50) & (freqs <= 300)
                features['chatter_power'] = float(simpson(psd[chatter_band], freqs[chatter_band]))
            else:
                features['chatter_power'] = 0.0
                
            # Telegraph Noise (Square Jumps)
            diffs = np.abs(np.diff(r_p3))
            features['telegraph_jumps'] = int(np.sum(diffs > 120)) # Jump > 120uOhm
            
            # Shelf Detection (Histogram)
            hist, _ = np.histogram(r_p3, bins=10)
            features['num_shelves'] = int(np.sum(hist > len(r_p3)*0.1)) # Bins with >10% data
            
        else:
            features.update({'main_mean': 9999, 'roughness_rms': 0, 'chatter_power': 0, 'telegraph_jumps': 0})

        # --- B. ARCING ZONES (Phase 2 & 4) ---
        test_current = kpis.get('Test Current (A)', 100.0)
        
        # Closing Arc
        p2_slice = slice(seg['phase2_start'], seg['phase2_end'])
        r_p2 = r_raw[p2_slice]
        features['closing_arc_duration'] = len(r_p2)
        if len(r_p2) > 0:
            power_p2 = (test_current ** 2) * r_p2 * 1e-6
            features['closing_arc_energy_joules'] = float(np.sum(power_p2) * 0.001)
            features['closing_critical_spikes'] = int(np.sum(r_p2 > self.ARCING_SPIKE_CRITICAL))
            features['closing_severe_spikes'] = int(np.sum(r_p2 > self.ARCING_SPIKE_SEVERE))
        else:
            features.update({'closing_arc_energy_joules': 0, 'closing_critical_spikes': 0, 'closing_severe_spikes': 0})

        # Opening Arc
        p4_slice = slice(seg['phase4_start'], seg['phase4_end'])
        r_p4 = r_raw[p4_slice]
        features['opening_arc_duration'] = len(r_p4)
        if len(r_p4) > 0:
            power_p4 = (test_current ** 2) * r_p4 * 1e-6
            features['opening_arc_energy_joules'] = float(np.sum(power_p4) * 0.001)
            features['opening_critical_spikes'] = int(np.sum(r_p4 > self.ARCING_SPIKE_CRITICAL))
            features['opening_severe_spikes'] = int(np.sum(r_p4 > self.ARCING_SPIKE_SEVERE))
            
            # Bounce Detection (Find Peaks)
            peaks, _ = find_peaks(r_p4, prominence=self.BOUNCE_PROMINENCE, distance=5)
            features['num_bounces'] = len(peaks)
            
            # Telegraph in Arcing
            features['arcing_telegraph'] = int(np.sum(np.abs(np.diff(r_p4)) > 400))
        else:
            features.update({'opening_arc_energy_joules': 0, 'opening_critical_spikes': 0, 'opening_severe_spikes': 0, 'num_bounces': 0, 'arcing_telegraph': 0})
            
        # --- C. KINEMATICS ---
        features['dur_closing'] = len(r_p2)
        features['dur_opening'] = len(r_p4)
        features['asymmetry_ratio'] = float(features['dur_opening'] / max(1, features['dur_closing']))
        
        acc_p3 = r_acc[p3_slice] if len(r_p3) > 0 else []
        features['max_micro_jerk'] = float(np.max(np.abs(acc_p3))) if len(acc_p3) > 0 else 0.0

        return features

    def _detect_defects(self, f, kpis):
        """
        Applies Heuristic Scoring Logic for 12 Defect Classes.
        Returns list of defects with 'Confidence' and 'Evidence'.
        """
        defects = []
        if not f['valid_data']: return defects
        
        # --- CLASS 1: HEALTHY (Implicit - if no defects found) ---
        
        # --- CLASS 2: MAIN CONTACT WEAR ---
        score = 0
        evidence = []
        if f['main_mean'] > self.R_MAIN_CRITICAL:
            score += 50; evidence.append(f"Critical Resistance ({f['main_mean']:.1f} µΩ)")
        elif f['main_mean'] > self.R_MAIN_WARNING:
            score += 30; evidence.append(f"Elevated Resistance ({f['main_mean']:.1f} µΩ)")
            
        if f['roughness_rms'] > 25:
            score += 25; evidence.append(f"Severe Surface Roughness (RMS {f['roughness_rms']:.1f})")
        elif f['roughness_rms'] > 15:
            score += 15; evidence.append(f"Moderate Roughness (RMS {f['roughness_rms']:.1f})")
            
        if f['roughness_skew'] > 1.5:
            score += 10; evidence.append("Positive Skew indicates pitting")
            
        if score > 40:
            defects.append(self._make_defect("Main Contact Wear", score, evidence))
            
        # --- CLASS 3: ARCING CONTACT WEAR ---
        score = 0
        evidence = []
        total_spikes = f['closing_critical_spikes'] + f['opening_critical_spikes']
        total_energy = f['closing_arc_energy_joules'] + f['opening_arc_energy_joules']
        
        if total_spikes >= 4:
            score += 50; evidence.append(f"{total_spikes} Critical Arc Flashes (>8000µΩ)")
        elif f['closing_severe_spikes'] + f['opening_severe_spikes'] >= 3:
            score += 40; evidence.append("Multiple Severe Spikes (>5000µΩ)")
            
        if total_energy > self.ARCING_ENERGY_CRITICAL:
            score += 30; evidence.append(f"Critical Arc Energy ({total_energy:.1f} J)")
            
        if score > 40:
            defects.append(self._make_defect("Arcing Contact Wear", score, evidence))
            
        # --- CLASS 4: MAIN CONTACT MISALIGNMENT ---
        score = 0
        evidence = []
        if f['telegraph_jumps'] >= 6:
            score += 45; evidence.append(f"Telegraph Pattern: {f['telegraph_jumps']} square jumps")
        elif f['telegraph_jumps'] >= 3:
            score += 25; evidence.append(f"Partial Telegraph: {f['telegraph_jumps']} jumps")
            
        if f['num_shelves'] >= 3:
            score += 20; evidence.append(f"Stepped Shelves: {f['num_shelves']} plateaus")
            
        if f['main_std'] > 70:
            score += 15; evidence.append(f"High Instability (Std {f['main_std']:.1f})")
            
        if score > 40:
            defects.append(self._make_defect("Main Contact Misalignment", score, evidence))
            
        # --- CLASS 5: ARCING CONTACT MISALIGNMENT ---
        score = 0
        evidence = []
        if f['asymmetry_ratio'] > 2.2:
            score += 35; evidence.append(f"Severe Asymmetry (Opening {f['asymmetry_ratio']:.1f}x Closing)")
        elif f['asymmetry_ratio'] > 1.6:
            score += 20; evidence.append(f"Moderate Asymmetry ({f['asymmetry_ratio']:.1f}x)")
            
        if f['num_bounces'] >= 5:
            score += 30; evidence.append(f"Mechanical Oscillation: {f['num_bounces']} bounces")
        elif f['num_bounces'] >= 3:
            score += 15; evidence.append(f"{f['num_bounces']} bounces detected")
            
        if f['arcing_telegraph'] > 10:
            score += 15; evidence.append("High-freq telegraph in arcing zone")
            
        if score > 40:
            defects.append(self._make_defect("Arcing Contact Misalignment", score, evidence))
            
        # --- CLASS 6: OPERATING MECHANISM (Timing) ---
        # Requires KPIs
        score = 0
        evidence = []
        c_time = kpis.get('Closing Time (ms)')
        o_time = kpis.get('Opening Time (ms)')
        
        if c_time and (c_time > 120 or c_time < 64):
            score += 40; evidence.append(f"Closing Time Deviation ({c_time}ms)")
        if o_time and (o_time > 48 or o_time < 24):
            score += 40; evidence.append(f"Opening Time Deviation ({o_time}ms)")
            
        if score > 40:
            defects.append(self._make_defect("Operating Mechanism Malfunction", score, evidence))
            
        # --- CLASS 7: DAMPING SYSTEM FAULT ---
        score = 0
        evidence = []
        if f['num_bounces'] > 7:
            score += 80; evidence.append(f"Excessive Bouncing ({f['num_bounces']} peaks) - Damper failure")
        elif f['num_bounces'] > 5:
            score += 50; evidence.append(f"High Bouncing ({f['num_bounces']} peaks)")
            
        if score > 40:
            defects.append(self._make_defect("Damping System Fault", score, evidence))
            
        # --- CLASS 9: LINKAGE/ROD OBSTRUCTION ---
        score = 0
        evidence = []
        if f['max_micro_jerk'] > self.JERK_THRESHOLD:
            score += 60; evidence.append(f"High Kinematic Jerk ({f['max_micro_jerk']:.1f}) - Stick-slip friction")
            
        if score > 40:
            defects.append(self._make_defect("Linkage/Rod Obstruction", score, evidence))
            
        # --- CLASS 10: FIXED CONTACT DAMAGE ---
        score = 0
        evidence = []
        dlro = kpis.get('DLRO Value (µΩ)')
        if dlro and dlro > 80 and f['roughness_rms'] < 15:
            # High resistance but smooth curve = Fixed contact issue
            score += 85; evidence.append(f"High DLRO ({dlro}µΩ) with Smooth Curve (Fixed Contact)")
            
        if score > 40:
            defects.append(self._make_defect("Fixed Contact Damage", score, evidence))
            
        # --- CLASS 11/12: COIL DAMAGE ---
        # Simple threshold checks
        cc = kpis.get('Peak Close Coil Current (A)')
        if cc and cc < 2.0:
            defects.append(self._make_defect("Close Coil Damage", 95, [f"Current {cc}A < 2A"]))
            
        tc1 = kpis.get('Peak Trip Coil 1 Current (A)')
        tc2 = kpis.get('Peak Trip Coil 2 Current (A)')
        if tc1 and tc2 and tc1 < 2.0 and tc2 < 2.0:
            defects.append(self._make_defect("Trip Coil Damage", 95, [f"Both Coils Failed (TC1:{tc1}A, TC2:{tc2}A)"]))

        return defects

    def _make_defect(self, name, score, evidence):
        return {
            "defect_name": name,
            "Confidence": f"{min(99.9, score):.1f} %",
            "Severity": "High" if score > 70 else "Medium",
            "description": "; ".join(evidence)
        }

    def _build_report(self, features, defects):
        """Constructs the final JSON report."""
        
        # Calculate Overall Health Score
        health_score = 100
        for d in defects:
            sev = d['Severity']
            if sev == 'High': health_score -= 30
            elif sev == 'Medium': health_score -= 15
            
        health_score = max(0, health_score)
        status = "Healthy"
        if health_score < 50: status = "Critical"
        elif health_score < 80: status = "Warning"
        
        # Sort defects by confidence
        defects.sort(key=lambda x: float(x['Confidence'].replace('%','')), reverse=True)
        
        return {
            "Fault_Detection": defects,
            "advanced_analysis": {
                "health_score": health_score,
                "status": status,
                "physics_metrics": {
                    "main_contact_resistance_uohm": round(features.get('main_mean', 0), 2),
                    "surface_roughness_rms": round(features.get('roughness_rms', 0), 2),
                    "arc_energy_joules": round(features.get('closing_arc_energy_joules', 0) + features.get('opening_arc_energy_joules', 0), 2),
                    "mechanical_chatter_power": round(features.get('chatter_power', 0), 2),
                    "kinematic_jerk_index": round(features.get('max_micro_jerk', 0), 2),
                    "telegraph_jumps": features.get('telegraph_jumps', 0),
                    "bounces": features.get('num_bounces', 0)
                }
            }
        }

if __name__ == "__main__":
    # Test Block
    t = np.linspace(0, 400, 401)
    r = np.ones_like(t) * 100000
    r[100:300] = 40 + np.random.normal(0, 2, 200)
    # Add synthetic defects
    r[150:200] += 10 * np.sin(2 * np.pi * 60 * t[150:200]/1000) # Chatter
    r[305:315] = 4000 # Bounce
    
    df = pd.DataFrame({'T_ms': t, 'Resistance': r})
    
    engine = AdvancedDCRMEngine()
    report = engine.analyze(df)
    print(json.dumps(report, indent=2))
