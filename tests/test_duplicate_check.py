"""
Test to verify JSON structure is correct after fixes
"""
import json

def check_json_structure(json_file_path):
    """
    Verify the JSON structure matches requirements:
    1. No duplicate maintenanceActions or futureFaultsPdf
    2. They should be INSIDE aiVerdict only
    3. No breakerId or operator fields
    4. Proper aiVerdict field ordering (related fields consecutive)
    5. Coil descriptions populated
    6. Overall healthScore averaging R, Y, B phases
    """
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    issues = []
    
    print(f"\n{'='*60}")
    print("TOP LEVEL STRUCTURE CHECK")
    print(f"{'='*60}")
    
    # Check for overall healthScore
    if "healthScore" in data:
        print(f"✅ Overall healthScore found: {data['healthScore']}")
        
        # Verify it's an average of R, Y, B
        phase_scores = []
        for phase in ['r', 'y', 'b']:
            if phase in data and "healthScore" in data[phase]:
                phase_scores.append(data[phase]["healthScore"])
        
        if phase_scores:
            calculated_avg = round(sum(phase_scores) / len(phase_scores), 1)
            if abs(data["healthScore"] - calculated_avg) < 0.2:  # Allow small rounding differences
                print(f"✅ healthScore ({data['healthScore']}) is correct average of R={phase_scores[0] if len(phase_scores)>0 else 'N/A'}, Y={phase_scores[1] if len(phase_scores)>1 else 'N/A'}, B={phase_scores[2] if len(phase_scores)>2 else 'N/A'}")
            else:
                issues.append(f"⚠️  healthScore ({data['healthScore']}) doesn't match calculated average ({calculated_avg})")
    else:
        issues.append("❌ Overall healthScore NOT found at top level")
    
    # Check top level should NOT have these
    if "breakerId" in data:
        issues.append("❌ breakerId found at top level (should be removed)")
    else:
        print(f"✅ No breakerId field at top level")
        
    if "operator" in data:
        issues.append("❌ operator found at top level (should be removed)")
    else:
        print(f"✅ No operator field at top level")
    
    # Check each phase (r, y, b)
    for phase in ['r', 'y', 'b']:
        if phase not in data:
            continue
            
        phase_data = data[phase]
        print(f"\n{'='*60}")
        print(f"Checking Phase: {phase.upper()}")
        print(f"{'='*60}")
        
        # Check for removed fields
        if "breakerId" in phase_data:
            issues.append(f"❌ [{phase.upper()}] breakerId found (should be removed)")
        else:
            print(f"✅ [{phase.upper()}] No breakerId field")
            
        if "operator" in phase_data:
            issues.append(f"❌ [{phase.upper()}] operator found (should be removed)")
        else:
            print(f"✅ [{phase.upper()}] No operator field")
        
        # Check top level should NOT have these
        if "maintenanceActions" in phase_data and "aiVerdict" in phase_data:
            issues.append(f"❌ [{phase.upper()}] maintenanceActions at top level (should be ONLY in aiVerdict)")
        
        if "futureFaultsPdf" in phase_data and "aiVerdict" in phase_data:
            issues.append(f"❌ [{phase.upper()}] futureFaultsPdf at top level (should be ONLY in aiVerdict)")
        
        # Check aiVerdict structure
        if "aiVerdict" in phase_data:
            ai_verdict = phase_data["aiVerdict"]
            
            # Check they exist inside aiVerdict
            if "maintenanceActions" not in ai_verdict:
                issues.append(f"❌ [{phase.upper()}] maintenanceActions NOT in aiVerdict")
            else:
                print(f"✅ [{phase.upper()}] maintenanceActions found in aiVerdict")
                
            if "futureFaultsPdf" not in ai_verdict:
                issues.append(f"❌ [{phase.upper()}] futureFaultsPdf NOT in aiVerdict")
            else:
                print(f"✅ [{phase.upper()}] futureFaultsPdf found in aiVerdict")
            
            # Check field ordering - related fields should be consecutive
            actual_keys = list(ai_verdict.keys())
            print(f"   Field order: {actual_keys[:10]}")
            
            # Check that core verdict fields are together
            core_fields = ["confidence", "faultLabel", "severity", "severityReason", "rulEstimate", "uncertainty"]
            core_indices = []
            for field in core_fields:
                if field in actual_keys:
                    core_indices.append(actual_keys.index(field))
            
            if core_indices:
                # Check if they are consecutive
                if core_indices == list(range(min(core_indices), min(core_indices) + len(core_indices))):
                    print(f"✅ [{phase.upper()}] Core verdict fields are CONSECUTIVE (indices: {core_indices})")
                else:
                    issues.append(f"⚠️  [{phase.upper()}] Core verdict fields are NOT consecutive (indices: {core_indices})")
            
            # Check formats
            if "maintenanceActions" in ai_verdict:
                actions = ai_verdict["maintenanceActions"]
                if actions and len(actions) > 0:
                    first_action = actions[0]
                    if "actions" in first_action and "priority" in first_action and "color" in first_action:
                        print(f"✅ [{phase.upper()}] maintenanceActions format is CORRECT")
                    else:
                        issues.append(f"❌ [{phase.upper()}] maintenanceActions has wrong format")
            
            if "futureFaultsPdf" in ai_verdict:
                faults = ai_verdict["futureFaultsPdf"]
                if faults and len(faults) > 0:
                    first_fault = faults[0]
                    required_keys = ["id", "fault", "probability", "timeline", "evidence", "color"]
                    missing = [k for k in required_keys if k not in first_fault]
                    if not missing:
                        print(f"✅ [{phase.upper()}] futureFaultsPdf format is CORRECT")
                    else:
                        issues.append(f"❌ [{phase.upper()}] futureFaultsPdf missing keys: {missing}")
        
        # Check phaseWiseAnalysis coil descriptions
        if "phaseWiseAnalysis" in phase_data:
            phases_list = phase_data["phaseWiseAnalysis"]
            has_descriptions = False
            for p in phases_list:
                if "waveformAnalysis" in p and "coilAnalysis" in p["waveformAnalysis"]:
                    coils = p["waveformAnalysis"]["coilAnalysis"]
                    for coil_name in ["closeCoil", "tripCoil1", "tripCoil2"]:
                        if coil_name in coils and "description" in coils[coil_name]:
                            desc = coils[coil_name]["description"]
                            if desc and len(desc) > 0:
                                has_descriptions = True
                                break
                    if has_descriptions:
                        break
            
            if has_descriptions:
                print(f"✅ [{phase.upper()}] Coil descriptions are populated")
            else:
                issues.append(f"⚠️  [{phase.upper()}] Coil descriptions appear to be empty")
        
        print()
    
    # Summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    if issues:
        print(f"❌ {len(issues)} ISSUE(S) FOUND:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("✅ ALL CHECKS PASSED!")
        print("✅ Overall healthScore present and correct")
        print("✅ No duplicates found")
        print("✅ Correct field placement")
        print("✅ Core verdict fields are consecutive")
        print("✅ No breakerId or operator fields")
        print("✅ Coil descriptions populated")
    print(f"{'='*60}")
    
    return len(issues) == 0

if __name__ == "__main__":
    import sys
    
    # Test with the new file
    json_path = "c:/Users/rkhanke/Downloads/dcrm_three_phase_1764926929196.json"
    
    if len(sys.argv) > 1:
        json_path = sys.argv[1]
    
    print(f"Checking: {json_path}\n")
    check_json_structure(json_path)

