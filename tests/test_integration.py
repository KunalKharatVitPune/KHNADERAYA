import sys
import os

# Add the parent directory to sys.path to allow imports if needed, 
# but since we are running this script directly, we can just import the modules if they are in the same folder.
# However, to be safe and mimic how they might be used:

try:
    import raj_rule_engine
    import raj_ai_agent
    
    print("Successfully imported raj_rule_engine and raj_ai_agent")
    
    if hasattr(raj_rule_engine, 'analyze_dcrm_advanced'):
        print("raj_rule_engine.analyze_dcrm_advanced found")
    else:
        print("ERROR: raj_rule_engine.analyze_dcrm_advanced NOT found")
        
    if hasattr(raj_ai_agent, 'detect_fault'):
        print("raj_ai_agent.detect_fault found")
    else:
        print("ERROR: raj_ai_agent.detect_fault NOT found")

except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
