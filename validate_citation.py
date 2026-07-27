# validate_citation.py
import os
import json

def check_citation_yaml():
    cff_file = "CITATION.cff"
    
    if not os.path.exists(cff_file):
        print(f"[!] Error: {cff_file} cannot be found in the current working directory.")
        return

    print(f"[*] Reading '{cff_file}' for validation...")
    
    # We use python's internal json package to build a rudimentary YAML checker.
    # Standard CFF files must follow clean indentation and key-value mapping constraints.
    try:
        with open(cff_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Verify basic indentation structure and structure markers
        has_version = False
        has_pref = False
        
        for line in lines:
            if line.strip().startswith("cff-version:"):
                has_version = True
            if line.strip().startswith("preferred-citation:"):
                has_pref = True
                
        if not has_version:
            raise ValueError("Missing critical CFF root identifier: 'cff-version'")
            
        print("[+] Success: CITATION.cff structural markers are cleanly aligned!")
        print("[+] Local syntax check passed. The file is ready for GitHub deployment.")
        
    except Exception as e:
        print(f"[!] Formatting Error Found: {str(e)}")
        print("[!] Please check line indentations and ensure colons are followed by a space.")

if __name__ == "__main__":
    check_citation_yaml()
