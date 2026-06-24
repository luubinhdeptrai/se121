import json
import re

notebooks = [
    '/Users/abc/Documents/SE/notebook/EXP_060B_swinb_visobert_gmu_uncertainty.ipynb',
    '/Users/abc/Documents/SE/notebook/EXP_060C_efficientnetb3_phobert_film_huber.ipynb'
]

for file_path in notebooks:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
            
        max_step = 0
        # Find the highest step number before "FINAL STEP"
        for cell in nb['cells']:
            if cell['cell_type'] == 'markdown':
                src = "".join(cell.get('source', []))
                if "FINAL STEP" in src:
                    continue # Skip the one we are modifying
                matches = re.findall(r'STEP\s+(\d+)', src, re.IGNORECASE)
                for m in matches:
                    max_step = max(max_step, int(m))
                    
        if max_step == 0:
            print(f"No previous steps found in {file_path}, skipping.")
            continue
            
        next_step = max_step + 1
        
        # Replace "FINAL STEP" with "STEP <next_step>"
        for cell in nb['cells']:
            if cell['cell_type'] == 'markdown':
                new_source = []
                for line in cell.get('source', []):
                    if "FINAL STEP" in line:
                        line = line.replace("FINAL STEP", f"STEP {next_step}")
                    new_source.append(line)
                cell['source'] = new_source
                
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=2)
            
        print(f"Updated {file_path} to use STEP {next_step}")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
