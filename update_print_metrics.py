import json

notebooks = [
    '/Users/abc/Documents/SE/notebook/EXP_060B_swinb_visobert_gmu_uncertainty.ipynb',
    '/Users/abc/Documents/SE/notebook/EXP_060C_efficientnetb3_phobert_film_huber.ipynb'
]

replacement_code = """import json
!cp -r ./experiments/$EXP_ID/* $DRIVE_EXP_PATH/

print(f'\\n=========================================')
print(f'=== {EXP_ID} RESULTS ===')
print(f'=========================================')

# --- 1. Print Validation Metrics ---
try:
    with open(f'./experiments/{EXP_ID}/metrics.json') as f:
        m = json.load(f)
    print('\\n[ VALIDATION SET ]')
    print(f"Loss: {m['loss']:.4f}")
    print("             MAE      RMSE      R2")
    for n in ['food', 'price', 'atmos', 'service', 'overall']:
        print(f"  {n:<8} : {m.get(f'mae_{n}',0):.4f}   {m.get(f'rmse_{n}',0):.4f}   {m.get(f'r2_{n}',0):.4f}")
    print(f"\\n  mean_mae   : {m.get('mean_mae',0):.4f}")
except FileNotFoundError:
    print("\\n[ VALIDATION SET ] - No metrics.json found.")

# --- 2. Print Test Metrics ---
try:
    with open(f'./experiments/{EXP_ID}/test_metrics.json') as f:
        t = json.load(f)
    print('\\n[ TEST SET ]')
    print("             MAE      RMSE      R2")
    for n in ['food', 'price', 'atmos', 'service', 'overall']:
        print(f"  {n:<8} : {t.get(f'mae_{n}',0):.4f}   {t.get(f'rmse_{n}',0):.4f}   {t.get(f'r2_{n}',0):.4f}")
    print(f"\\n  mean_mae   : {t.get('mean_mae',0):.4f}")
    print(f"  aspect_mae : {t.get('aspect_mae',0):.4f}")
    print(f"  overall_mae: {t.get('overall_mae',0):.4f}")
except FileNotFoundError:
    print("\\n[ TEST SET ] - No test_metrics.json found.")
"""

for file_path in notebooks:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
            
        cells = nb['cells']
        # The print cell should be the very last code cell
        for i in range(len(cells)-1, -1, -1):
            if cells[i]['cell_type'] == 'code':
                src = "".join(cells[i].get('source', []))
                if '!cp -r' in src or 'metrics.json' in src:
                    cells[i]['source'] = [line + '\n' for line in replacement_code.split('\n')]
                    # Fix the trailing newline for the last line
                    cells[i]['source'][-1] = cells[i]['source'][-1].strip('\n')
                    break
                    
        nb['cells'] = cells
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=2)
        print(f"Updated print metrics cell in {file_path}")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
