import json

notebooks = [
    '/Users/abc/Documents/SE/notebook/EXP_060B_swinb_visobert_gmu_uncertainty.ipynb',
    '/Users/abc/Documents/SE/notebook/EXP_060C_efficientnetb3_phobert_film_huber.ipynb'
]

exact_format = """import json
!cp -r ./experiments/$EXP_ID/* $DRIVE_EXP_PATH/

# --- VALIDATION METRICS ---
try:
    with open(f'./experiments/{EXP_ID}/metrics.json') as f:
        m = json.load(f)

    print(f'\\n=== {EXP_ID} Results (Validation) ===')
    print(f"Loss (val)   : {m['loss']:.4f}")
    print()
    print("             MAE      RMSE      R2")
    print(f"  food     : {m['mae_food']:.4f}   {m['rmse_food']:.4f}   {m['r2_food']:.4f}")
    print(f"  price    : {m['mae_price']:.4f}   {m['rmse_price']:.4f}   {m['r2_price']:.4f}")
    print(f"  atmos    : {m['mae_atmos']:.4f}   {m['rmse_atmos']:.4f}   {m['r2_atmos']:.4f}")
    print(f"  service  : {m['mae_service']:.4f}   {m['rmse_service']:.4f}   {m['r2_service']:.4f}")
    print(f"  overall  : {m['mae_overall']:.4f}   {m['rmse_overall']:.4f}   {m['r2_overall']:.4f}")
    print()
    print(f"  mean_mae   : {m['mean_mae']:.4f}")
    print(f"  aspect_mae : {m['aspect_mae']:.4f}")
    print(f"  overall_mae: {m['overall_mae']:.4f}")
except FileNotFoundError:
    pass

# --- TEST METRICS ---
try:
    with open(f'./experiments/{EXP_ID}/test_metrics.json') as f:
        t = json.load(f)

    print(f'\\n=== {EXP_ID} Results (Test) ===')
    print()
    print("             MAE      RMSE      R2")
    print(f"  food     : {t['mae_food']:.4f}   {t['rmse_food']:.4f}   {t['r2_food']:.4f}")
    print(f"  price    : {t['mae_price']:.4f}   {t['rmse_price']:.4f}   {t['r2_price']:.4f}")
    print(f"  atmos    : {t['mae_atmos']:.4f}   {t['rmse_atmos']:.4f}   {t['r2_atmos']:.4f}")
    print(f"  service  : {t['mae_service']:.4f}   {t['rmse_service']:.4f}   {t['r2_service']:.4f}")
    print(f"  overall  : {t['mae_overall']:.4f}   {t['rmse_overall']:.4f}   {t['r2_overall']:.4f}")
    print()
    print(f"  mean_mae   : {t['mean_mae']:.4f}")
    print(f"  aspect_mae : {t['aspect_mae']:.4f}")
    print(f"  overall_mae: {t['overall_mae']:.4f}")
except FileNotFoundError:
    pass
"""

for file_path in notebooks:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
            
        cells = nb['cells']
        for i in range(len(cells)-1, -1, -1):
            if cells[i]['cell_type'] == 'code':
                src = "".join(cells[i].get('source', []))
                if '!cp -r' in src or 'metrics.json' in src:
                    # Write the exact_format
                    cells[i]['source'] = [line + '\n' for line in exact_format.split('\n')]
                    cells[i]['source'][-1] = cells[i]['source'][-1].strip('\n')
                    break
                    
        nb['cells'] = cells
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=2)
        print(f"Restored format in {file_path}")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
