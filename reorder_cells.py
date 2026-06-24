import json

notebooks = [
    '/Users/abc/Documents/SE/notebook/EXP_060B_swinb_visobert_gmu_uncertainty.ipynb',
    '/Users/abc/Documents/SE/notebook/EXP_060C_efficientnetb3_phobert_film_huber.ipynb'
]

for file_path in notebooks:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
            
        cells = nb['cells']
        
        # We want to identify the "Save to Drive" cells and the "Test" cells
        save_drive_idx = -1
        test_idx = -1
        
        for i, cell in enumerate(cells):
            src = "".join(cell.get('source', []))
            if 'Save to Drive' in src and cell['cell_type'] == 'markdown':
                save_drive_idx = i
            if 'Evaluate on Test Set' in src and cell['cell_type'] == 'markdown':
                test_idx = i
                
        if save_drive_idx != -1 and test_idx != -1 and save_drive_idx < test_idx:
            # We need to swap the two blocks
            # Save to Drive block is save_drive_idx and save_drive_idx + 1
            # Test block is test_idx and test_idx + 1
            
            save_md = cells[save_drive_idx]
            save_code = cells[save_drive_idx + 1]
            test_md = cells[test_idx]
            test_code = cells[test_idx + 1]
            
            # Remove the extra !cp -r in the test code
            test_src = test_code['source']
            new_test_src = [line for line in test_src if 'cp -r' not in line and 'Sync test results' not in line]
            test_code['source'] = new_test_src
            
            # Fix numbering
            # test block becomes STEP 7
            test_md['source'][0] = "### STEP 7: Evaluate on Test Set\n"
            # save block becomes STEP 8
            save_md['source'][0] = "### STEP 8: Save to Drive + print metrics\n"
            
            # Reconstruct the cells
            new_cells = cells[:save_drive_idx] + [test_md, test_code, save_md, save_code]
            if len(cells) > test_idx + 2:
                new_cells.extend(cells[test_idx + 2:])
                
            nb['cells'] = new_cells
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(nb, f, indent=2)
            print(f"Reordered cells successfully for {file_path}")
        else:
            print(f"Blocks not found or already in correct order for {file_path}")
            
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
