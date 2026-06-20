import json

file_path = 'notebook/kaggle_1.3_deberta_siglip2.ipynb'
with open(file_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

for cell in notebook.get('cells', []):
    if cell.get('cell_type') == 'code':
        lines = cell['source']
        for i, line in enumerate(lines):
            # Thay thế tên mô hình sai thành tên chuẩn của timm
            if 'vit_base_patch16_siglip2_256' in line:
                lines[i] = line.replace('vit_base_patch16_siglip2_256', 'vit_base_patch16_siglip_256')

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, ensure_ascii=False, indent=2)
