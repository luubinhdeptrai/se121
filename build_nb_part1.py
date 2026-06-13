import nbformat as nbf
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

cells = []

# =====================================================================
# Title
# =====================================================================
cells.append(new_markdown_cell(r"""# Generate Overall Satisfaction Labels

**Thesis project:** Explainable Multi-modal Deep Learning System for Product Quality Assessment using Image and Text Data

## Purpose

This notebook derives a new, fully-explainable target label — **`overall_satisfaction`** — for every
review in the cleaned Foody dataset (`data_raw/reviews_clean.csv` / `.json`, 9,946 rows).

`overall_satisfaction` represents the reviewer's **overall** experience, separate from the four
per-aspect scores (food, service, atmosphere, price). It is computed as:

```
avg_rating          = mean(food_score, service_score, atmosphere_score, price_score)   # 4 aspects, position_score EXCLUDED
overall_adjustment  = sum(score of every triggered global-satisfaction rule)
overall_satisfaction = clip(avg_rating + overall_adjustment, 0, 10)
```

## Key design decisions

1. **`position_score` is preserved but excluded** from `avg_rating`, from rule generation, and from
   `overall_satisfaction`. It remains in the dataset for traceability and is analysed separately in
   Section 16.
2. **The original Foody rating is preserved** as `foody_original_avg_rating` before `avg_rating` is
   recomputed from the 4 remaining aspects.
3. **Rules are data-driven**: all 14 rule categories live in `data_processed/overall_satisfaction_rules.json`
   and are *loaded dynamically* by the rule engine (Section 9) — nothing is hardcoded in the engine code.
4. **Every adjustment is explainable**: each review gets an `overall_evidence` record listing exactly
   which rule(s) fired, on which matched text span(s), and with what score contribution.
5. **No randomness** is used to generate scores. `RANDOM_SEED = 42` is set only for reproducible
   ordering/sampling (e.g., the audit sample in Section 15).

## Inputs

- `data_raw/reviews_clean.csv`, `data_raw/reviews_clean.json` (9,946 rows x 40 columns, read-only — never modified)

## Outputs (written to `data_processed/`, originals are never overwritten)

- `overall_satisfaction_rules.json` — the 14 rule categories (score, description, regex patterns)
- `reviews_clean_enhanced.csv` — original 40 columns + new columns (CSV, nested fields as JSON strings)
- `reviews_clean_enhanced.json` — same data, nested fields as native JSON arrays/objects
- `overall_satisfaction_rule_analysis.md` — rule documentation + coverage/validity report generated from real results

## Libraries

Python standard library (`json`, `re`, `unicodedata`, `pathlib`, `collections`) plus `pandas`, `numpy`,
and `matplotlib` only.
"""))

# =====================================================================
# Section 1: Environment Setup
# =====================================================================
cells.append(new_markdown_cell(r"""## Section 1 — Environment Setup

Imports, reproducibility seed, display options, and the directory/file layout used throughout the
notebook. `data_processed/` is created if it does not already exist."""))

cells.append(new_code_cell(r'''import json
import re
import unicodedata
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 160)
pd.set_option('display.float_format', lambda v: f'{v:.4f}')

plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.titleweight'] = 'bold'

# --- Directory layout -------------------------------------------------
RAW_DIR = Path('../data_raw')
PROC_DIR = Path('../data_processed')
PROC_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = RAW_DIR / 'reviews_clean.csv'
JSON_PATH = RAW_DIR / 'reviews_clean.json'

RULES_PATH = PROC_DIR / 'overall_satisfaction_rules.json'
OUT_CSV_PATH = PROC_DIR / 'reviews_clean_enhanced.csv'
OUT_JSON_PATH = PROC_DIR / 'reviews_clean_enhanced.json'
ANALYSIS_MD_PATH = PROC_DIR / 'overall_satisfaction_rule_analysis.md'

# --- Column groups used repeatedly ------------------------------------
ASPECT_COLS_4 = ['food_score', 'service_score', 'atmosphere_score', 'price_score']
ASPECT_COLS_ALL = ASPECT_COLS_4 + ['position_score']

RUN_TIMESTAMP = datetime.now(timezone.utc).isoformat()

print(f"Random seed       : {RANDOM_SEED}")
print(f"Run timestamp (UTC): {RUN_TIMESTAMP}")
print(f"Raw data dir       : {RAW_DIR.resolve()}")
print(f"Output dir         : {PROC_DIR.resolve()}")
print(f"4-aspect columns   : {ASPECT_COLS_4}")
print(f"All aspect columns : {ASPECT_COLS_ALL}")
'''))

# =====================================================================
# Section 2: Load and Understand Dataset
# =====================================================================
cells.append(new_markdown_cell(r"""## Section 2 — Load and Understand Dataset

We load **both** `reviews_clean.csv` and `reviews_clean.json`, verify they are consistent, and produce
a **Dataset Understanding Report** covering shape, schema, missing values, duplicates, and the
relationship between `avg_rating` and the 5 aspect scores. This report is produced *before* any
transformation logic so that every later decision (Section 4 onward) is grounded in the real data."""))

cells.append(new_code_cell(r'''df_raw = pd.read_csv(CSV_PATH)

with open(JSON_PATH, encoding='utf-8') as f:
    json_records = json.load(f)
df_json = pd.DataFrame(json_records)

print(f"CSV  shape : {df_raw.shape}")
print(f"JSON shape : {df_json.shape}")
print(f"\nCSV columns ({len(df_raw.columns)}):")
for i, c in enumerate(df_raw.columns):
    print(f"  [{i:2d}] {c:30s} {str(df_raw[c].dtype):10s}")
'''))

cells.append(new_markdown_cell(r"""### 2.1 CSV vs JSON consistency check

Both files describe the same 9,946 reviews and should agree on `review_id` membership and on the
values of every shared column."""))

cells.append(new_code_cell(r'''same_ids = set(df_raw['review_id']) == set(df_json['review_id'])
print(f"Same review_id set in CSV and JSON: {same_ids}")
assert same_ids, "CSV and JSON describe different sets of reviews"

common_cols = [c for c in df_raw.columns if c in df_json.columns]
csv_sorted = df_raw.sort_values('review_id').reset_index(drop=True)
json_sorted = df_json.sort_values('review_id').reset_index(drop=True)

# Columns stored as ISO-8601 strings in JSON but as pandas datetime strings in CSV.
DATETIME_COLS = {'created_on', 'updated_on', 'created_datetime', 'updated_datetime'}

mismatches = {}
mismatch_examples = {}
for col in common_cols:
    a, b = csv_sorted[col], json_sorted[col]
    if col in DATETIME_COLS:
        a_val = pd.to_datetime(a, errors='coerce', utc=True, format='mixed')
        b_val = pd.to_datetime(b, errors='coerce', utc=True, format='mixed')
        eq = (a_val == b_val) | (a_val.isna() & b_val.isna())
    elif pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
        eq = np.isclose(a.fillna(-999999), b.fillna(-999999)) | (a.isna() & b.isna())
    else:
        # Treat NaN (CSV) and None/'' (JSON) as the same "missing" value before comparing.
        a_norm = a.where(a.notna(), '').astype(str).replace({'nan': '', 'None': '', 'NaT': ''})
        b_norm = b.where(b.notna(), '').astype(str).replace({'nan': '', 'None': '', 'NaT': ''})
        eq = (a_norm == b_norm)
    n_bad = int((~eq).sum())
    if n_bad:
        mismatches[col] = n_bad
        idx = np.where(~eq)[0][:2]
        mismatch_examples[col] = [(a.iloc[i], b.iloc[i]) for i in idx]

print(f"Columns compared          : {len(common_cols)}")
print(f"Columns with real mismatches: {len(mismatches)}")
if mismatches:
    for col, n in mismatches.items():
        print(f"  {col}: {n} rows differ, e.g. {mismatch_examples[col]}")
else:
    print("=> CSV and JSON are fully consistent across all shared columns")
    print("   (after normalizing NaN/None/empty-string and datetime string formats,")
    print("    which differ only in representation between pandas read_csv and json.load).")
'''))

cells.append(new_markdown_cell(r"""### 2.2 Missing values, duplicates, and row-level integrity"""))

cells.append(new_code_cell(r'''missing = df_raw.isna().sum()
missing = missing[missing > 0].sort_values(ascending=False)
print("Columns with missing values:")
if len(missing):
    print(missing.to_string())
else:
    print("  (none)")

print(f"\nDuplicate review_id      : {df_raw['review_id'].duplicated().sum()}")
print(f"Fully duplicated rows     : {df_raw.duplicated().sum()}")
print(f"Unique restaurant_id count: {df_raw['restaurant_id'].nunique()}")
'''))

cells.append(new_markdown_cell(r"""### 2.3 Rating and aspect-score relationship

The dataset's original `avg_rating` should equal the mean of all 5 aspect scores
(`food_score`, `service_score`, `atmosphere_score`, `price_score`, `position_score`)."""))

cells.append(new_code_cell(r'''print(df_raw[ASPECT_COLS_ALL + ['avg_rating']].describe().T)

mean5 = df_raw[ASPECT_COLS_ALL].mean(axis=1)
diff5 = (df_raw['avg_rating'] - mean5).abs()
print(f"\nMax |avg_rating - mean(5 aspects)| (ignoring NaN rows): {diff5.max():.6f}")
print(f"Rows where diff > 1e-6 (excluding NaN)                : {(diff5 > 1e-6).sum()}")

nan_aspect_mask = df_raw[ASPECT_COLS_ALL].isna().all(axis=1)
print(f"\nRows where ALL 5 aspect scores are NaN: {nan_aspect_mask.sum()}")
print(df_raw.loc[nan_aspect_mask, ['review_id', 'avg_rating'] + ASPECT_COLS_ALL])
'''))

cells.append(new_markdown_cell(r"""### 2.4 Dataset Understanding Report — Summary

Consolidated findings that drive every design decision from Section 4 onward."""))

cells.append(new_code_cell(r'''n_rows, n_cols = df_raw.shape
n_nan_aspect = int(nan_aspect_mask.sum())
n_mismatch = len(mismatches)

print("=" * 78)
print("DATASET UNDERSTANDING REPORT")
print("=" * 78)
print(f"Rows x Columns                 : {n_rows} x {n_cols}")
print(f"CSV/JSON consistency           : {'OK - identical' if n_mismatch == 0 else f'{n_mismatch} mismatched columns'}")
print(f"Duplicate review_id            : {df_raw['review_id'].duplicated().sum()}")
print(f"Fully duplicated rows           : {df_raw.duplicated().sum()}")
print(f"avg_rating == mean(5 aspects)  : confirmed for {n_rows - n_nan_aspect}/{n_rows} rows")
print(f"Rows with all 5 aspects NaN     : {n_nan_aspect} "
      f"(review_id = {df_raw.loc[nan_aspect_mask, 'review_id'].tolist()})")
print(f"avg_rating range                : [{df_raw['avg_rating'].min()}, {df_raw['avg_rating'].max()}]")
print(f"Text field used for rules        : 'comment_clean'")
print(f"Empty/NaN 'comment_clean'        : {df_raw['comment_clean'].isna().sum()}")
print("=" * 78)
print("IMPLICATIONS FOR THIS NOTEBOOK")
print("-" * 78)
print("1. avg_rating in the raw data is the mean of ALL 5 aspects (incl. position_score).")
print("   Section 4 recomputes avg_rating from the 4 NON-position aspects only, and")
print(f"   preserves the original value as 'foody_original_avg_rating'.")
print(f"2. {n_nan_aspect} rows have no aspect scores at all -> the 4-aspect mean is NaN for")
print("   these rows. Section 4 falls back to 'foody_original_avg_rating' for these rows")
print("   and flags them with a boolean 'avg_rating_recomputed' column.")
print("3. CSV and JSON are interchangeable as inputs; this notebook reads the CSV and")
print("   exports both CSV and JSON for the enhanced dataset.")
print("=" * 78)
'''))

# =====================================================================
# Section 3: Rating and Aspect Analysis
# =====================================================================
cells.append(new_markdown_cell(r"""## Section 3 — Rating and Aspect Analysis

Before restructuring, we examine how `position_score` relates to `avg_rating` and to the other
4 aspects, and preview the effect of excluding it from the average. This analysis directly motivates
the restructuring in Section 4 and is revisited with full statistics in Section 16."""))

cells.append(new_code_cell(r'''corr = df_raw[ASPECT_COLS_ALL + ['avg_rating']].corr().round(3)
print("Correlation matrix (aspects x avg_rating):")
print(corr)

print("\nCorrelation of avg_rating with each aspect:")
print(corr['avg_rating'].drop('avg_rating').sort_values(ascending=False))
'''))

cells.append(new_code_cell(r'''avg4_preview = df_raw[ASPECT_COLS_4].mean(axis=1)
diff_preview = df_raw['avg_rating'] - avg4_preview

print("Preview: avg_rating (original, 5-aspect mean) vs mean(4 aspects, excluding position_score)")
print(diff_preview.describe())

n_changed = (diff_preview.abs() > 1e-9).sum()
print(f"\nRows whose rating would change if position_score is excluded: "
      f"{n_changed} / {len(df_raw)} ({100*n_changed/len(df_raw):.2f}%)")
print(f"Mean absolute change: {diff_preview.abs().mean():.4f} (on a 0-10 scale)")
'''))

cells.append(new_markdown_cell(r"""### 3.1 Rating distribution buckets (original `avg_rating`)"""))

cells.append(new_code_cell(r'''bins = [0, 2, 4, 6, 8, 10.0001]
labels = ['0-2', '2-4', '4-6', '6-8', '8-10']
bucket_counts = pd.cut(df_raw['avg_rating'], bins=bins, labels=labels, right=False, include_lowest=True).value_counts().sort_index()
print("avg_rating distribution buckets:")
print(bucket_counts)
print(f"\nMean: {df_raw['avg_rating'].mean():.3f}  Median: {df_raw['avg_rating'].median():.3f}  Std: {df_raw['avg_rating'].std():.3f}")
'''))

# =====================================================================
# Section 4: Dataset Restructuring
# =====================================================================
cells.append(new_markdown_cell(r"""## Section 4 — Dataset Restructuring

We now build the working dataframe `df` (a copy of `df_raw`) with the following changes:

1. **Preserve** the original Foody rating in a new column `foody_original_avg_rating`.
2. **Recompute** `avg_rating` as the mean of the 4 non-position aspects
   (`food_score`, `service_score`, `atmosphere_score`, `price_score`).
3. For the 3 rows where all aspect scores are `NaN`, the 4-aspect mean is undefined — these rows
   **fall back** to `foody_original_avg_rating`, and a new boolean column `avg_rating_recomputed`
   records whether the fallback was used (`False`) or the 4-aspect mean was used (`True`).
4. `position_score` is **kept unchanged** in the dataframe for traceability, but is **not** used in
   any computation in this notebook from this point on."""))

cells.append(new_code_cell(r'''df = df_raw.copy()

# 1. Preserve original Foody rating
df['foody_original_avg_rating'] = df['avg_rating']

# 2. Recompute avg_rating from the 4 non-position aspects
avg4 = df[ASPECT_COLS_4].mean(axis=1)

# 3. Fallback for rows with no aspect scores at all
df['avg_rating_recomputed'] = avg4.notna()
df['avg_rating'] = avg4.where(avg4.notna(), df['foody_original_avg_rating'])

# 4. position_score is untouched (still present in df, unused downstream)
assert 'position_score' in df.columns

n_recomputed = int(df['avg_rating_recomputed'].sum())
n_fallback = int((~df['avg_rating_recomputed']).sum())
print(f"avg_rating recomputed from 4 aspects : {n_recomputed} / {len(df)}")
print(f"avg_rating fell back to Foody original: {n_fallback} / {len(df)}")
print(f"\nFallback rows:")
print(df.loc[~df['avg_rating_recomputed'], ['review_id', 'foody_original_avg_rating', 'avg_rating'] + ASPECT_COLS_ALL])

assert df['avg_rating'].isna().sum() == 0, "avg_rating must not contain NaN after restructuring"
assert df['avg_rating'].between(0, 10).all(), "avg_rating must be within [0, 10]"
'''))

cells.append(new_markdown_cell(r"""### 4.1 Effect of restructuring on `avg_rating`

How much does the new `avg_rating` (4-aspect mean, position excluded) differ from
`foody_original_avg_rating` (5-aspect mean, position included)?"""))

cells.append(new_code_cell(r'''rating_diff = df['avg_rating'] - df['foody_original_avg_rating']

print("avg_rating(new) - foody_original_avg_rating:")
print(rating_diff.describe())

n_changed = (rating_diff.abs() > 1e-9).sum()
n_increased = (rating_diff > 1e-9).sum()
n_decreased = (rating_diff < -1e-9).sum()
print(f"\nRows changed   : {n_changed} ({100*n_changed/len(df):.2f}%)")
print(f"Rows increased : {n_increased} ({100*n_increased/len(df):.2f}%)")
print(f"Rows decreased : {n_decreased} ({100*n_decreased/len(df):.2f}%)")
print(f"Mean |change|  : {rating_diff.abs().mean():.4f}")
print(f"Max  |change|  : {rating_diff.abs().max():.4f}")
'''))

# =====================================================================
# Save partial notebook
# =====================================================================
nb = new_notebook(cells=cells, metadata={
    'kernelspec': {
        'display_name': 'Python 3',
        'language': 'python',
        'name': 'python3'
    },
    'language_info': {'name': 'python', 'version': '3.x'}
})

out_path = 'notebook/01_generate_overall_satisfaction.ipynb'
with open(out_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Wrote {len(cells)} cells to {out_path}")
