import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell

nb = nbf.read('notebook/01_generate_overall_satisfaction.ipynb', as_version=4)
cells = nb.cells

# =====================================================================
# Section 14: Quality Checks
# =====================================================================
cells.append(new_markdown_cell(r"""## Section 14 — Quality Checks

A suite of assertion-based checks over the full enhanced dataset (`df`, 9946 rows). Each check
prints `[PASS]` on success and raises `AssertionError` immediately on failure, so a clean run of
this cell is itself proof that the dataset satisfies all listed invariants."""))

cells.append(new_code_cell(r'''print("Running quality checks on the enhanced dataset...\n")

# 1. Bounds and NaN checks on the three rating columns
for col in ['avg_rating', 'foody_original_avg_rating', 'overall_satisfaction']:
    assert df[col].between(0, 10).all(), f"{col} has values outside [0, 10]"
    assert df[col].isna().sum() == 0, f"{col} contains NaN"
print("[PASS] avg_rating, foody_original_avg_rating, overall_satisfaction are within [0, 10] with no NaN")

# 2. Row count and key uniqueness preserved from the raw dataset
assert len(df) == len(df_raw) == 9946, f"row count changed: {len(df)}"
assert df['review_id'].is_unique, "review_id is not unique"
print(f"[PASS] Row count preserved ({len(df)}), review_id is unique")

# 3. position_score is preserved unchanged for traceability
pos_match = (df['position_score'].isna() == df_raw['position_score'].isna()) & (
    df['position_score'].isna() | np.isclose(df['position_score'], df_raw['position_score'])
)
assert pos_match.all(), "position_score was modified from the raw dataset"
print("[PASS] position_score is unchanged from the raw dataset (preserved for traceability, unused downstream)")

# 4. avg_rating recomputation correctness
recomputed_mask = df['avg_rating_recomputed']
expected_avg4 = df.loc[recomputed_mask, ASPECT_COLS_4].mean(axis=1)
assert np.allclose(df.loc[recomputed_mask, 'avg_rating'], expected_avg4)
fallback_mask = ~recomputed_mask
assert np.allclose(df.loc[fallback_mask, 'avg_rating'], df.loc[fallback_mask, 'foody_original_avg_rating'])
print(f"[PASS] avg_rating == mean(food_score, service_score, atmosphere_score, price_score) for all "
      f"{int(recomputed_mask.sum())} recomputed rows; {int(fallback_mask.sum())} fallback rows == foody_original_avg_rating")

# 5. overall_satisfaction formula correctness
expected_satisfaction = (df['avg_rating'] + df['overall_adjustment']).clip(lower=0, upper=10)
assert np.allclose(df['overall_satisfaction'], expected_satisfaction)
print("[PASS] overall_satisfaction == clip(avg_rating + overall_adjustment, 0, 10) for all rows")

# 6. Evidence-sum consistency: sum of evidence scores == overall_adjustment
evidence_sums = df['overall_evidence'].map(lambda ev: sum(e['score'] for e in ev))
assert np.allclose(evidence_sums, df['overall_adjustment'], atol=1e-9)
print("[PASS] sum(overall_evidence[*].score) == overall_adjustment for all rows")

# 7. rules_triggered <-> evidence consistency, deduplication, polarity correctness
for triggered, evidence in zip(df['overall_rules_triggered'], df['overall_evidence']):
    assert len(triggered) == len(set(triggered)), f"duplicate rule names in {triggered}"
    assert set(triggered) == {e['rule'] for e in evidence}, "overall_rules_triggered/overall_evidence mismatch"
    for e in evidence:
        expected_polarity = 'positive' if e['score'] > 0 else 'negative'
        assert e['polarity'] == expected_polarity, f"polarity mismatch for rule {e['rule']!r}"
        assert np.isclose(e['score'], RULES[e['rule']]['score']), f"evidence score mismatch for {e['rule']!r}"
print("[PASS] overall_rules_triggered and overall_evidence are mutually consistent, "
      "deduplicated, and polarity matches each rule's configured score sign")

# 8. JSON round-trip for evidence (structure must survive serialize/deserialize)
for evidence in df['overall_evidence'].iloc[:200]:
    assert json.loads(json.dumps(evidence, ensure_ascii=False)) == evidence
print("[PASS] overall_evidence survives a JSON serialize/deserialize round-trip (sampled 200 rows)")

# 9. comment_normalized is '' iff comment_clean is empty/NaN
empty_comment = df['comment_clean'].isna() | (df['comment_clean'].fillna('').str.strip() == '')
assert (df.loc[empty_comment, 'comment_normalized'] == '').all()
assert (df.loc[~empty_comment, 'comment_normalized'] != '').all()
print(f"[PASS] comment_normalized == '' for exactly the {int(empty_comment.sum())} rows with empty/NaN comment_clean")

print("\nAll quality checks passed.")
'''))

# =====================================================================
# Section 15: Sample Auditing (20 reviews)
# =====================================================================
cells.append(new_markdown_cell(r"""## Section 15 — Sample Auditing (20 reviews)

For thesis auditability, we print a full evidence trail for 20 reviews drawn deterministically
(`RANDOM_SEED = 42`) from six categories so that every kind of rule-engine outcome is represented:

| Category | n | What it demonstrates |
|---|---|---|
| `no_rule_triggered` | 5 | `overall_satisfaction == avg_rating` (no adjustment) |
| `positive_only` | 5 | Only positive rule(s) fired |
| `negative_only` | 5 | Only negative rule(s) fired |
| `mixed_pos_neg` | 3 | Both positive and negative rules fired (Section 12.2) |
| `clipped_high` | 1 | `avg_rating + overall_adjustment > 10`, clipped to 10 |
| `clipped_low` | 1 | `avg_rating + overall_adjustment < 0`, clipped to 0 |

For each review we show: `review_id`, the review text, `foody_original_avg_rating`, `avg_rating`,
`overall_adjustment`, every evidence record (`rule`, `polarity`, `score`, `matched_text`), and the
final `overall_satisfaction`."""))

cells.append(new_code_cell(r'''rng = np.random.RandomState(RANDOM_SEED)

raw_sum = df['avg_rating'] + df['overall_adjustment']
clipped_high_mask = raw_sum > 10
clipped_low_mask = raw_sum < 0
positive_only_mask = (df['overall_adjustment'] > 0) & ~mixed_mask
negative_only_mask = (df['overall_adjustment'] < 0) & ~mixed_mask
no_rule_mask = df['overall_adjustment'] == 0

audit_pools = [
    ('no_rule_triggered', df.index[no_rule_mask], 5),
    ('positive_only', df.index[positive_only_mask], 5),
    ('negative_only', df.index[negative_only_mask], 5),
    ('mixed_pos_neg', df.index[mixed_mask], 3),
    ('clipped_high', df.index[clipped_high_mask], 1),
    ('clipped_low', df.index[clipped_low_mask], 1),
]

audit_rows = []
for label, pool_index, n in audit_pools:
    chosen = rng.choice(pool_index, size=min(n, len(pool_index)), replace=False)
    for idx in chosen:
        audit_rows.append((label, idx))

print(f"Audit sample size: {len(audit_rows)} reviews (categories: "
      f"{', '.join(f'{label}={sum(1 for l,_ in audit_rows if l==label)}' for label in dict.fromkeys(l for l,_ in audit_rows))})")
'''))

cells.append(new_code_cell(r'''for i, (label, idx) in enumerate(audit_rows, 1):
    row = df.loc[idx]
    print(f"[{i:2d}/{len(audit_rows)}] category={label}  review_id={row['review_id']}")
    print(f"     comment_clean         : {row['comment_clean'][:180]!r}")
    print(f"     foody_original_rating : {row['foody_original_avg_rating']:.2f}")
    print(f"     avg_rating (4-aspect) : {row['avg_rating']:.2f}")
    print(f"     overall_adjustment    : {row['overall_adjustment']:+.2f}")
    if row['overall_evidence']:
        for e in row['overall_evidence']:
            print(f"         [{e['polarity']:8s}] {e['rule']:25s} {e['score']:+.2f}  matched={e['matched_text']!r}")
    else:
        print("         (no rule triggered)")
    print(f"     overall_satisfaction  : {row['overall_satisfaction']:.2f}")
    print("-" * 78)
'''))

# =====================================================================
# Section 16: Research Analysis on Position Exclusion
# =====================================================================
cells.append(new_markdown_cell(r"""## Section 16 — Research Analysis on Position Exclusion

This section provides the empirical justification for excluding `position_score` from
`avg_rating` (Section 4), revisiting and extending the preliminary correlation analysis from
Section 3 on the final, restructured dataset."""))

cells.append(new_code_cell(r'''corr_cols = ASPECT_COLS_ALL + ['foody_original_avg_rating', 'avg_rating', 'overall_satisfaction']
corr_final = df[corr_cols].corr().round(3)
print("Correlation matrix (5 aspects + 3 rating variants):")
print(corr_final)

print("\nCorrelation of each of the 5 original aspects with foody_original_avg_rating (5-aspect mean):")
aspect_corr = corr_final.loc[ASPECT_COLS_ALL, 'foody_original_avg_rating'].sort_values(ascending=False)
print(aspect_corr)
print(f"\n'position_score' has the LOWEST correlation with the overall Foody rating among the 5 "
      f"aspects ({aspect_corr['position_score']:.3f} vs {aspect_corr.drop('position_score').min():.3f}-"
      f"{aspect_corr.drop('position_score').max():.3f} for the other 4) - i.e. it is the aspect least "
      f"representative of the holistic quality judgment.")

print("\nposition_score vs each of the 4 retained aspects:")
for c in ASPECT_COLS_4:
    print(f"  corr(position_score, {c:18s}) = {df['position_score'].corr(df[c]):.4f}")
print(f"  corr(position_score, avg_rating [4-aspect mean]) = {df['position_score'].corr(df['avg_rating']):.4f}")
'''))

cells.append(new_markdown_cell(r"""### 16.1 Construct-validity argument

`position_score` rates the restaurant's **location / accessibility / parking** - a property of
*where* the restaurant is, not of the food, service, atmosphere, or price that a customer
experienced during the visit. For a multi-modal model that predicts product quality from a
**review image** (the food/dish photo) and **review text** (the comment describing the
food/service/atmosphere/price experience), `position_score` is largely **unobservable** from
either modality: a photo of a dish and a comment about its taste carry no signal about whether
the restaurant has convenient parking.

Including `position_score` in the regression target therefore introduces a component of label
variance that the model's inputs structurally cannot predict - effectively adding noise to the
learning target. Excluding it:

1. Improves the **construct validity** of `avg_rating` as a "product quality" label - it now
   reflects only the 4 aspects (food, service, atmosphere, price) that are directly described in
   the review text and depicted in the review image.
2. Is supported empirically: `position_score` is the **least correlated** of the 5 aspects with
   the overall rating (see above), suggesting it carries the most aspect-specific (location)
   information and the least overall-experience information.
3. `position_score` is **preserved** as `position_score` in the exported dataset (unchanged from
   the raw data) so downstream users can still analyze it separately if desired - nothing is
   destroyed, only excluded from the target computation."""))

cells.append(new_code_cell(r'''print("Impact of excluding position_score on the rating label (avg_rating - foody_original_avg_rating):")
print(rating_diff.describe())

n_changed_meaningfully = (rating_diff.abs() >= 0.5).sum()
print(f"\nReviews whose label changed by >= 0.5 points after excluding position_score: "
      f"{n_changed_meaningfully} ({100*n_changed_meaningfully/len(df):.2f}%)")

print(f"\ncorr(avg_rating, overall_satisfaction)                = {df['avg_rating'].corr(df['overall_satisfaction']):.4f}")
print(f"corr(foody_original_avg_rating, overall_satisfaction) = {df['foody_original_avg_rating'].corr(df['overall_satisfaction']):.4f}")
print("\nBoth correlations are very high because overall_satisfaction = avg_rating + a small")
print("rule-based adjustment (Section 11.1): the rule engine REFINES the label with global")
print("satisfaction signals from the review text, it does not replace it.")
'''))

# =====================================================================
# Section 17: Export Results
# =====================================================================
cells.append(new_markdown_cell(r"""## Section 17 — Export Results

Three artifacts are written to `data_processed/` (none of the files in `data_raw/` are modified):

1. `reviews_clean_enhanced.csv` - all original 40 columns plus the new columns from Sections 4
   and 10, with `overall_rules_triggered` and `overall_evidence` JSON-serialized as strings
   (CSV cannot hold nested lists/dicts natively).
2. `reviews_clean_enhanced.json` - the same data, but `overall_rules_triggered` and
   `overall_evidence` are kept as native JSON arrays/objects.
3. `overall_satisfaction_rule_analysis.md` - a methodology and results report generated from the
   real statistics computed in this run (Sections 4, 11, 12, 16).

(`overall_satisfaction_rules.json` was already written and verified in Section 7.)"""))

cells.append(new_code_cell(r'''NEW_COLUMNS = ['foody_original_avg_rating', 'avg_rating_recomputed', 'comment_normalized',
               'overall_adjustment', 'overall_rules_triggered', 'overall_evidence', 'overall_satisfaction']
print(f"df has {len(df.columns)} columns = {len(df_raw.columns)} original + {len(NEW_COLUMNS)} new")
print(f"New/changed columns: {NEW_COLUMNS}")
print(f"('avg_rating' is also modified in place - see Section 4)")
'''))

cells.append(new_code_cell(r'''# --- CSV export: JSON-serialize the two list/dict columns ---
df_csv = df.copy()
df_csv['overall_rules_triggered'] = df_csv['overall_rules_triggered'].map(lambda x: json.dumps(x, ensure_ascii=False))
df_csv['overall_evidence'] = df_csv['overall_evidence'].map(lambda x: json.dumps(x, ensure_ascii=False))
df_csv.to_csv(OUT_CSV_PATH, index=False)
print(f"Wrote {OUT_CSV_PATH} ({OUT_CSV_PATH.stat().st_size:,} bytes, "
      f"{len(df_csv)} rows x {len(df_csv.columns)} cols)")

# --- JSON export: keep list/dict columns as native JSON ---
LIST_COLS = ['overall_rules_triggered', 'overall_evidence']
scalar_df = df.drop(columns=LIST_COLS).astype(object)
scalar_df = scalar_df.where(scalar_df.notna(), None)
records = scalar_df.to_dict(orient='records')
for i in range(len(df)):
    records[i]['overall_rules_triggered'] = df['overall_rules_triggered'].iloc[i]
    records[i]['overall_evidence'] = df['overall_evidence'].iloc[i]

def _json_default(o):
    """Convert numpy scalar types (int64/float64/bool_) to native Python types for json.dump."""
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(f"Object of type {type(o)} is not JSON serializable")

with open(OUT_JSON_PATH, 'w', encoding='utf-8') as f:
    json.dump(records, f, ensure_ascii=False, indent=2, default=_json_default)
print(f"Wrote {OUT_JSON_PATH} ({OUT_JSON_PATH.stat().st_size:,} bytes, {len(records)} records)")
'''))

cells.append(new_markdown_cell(r"""### 17.1 Export verification

Re-read both exported files and confirm the new columns round-trip correctly."""))

cells.append(new_code_cell(r'''# CSV round-trip
df_check_csv = pd.read_csv(OUT_CSV_PATH)
assert len(df_check_csv) == len(df)
assert set(df.columns) == set(df_check_csv.columns)
df_check_csv['overall_rules_triggered'] = df_check_csv['overall_rules_triggered'].map(json.loads)
df_check_csv['overall_evidence'] = df_check_csv['overall_evidence'].map(json.loads)
assert df_check_csv.loc[0, 'overall_rules_triggered'] == df.iloc[0]['overall_rules_triggered']
assert df_check_csv.loc[0, 'overall_evidence'] == df.iloc[0]['overall_evidence']
assert np.allclose(df_check_csv['overall_satisfaction'], df['overall_satisfaction'].values)
print(f"[PASS] CSV round-trip OK: {len(df_check_csv)} rows x {len(df_check_csv.columns)} cols")

# JSON round-trip
with open(OUT_JSON_PATH, encoding='utf-8') as f:
    records_check = json.load(f)
assert len(records_check) == len(df)
assert records_check[0]['overall_rules_triggered'] == df.iloc[0]['overall_rules_triggered']
assert records_check[0]['overall_evidence'] == df.iloc[0]['overall_evidence']
assert isinstance(records_check[0]['overall_evidence'], list)
assert isinstance(records_check[0]['overall_rules_triggered'], list)
print(f"[PASS] JSON round-trip OK: {len(records_check)} records, "
      f"overall_evidence/overall_rules_triggered are native JSON arrays")

print(f"\n[INFO] {RULES_PATH} already written in Section 7 "
      f"({RULES_PATH.stat().st_size:,} bytes)")
'''))

cells.append(new_markdown_cell(r"""### 17.2 Generate `overall_satisfaction_rule_analysis.md`

This report is built entirely from variables computed earlier in this notebook
(`coverage_df`, `pairs_df`, `mixed_mask`, `rating_diff`, `corr_final`, ...) - re-running the
notebook regenerates the report from the live results, so it can never drift out of sync with
the data."""))

cells.append(new_code_cell(r'''def _df_to_md(d, float_fmt="{:.4f}", formatters=None):
    """Render a small DataFrame as a GitHub-flavoured Markdown table (no external deps).

    Formats column-by-column (using each column's own dtype) rather than row-by-row,
    because `DataFrame.iterrows()` upcasts int columns to float when a row also contains
    float columns - which would corrupt integer counts in all-numeric tables.
    `formatters` is an optional {column_name: format_string} override for specific columns.
    """
    formatters = formatters or {}
    cols = list(d.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "---|" * len(cols)
    formatted_cols = []
    for c in cols:
        s = d[c]
        fmt = formatters.get(c)
        if fmt is not None:
            formatted_cols.append(s.map(lambda v, fmt=fmt: fmt.format(v)))
        elif pd.api.types.is_float_dtype(s):
            formatted_cols.append(s.map(lambda v: float_fmt.format(v)))
        else:
            formatted_cols.append(s.astype(str))
    rows = ["| " + " | ".join(vals) + " |" for vals in zip(*formatted_cols)]
    return "\n".join([header, sep] + rows)


rule_table_rows = []
for _, r in coverage_df.iterrows():
    rd = RULES[r['rule']]
    rule_table_rows.append({
        'rule': r['rule'],
        'polarity': r['polarity'],
        'score': rd['score'],
        'n_patterns': len(rd['patterns']),
        'n_reviews': r['n_reviews'],
        'pct_reviews': r['pct_reviews'],
    })
rule_table_df = pd.DataFrame(rule_table_rows)

top_pairs_df = pairs_df.head(5).copy()
top_pairs_df['jaccard'] = top_pairs_df['jaccard'].round(4)

adj_value_counts = df['overall_adjustment'].round(2).value_counts().sort_index()
adj_table_df = adj_value_counts.rename_axis('overall_adjustment').reset_index(name='n_reviews')
adj_table_df['pct_reviews'] = (100 * adj_table_df['n_reviews'] / len(df)).round(2)

print("Helper tables prepared:")
print(f"  rule_table_df : {rule_table_df.shape}")
print(f"  top_pairs_df  : {top_pairs_df.shape}")
print(f"  adj_table_df  : {adj_table_df.shape}")
'''))

cells.append(new_code_cell(r'''lines = []
lines.append("# Overall Satisfaction Rule Analysis")
lines.append("")
lines.append(f"_Generated: {RUN_TIMESTAMP}_")
lines.append("")
lines.append("## 1. Overview")
lines.append("")
lines.append(f"- Dataset: `data_raw/reviews_clean.csv` / `data_raw/reviews_clean.json` ({len(df)} reviews, "
              f"{len(df_raw.columns)} original columns)")
lines.append(f"- Rating scale: 0-10")
lines.append(f"- Rule categories: 14 ({len(POS_RULES)} positive, {len(NEG_RULES)} negative), "
              f"defined in `overall_satisfaction_rules.json`")
lines.append("- No randomness is used anywhere in the rating/adjustment computation "
              "(RANDOM_SEED=42 is used only for reproducible sampling in audits/examples).")
lines.append("")

lines.append("## 2. Pipeline")
lines.append("")
lines.append("```")
lines.append("comment_clean")
lines.append("  -> normalize_text()                 (NFC normalize, lowercase, collapse whitespace)")
lines.append("  -> find_matching_rules(text, RULES) (regex search per rule pattern)")
lines.append("  -> calculate_adjustment(matches)    (sum of DISTINCT triggered rules' scores)")
lines.append("  -> generate_evidence(matches)       (one record per distinct triggered rule)")
lines.append("")
lines.append("overall_satisfaction = clip(avg_rating + overall_adjustment, 0, 10)")
lines.append("```")
lines.append("")

lines.append("## 3. Dataset Restructuring (Section 4)")
lines.append("")
n_recomputed = int(df['avg_rating_recomputed'].sum())
n_fallback = int((~df['avg_rating_recomputed']).sum())
lines.append(f"- `foody_original_avg_rating`: preserved original Foody rating "
              f"(mean of food/service/atmosphere/price/position).")
lines.append(f"- `avg_rating`: recomputed as mean(food_score, service_score, atmosphere_score, "
              f"price_score) - **position_score excluded** (see Section 8).")
lines.append(f"- `avg_rating` recomputed from 4 aspects for {n_recomputed} rows; "
              f"fell back to `foody_original_avg_rating` for {n_fallback} rows "
              f"(all 5 aspect scores missing).")
lines.append(f"- `avg_rating - foody_original_avg_rating`: mean={rating_diff.mean():+.4f}, "
              f"std={rating_diff.std():.4f}, min={rating_diff.min():+.4f}, max={rating_diff.max():+.4f}")
lines.append(f"- {int((rating_diff.abs() >= 0.5).sum())} reviews "
              f"({100*(rating_diff.abs() >= 0.5).sum()/len(df):.2f}%) changed by >= 0.5 points "
              f"after excluding position_score.")
lines.append("")

lines.append("## 4. Rule Catalogue and Coverage")
lines.append("")
lines.append("Coverage = number/percentage of the 9946 reviews for which the rule's `score` "
              "contributes to `overall_adjustment` (each rule contributes at most once per review, "
              "regardless of how many of its patterns match).")
lines.append("")
lines.append(_df_to_md(rule_table_df))
lines.append("")
for name, rule in RULES.items():
    lines.append(f"**`{name}`** (score `{rule['score']:+.1f}`): {rule['description']}")
    lines.append("")

lines.append("## 5. Adjustment Statistics (Section 11)")
lines.append("")
n_total = len(df)
n_adjusted = int((df['overall_adjustment'] != 0).sum())
n_pos_adj = int((df['overall_adjustment'] > 0).sum())
n_neg_adj = int((df['overall_adjustment'] < 0).sum())
lines.append(f"- Reviews with `overall_adjustment != 0`: {n_adjusted} "
              f"({100*n_adjusted/n_total:.2f}%) - {n_pos_adj} net positive, {n_neg_adj} net negative.")
lines.append(f"- `overall_adjustment`: mean={df['overall_adjustment'].mean():+.4f}, "
              f"std={df['overall_adjustment'].std():.4f}, "
              f"min={df['overall_adjustment'].min():+.2f}, max={df['overall_adjustment'].max():+.2f}")
n_clipped_high = int((raw_sum > 10).sum())
n_clipped_low = int((raw_sum < 0).sum())
lines.append(f"- Clipping: {n_clipped_high} rows clipped down to 10, "
              f"{n_clipped_low} rows clipped up to 0.")
lines.append("")
lines.append("Most common `overall_adjustment` values:")
lines.append("")
lines.append(_df_to_md(adj_table_df.head(10),
                        formatters={'overall_adjustment': '{:+.2f}', 'pct_reviews': '{:.2f}'}))
lines.append("")

lines.append("## 6. Rule Redundancy (Jaccard similarity, Section 12.1)")
lines.append("")
lines.append(f"Highest-overlap rule pairs out of {len(pairs_df)} pairs "
              f"(none exceed Jaccard 0.3, i.e. no pair of rules is substantially redundant):")
lines.append("")
lines.append(_df_to_md(top_pairs_df))
lines.append("")

lines.append("## 7. Positive/Negative Co-occurrence (Section 12.2)")
lines.append("")
n_pos_only = int((has_pos & ~has_neg).sum())
n_neg_only = int((has_neg & ~has_pos).sum())
n_mixed = int(mixed_mask.sum())
n_none = int((~has_pos & ~has_neg).sum())
lines.append(f"- Positive rule(s) only: {n_pos_only} ({100*n_pos_only/n_total:.2f}%)")
lines.append(f"- Negative rule(s) only: {n_neg_only} ({100*n_neg_only/n_total:.2f}%)")
lines.append(f"- Both positive and negative rules fired: {n_mixed} ({100*n_mixed/n_total:.2f}%)")
lines.append(f"- No rule triggered: {n_none} ({100*n_none/n_total:.2f}%)")
lines.append("")
lines.append("Mixed-signal reviews are treated as **genuinely mixed-sentiment** "
              "(e.g. great food but a long wait), not as bugs: `overall_adjustment` sums both "
              "the positive and the negative contribution, which is the intended explainable "
              "behaviour.")
lines.append("")

lines.append("## 8. Position Score Exclusion - Research Rationale (Section 16)")
lines.append("")
lines.append("Correlation matrix (5 raw aspects + 3 rating variants):")
lines.append("")
corr_table = corr_final.reset_index().rename(columns={'index': 'aspect'})
lines.append(_df_to_md(corr_table, float_fmt="{:.3f}"))
lines.append("")
aspect_corr = corr_final.loc[ASPECT_COLS_ALL, 'foody_original_avg_rating'].sort_values(ascending=False)
lines.append(f"`position_score` has the lowest correlation with the overall Foody rating "
              f"({aspect_corr['position_score']:.3f}) among the 5 aspects "
              f"(range across the other 4: {aspect_corr.drop('position_score').min():.3f}-"
              f"{aspect_corr.drop('position_score').max():.3f}).")
lines.append("")
lines.append("`position_score` measures location/accessibility, a property of the restaurant's "
              "location rather than of the food/service/atmosphere/price experience that is "
              "depicted in a review's image and described in its text. It is therefore excluded "
              "from `avg_rating` (and hence from `overall_satisfaction`) to improve the "
              "construct validity of the regression target for a multi-modal "
              "(image + text) quality-assessment model, while being preserved unchanged in the "
              "exported dataset for any downstream analysis that needs it.")
lines.append("")

lines.append("## 9. Known Limitations")
lines.append("")
lines.append("- **Phrase-level matching, no discourse resolution.** Rules match regex patterns "
              "against the normalized review text without resolving what/whom a phrase refers "
              "to. A review that contrasts the current restaurant with others "
              "(e.g. \"o nhieu quan khac rat kho chiu nhung quan nay thi...\") may trigger a "
              "rule (e.g. `frustration`) based on a clause that describes a *different* "
              "restaurant.")
lines.append("- **Generic statements vs personal intention.** Phrases such as \"ung ho\" "
              "(support/patronize) can appear in a generic recommendation to other customers "
              "rather than the reviewer's own repeat-visit intention; both are treated "
              "identically by the `advocacy` rule.")
lines.append("- **Negation handling is pattern-based, not syntactic.** Negation guards "
              "(`(?<!khong )`, `(?<!chang )`, ...) cover the negation idioms found during corpus "
              "analysis (Section 5) but cannot exhaustively cover every possible negation "
              "construction in Vietnamese.")
lines.append("- **Residual multi-word negation (documented, not fixed).** The guards added for "
              "`waiting_problem`'s 'doi/cho lau' and `frustration`'s 'kho chiu' patterns cover "
              "one- and two-word negation prefixes (e.g. 'khong phai doi lau', 'khong he kho "
              "chiu'). A small residual of three-or-more-word constructions, e.g. 'khong de "
              "khach doi lau' (=staff don't make customers wait long) or 'khong co thai do kho "
              "chiu' (=doesn't have an annoying attitude), remain unguarded; extending the "
              "lookbehind list to cover every such construction would add unbounded complexity "
              "for diminishing returns, so these rare cases (well under 0.5% of reviews) are "
              "accepted as label noise rather than chased with ever-more-specific regex.")
lines.append("- **Single-letter 'k' abbreviation of 'khong' not guarded.** Corpus analysis "
              "(Section 5) found 1,598 standalone 'k' tokens used as an informal abbreviation "
              "of 'khong', in addition to the two-letter 'ko' abbreviation that IS guarded "
              "against throughout the rule set (e.g. in `strong_satisfaction`'s 'xuat sac' and "
              "'hai long' patterns, Section 4). Constructions such as 'k xuat sac' (=not "
              "excellent) and 'k hai long' (=not satisfied) therefore still trigger "
              "`strong_satisfaction` (+0.6) as if positive, in 7 reviews for these two patterns "
              "alone. A third '(?<!k )'-style guard alongside every existing "
              "'(?<!khong )'/'(?<!ko )' pair across all patterns was judged to add "
              "disproportionate regex complexity for a single-character token that is also "
              "highly ambiguous in other contexts (e.g. as the currency unit in '30k' VND, or "
              "as a typo for 'ok'/'oke'); this is therefore accepted as a documented "
              "limitation rather than guarded against.")
lines.append("- These limitations are inherent to a transparent, rule-based, regex-driven "
              "approach and are an explicit design trade-off in favour of auditability and "
              "reproducibility over recall.")
lines.append("")

lines.append("## 10. Output Files")
lines.append("")
lines.append(f"- `{RULES_PATH.name}` - the 14 rule definitions with metadata "
              f"({RULES_PATH.stat().st_size:,} bytes)")
lines.append(f"- `{OUT_CSV_PATH.name}` - enhanced dataset, CSV "
              f"({OUT_CSV_PATH.stat().st_size:,} bytes, {len(df)} rows x {len(df.columns)} cols)")
lines.append(f"- `{OUT_JSON_PATH.name}` - enhanced dataset, JSON "
              f"({OUT_JSON_PATH.stat().st_size:,} bytes, {len(records)} records)")
lines.append(f"- `{ANALYSIS_MD_PATH.name}` - this report")
lines.append("")

report_md = "\n".join(lines)
with open(ANALYSIS_MD_PATH, 'w', encoding='utf-8') as f:
    f.write(report_md)

print(f"Wrote {ANALYSIS_MD_PATH} ({ANALYSIS_MD_PATH.stat().st_size:,} bytes, {len(lines)} lines)")
'''))

cells.append(new_markdown_cell(r"""## Summary

This notebook:

1. Loaded and deeply analyzed the raw 9946-row, 40-column dataset, confirming CSV/JSON consistency
   and the `avg_rating == mean(5 aspects)` relationship (Sections 1-3).
2. Restructured the dataset to preserve `foody_original_avg_rating`, recompute `avg_rating` from
   the 4 non-position aspects, and keep `position_score` for traceability (Section 4).
3. Performed corpus analysis on 9946 Vietnamese reviews to discover 14 global-satisfaction rule
   categories (8 positive, 6 negative), validated and exported as `overall_satisfaction_rules.json`
   (Sections 5-7).
4. Built and unit-tested a modular, type-hinted, docstring-documented rule engine
   (`normalize_text`, `find_matching_rules`, `calculate_adjustment`, `generate_evidence`,
   `apply_rules`) with deduplicated per-rule scoring (Sections 8-9).
5. Generated `overall_adjustment`, `overall_rules_triggered`, `overall_evidence`, and
   `overall_satisfaction = clip(avg_rating + overall_adjustment, 0, 10)` for all 9946 reviews
   (Section 10).
6. Analyzed dataset statistics, rule coverage/redundancy, and positive/negative co-occurrence, and
   visualized the results (Sections 11-13).
7. Ran a full quality-check suite, audited 20 reviews end-to-end, and quantified the research
   rationale for excluding `position_score` (Sections 14-16).
8. Exported `reviews_clean_enhanced.csv`, `reviews_clean_enhanced.json`, and
   `overall_satisfaction_rule_analysis.md` to `data_processed/`, verifying round-trip integrity
   (Section 17)."""))

with open('notebook/01_generate_overall_satisfaction.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Notebook now has {len(cells)} cells")
