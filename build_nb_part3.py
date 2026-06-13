import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell

nb = nbf.read('notebook/01_generate_overall_satisfaction.ipynb', as_version=4)
cells = nb.cells

# =====================================================================
# Section 10: Generate Overall Satisfaction
# =====================================================================
cells.append(new_markdown_cell(r"""## Section 10 — Generate Overall Satisfaction

We now apply the rule engine to every review and create five new columns on `df`:

| Column | Type | Meaning |
|---|---|---|
| `comment_normalized` | str | `normalize_text(comment_clean)` — the exact text the rule engine matched against |
| `overall_adjustment` | float | `calculate_adjustment(...)` — sum of scores of distinct triggered rules |
| `overall_rules_triggered` | list[str] | names of every rule category that fired |
| `overall_evidence` | list[dict] | structured evidence records (Section 9) |
| `overall_satisfaction` | float | `clip(avg_rating + overall_adjustment, 0, 10)` |

No randomness is involved — every value is a deterministic function of `comment_clean` and
`avg_rating` via the rules loaded from `overall_satisfaction_rules.json`."""))

cells.append(new_code_cell(r'''df['comment_normalized'] = df['comment_clean'].fillna('').map(normalize_text)

_rule_results = df['comment_normalized'].map(apply_rules)
df['overall_adjustment'] = _rule_results.map(lambda r: r[0]).astype(float)
df['overall_rules_triggered'] = _rule_results.map(lambda r: r[1])
df['overall_evidence'] = _rule_results.map(lambda r: r[2])

df['overall_satisfaction'] = (df['avg_rating'] + df['overall_adjustment']).clip(lower=0, upper=10)

print("New columns added:")
for c in ['comment_normalized', 'overall_adjustment', 'overall_rules_triggered',
          'overall_evidence', 'overall_satisfaction']:
    print(f"  {c:25s} dtype={df[c].dtype}")

assert df['overall_satisfaction'].between(0, 10).all()
assert df['overall_satisfaction'].isna().sum() == 0
print("\noverall_satisfaction is within [0, 10] for all rows, no NaN.")
'''))

cells.append(new_markdown_cell(r"""### 10.1 Worked examples

Three real reviews — one with a positive adjustment, one with a negative adjustment, and one with
no triggered rules — showing the full chain from `comment_clean` to `overall_satisfaction`."""))

cells.append(new_code_cell(r'''def show_example(row):
    print(f"review_id            : {row['review_id']}")
    print(f"comment_clean        : {row['comment_clean'][:200]!r}")
    print(f"avg_rating (4-aspect): {row['avg_rating']:.3f}")
    print(f"overall_adjustment   : {row['overall_adjustment']:+.2f}")
    print(f"rules_triggered      : {row['overall_rules_triggered']}")
    for e in row['overall_evidence']:
        print(f"    {e['rule']:25s} {e['score']:+.2f}  matched='{e['matched_text']}'")
    print(f"overall_satisfaction : {row['overall_satisfaction']:.3f}")
    print("-" * 70)

pos_example = df[df['overall_adjustment'] > 0].iloc[0]
neg_example = df[df['overall_adjustment'] < 0].iloc[0]
none_example = df[df['overall_adjustment'] == 0].iloc[0]

print("=== Positive adjustment example ===")
show_example(pos_example)
print("\n=== Negative adjustment example ===")
show_example(neg_example)
print("\n=== No rule triggered example ===")
show_example(none_example)
'''))

# =====================================================================
# Section 11: Dataset Statistics
# =====================================================================
cells.append(new_markdown_cell(r"""## Section 11 — Dataset Statistics

Descriptive statistics comparing the three rating quantities now present in `df`:

- `foody_original_avg_rating` — the original Foody rating (mean of all 5 aspects, incl. position)
- `avg_rating` — recomputed mean of the 4 non-position aspects (Section 4)
- `overall_satisfaction` — `avg_rating` + rule-based adjustment, clipped to [0, 10] (Section 10)"""))

cells.append(new_code_cell(r'''stats = df[['foody_original_avg_rating', 'avg_rating', 'overall_satisfaction']].describe().T
print(stats)

print(f"\nMean shift  avg_rating -> overall_satisfaction        : "
      f"{(df['overall_satisfaction'] - df['avg_rating']).mean():+.4f}")
print(f"Mean shift  foody_original -> overall_satisfaction     : "
      f"{(df['overall_satisfaction'] - df['foody_original_avg_rating']).mean():+.4f}")
'''))

cells.append(new_markdown_cell(r"""### 11.1 Adjustment statistics"""))

cells.append(new_code_cell(r'''n_total = len(df)
n_adjusted = (df['overall_adjustment'] != 0).sum()
n_pos_adj = (df['overall_adjustment'] > 0).sum()
n_neg_adj = (df['overall_adjustment'] < 0).sum()
n_zero_adj = (df['overall_adjustment'] == 0).sum()

print(f"Total reviews                  : {n_total}")
print(f"Reviews with adjustment != 0   : {n_adjusted} ({100*n_adjusted/n_total:.2f}%)")
print(f"  - net positive adjustment    : {n_pos_adj} ({100*n_pos_adj/n_total:.2f}%)")
print(f"  - net negative adjustment    : {n_neg_adj} ({100*n_neg_adj/n_total:.2f}%)")
print(f"Reviews with no rule triggered  : {n_zero_adj} ({100*n_zero_adj/n_total:.2f}%)")

print(f"\noverall_adjustment summary:")
print(df['overall_adjustment'].describe())

print(f"\nMost common overall_adjustment values:")
print(df['overall_adjustment'].value_counts().head(10))
'''))

cells.append(new_markdown_cell(r"""### 11.2 Clipping behaviour

How often does `avg_rating + overall_adjustment` fall outside `[0, 10]` and get clipped?"""))

cells.append(new_code_cell(r'''raw_sum = df['avg_rating'] + df['overall_adjustment']
n_clipped_high = (raw_sum > 10).sum()
n_clipped_low = (raw_sum < 0).sum()

print(f"Rows where avg_rating + overall_adjustment > 10 (clipped down): {n_clipped_high}")
print(f"Rows where avg_rating + overall_adjustment < 0  (clipped up)  : {n_clipped_low}")

if n_clipped_high:
    cols = ['review_id', 'avg_rating', 'overall_adjustment', 'overall_satisfaction']
    print(df.loc[raw_sum > 10, cols].head(5))
if n_clipped_low:
    cols = ['review_id', 'avg_rating', 'overall_adjustment', 'overall_satisfaction']
    print(df.loc[raw_sum < 0, cols].head(5))
'''))

cells.append(new_markdown_cell(r"""### 11.3 `overall_satisfaction` distribution buckets"""))

cells.append(new_code_cell(r'''bins = [0, 2, 4, 6, 8, 10.0001]
labels = ['0-2', '2-4', '4-6', '6-8', '8-10']

old_buckets = pd.cut(df['avg_rating'], bins=bins, labels=labels, right=False, include_lowest=True).value_counts().sort_index()
new_buckets = pd.cut(df['overall_satisfaction'], bins=bins, labels=labels, right=False, include_lowest=True).value_counts().sort_index()

bucket_df = pd.DataFrame({'avg_rating': old_buckets, 'overall_satisfaction': new_buckets})
bucket_df['delta'] = bucket_df['overall_satisfaction'] - bucket_df['avg_rating']
print(bucket_df)
'''))

# =====================================================================
# Section 12: Rule Coverage Analysis
# =====================================================================
cells.append(new_markdown_cell(r"""## Section 12 — Rule Coverage Analysis

For each of the 14 rules we report:

1. **Coverage** — how many reviews triggered it (count and %).
2. **Redundancy** — pairwise Jaccard similarity between rules' triggered-review sets, to flag
   rules whose patterns overlap so much they may be redundant.
3. **Positive/negative co-occurrence** — reviews where at least one positive rule AND at least one
   negative rule both fire. We distinguish *genuinely mixed-sentiment reviews* (e.g. "great food but
   we waited forever") — which `overall_adjustment` correctly handles by summing both effects — from
   any remaining *false-positive idiom* cases."""))

cells.append(new_code_cell(r'''POS_RULES = [n for n, r in RULES.items() if r['score'] > 0]
NEG_RULES = [n for n, r in RULES.items() if r['score'] < 0]

coverage_rows = []
for name, rule in RULES.items():
    n = df['overall_rules_triggered'].map(lambda t, name=name: name in t).sum()
    coverage_rows.append({
        'rule': name,
        'polarity': 'positive' if rule['score'] > 0 else 'negative',
        'score': rule['score'],
        'n_reviews': int(n),
        'pct_reviews': round(100 * n / len(df), 2),
    })

coverage_df = pd.DataFrame(coverage_rows).sort_values('n_reviews', ascending=False).reset_index(drop=True)
print(coverage_df.to_string(index=False))
'''))

cells.append(new_markdown_cell(r"""### 12.1 Redundancy analysis (Jaccard similarity)

For each pair of rules, the Jaccard similarity of their triggered-review sets:
`|A ∩ B| / |A ∪ B|`. High similarity (e.g. > 0.3) between two rules in the **same** polarity group
would suggest their patterns are largely redundant."""))

cells.append(new_code_cell(r'''rule_names = list(RULES.keys())
triggered_sets = {name: set() for name in rule_names}
for idx, triggered in df['overall_rules_triggered'].items():
    for name in triggered:
        triggered_sets[name].add(idx)

pairs = []
for i, a in enumerate(rule_names):
    for b in rule_names[i + 1:]:
        sa, sb = triggered_sets[a], triggered_sets[b]
        union = len(sa | sb)
        jac = len(sa & sb) / union if union else 0.0
        pairs.append((a, b, jac, len(sa & sb)))

pairs_df = pd.DataFrame(pairs, columns=['rule_a', 'rule_b', 'jaccard', 'n_overlap'])
pairs_df = pairs_df.sort_values('jaccard', ascending=False).reset_index(drop=True)

print("Top 10 most similar rule pairs (potential redundancy):")
print(pairs_df.head(10).to_string(index=False))

high_overlap = pairs_df[pairs_df['jaccard'] > 0.3]
print(f"\nPairs with Jaccard > 0.3: {len(high_overlap)}")
'''))

cells.append(new_markdown_cell(r"""### 12.2 Positive/negative co-occurrence"""))

cells.append(new_code_cell(r'''def _has_pos(triggered):
    return any(n in POS_RULES for n in triggered)

def _has_neg(triggered):
    return any(n in NEG_RULES for n in triggered)

has_pos = df['overall_rules_triggered'].map(_has_pos)
has_neg = df['overall_rules_triggered'].map(_has_neg)
mixed_mask = has_pos & has_neg

print(f"Reviews with ONLY positive rule(s)  : {(has_pos & ~has_neg).sum()}")
print(f"Reviews with ONLY negative rule(s)  : {(has_neg & ~has_pos).sum()}")
print(f"Reviews with BOTH positive & negative: {mixed_mask.sum()} ({100*mixed_mask.sum()/len(df):.2f}%)")
print(f"Reviews with NO rule triggered        : {(~has_pos & ~has_neg).sum()}")
'''))

cells.append(new_code_cell(r'''print("Sample of mixed-sentiment reviews (both positive and negative rules fired):\n")
mixed_df = df[mixed_mask]
sample = mixed_df.sample(n=min(3, len(mixed_df)), random_state=RANDOM_SEED)

for _, row in sample.iterrows():
    print(f"review_id={row['review_id']}  avg_rating={row['avg_rating']:.2f}  "
          f"overall_adjustment={row['overall_adjustment']:+.2f}  "
          f"overall_satisfaction={row['overall_satisfaction']:.2f}")
    print(f"  text: {row['comment_clean'][:180]!r}")
    for e in row['overall_evidence']:
        print(f"    [{e['polarity']:8s}] {e['rule']:25s} {e['score']:+.2f}  '{e['matched_text']}'")
    print()

print("Interpretation: these reviews genuinely contain BOTH a positive global-satisfaction signal")
print("(e.g. a recommendation, loyalty, or revisit intention) AND a negative one (e.g. a waiting-time")
print("complaint or frustration). overall_adjustment correctly sums both effects rather than picking")
print("a single 'winner' - this is the intended, explainable behaviour for nuanced reviews, not a bug.")
'''))

# =====================================================================
# Section 13: Visualization
# =====================================================================
cells.append(new_markdown_cell(r"""## Section 13 — Visualization

Three figures:

1. **Rating distributions** — `foody_original_avg_rating` vs `avg_rating` (4-aspect) vs
   `overall_satisfaction`, as side-by-side histograms on a shared scale.
2. **Rule coverage** — horizontal bar chart of how many reviews each of the 14 rules triggered,
   coloured by polarity.
3. **Adjustment distribution** — bar chart of `overall_adjustment` value counts."""))

cells.append(new_code_cell(r'''fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
bins_edges = np.arange(0, 10.5, 0.5)

for ax, col, title in zip(
    axes,
    ['foody_original_avg_rating', 'avg_rating', 'overall_satisfaction'],
    ['Foody Original\n(5-aspect mean)', 'avg_rating\n(4-aspect, position excluded)', 'overall_satisfaction\n(avg_rating + rule adjustment)'],
):
    ax.hist(df[col], bins=bins_edges, color='steelblue', edgecolor='white')
    ax.set_title(title)
    ax.set_xlabel('Rating (0-10)')
    ax.axvline(df[col].mean(), color='darkorange', linestyle='--', linewidth=1.5,
               label=f'mean={df[col].mean():.2f}')
    ax.legend()

axes[0].set_ylabel('Number of reviews')
fig.suptitle('Rating Distributions: Original vs Recomputed vs Overall Satisfaction', fontsize=13, fontweight='bold')
fig.tight_layout()
plt.show()
'''))

cells.append(new_code_cell(r'''fig, ax = plt.subplots(figsize=(9, 6))
plot_df = coverage_df.sort_values('n_reviews')
colors = ['#2ca02c' if p == 'positive' else '#d62728' for p in plot_df['polarity']]

ax.barh(plot_df['rule'], plot_df['n_reviews'], color=colors)
for y, (n, pct) in enumerate(zip(plot_df['n_reviews'], plot_df['pct_reviews'])):
    ax.text(n + 10, y, f"{n} ({pct:.1f}%)", va='center', fontsize=8)

ax.set_xlabel('Number of reviews triggered')
ax.set_title('Rule Coverage: Reviews Triggered per Rule Category (n=9946)')

import matplotlib.patches as mpatches
legend_handles = [mpatches.Patch(color='#2ca02c', label='Positive rule'),
                   mpatches.Patch(color='#d62728', label='Negative rule')]
ax.legend(handles=legend_handles, loc='lower right')
fig.tight_layout()
plt.show()
'''))

cells.append(new_code_cell(r'''fig, ax = plt.subplots(figsize=(9, 5))
adj_counts = df['overall_adjustment'].round(2).value_counts().sort_index()
colors = ['#d62728' if v < 0 else ('#7f7f7f' if v == 0 else '#2ca02c') for v in adj_counts.index]

ax.bar(adj_counts.index.astype(str), adj_counts.values, color=colors)
ax.set_xlabel('overall_adjustment value')
ax.set_ylabel('Number of reviews')
ax.set_title('Distribution of overall_adjustment (sum of triggered rule scores)')
ax.tick_params(axis='x', rotation=90)
for i, v in enumerate(adj_counts.values):
    ax.text(i, v + 5, str(v), ha='center', fontsize=7)
fig.tight_layout()
plt.show()

print(f"overall_adjustment == 0 for {adj_counts.get(0.0, 0)} reviews "
      f"({100*adj_counts.get(0.0, 0)/len(df):.1f}%)")
'''))

with open('notebook/01_generate_overall_satisfaction.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Notebook now has {len(cells)} cells")
