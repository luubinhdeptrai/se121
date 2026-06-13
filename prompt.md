# ROLE

You are a Principal AI Research Engineer, Senior Data Scientist, Machine Learning Engineer, Data Engineer, and Research Scientist specializing in:

- Natural Language Processing (NLP)
- Sentiment Analysis
- Aspect-Based Sentiment Analysis (ABSA)
- Multi-modal Deep Learning
- Explainable AI (XAI)
- Dataset Engineering
- Data Annotation Pipelines
- Python
- Pandas
- Jupyter Notebook
- Reproducible Research
- Research Methodology

You are also an experienced thesis advisor and reviewer who follows strict academic standards regarding auditability, reproducibility, explainability, traceability, data provenance, and research validity.

---

# GOAL

Create a complete, production-quality Jupyter Notebook:

`01_generate_overall_satisfaction.ipynb`

that performs:

1. Read and deeply analyze @reviews_clean.csv and @reviews_clean.json
2. Dataset restructuring
3. Position aspect exclusion from modeling
4. Avg rating recomputation
5. Corpus analysis
6. Rule discovery
7. Rule generation
8. Overall satisfaction generation
9. Dataset auditing
10. Statistical analysis
11. Dataset export

---

# MANDATORY DATASET READING REQUIREMENT

Before writing transformation logic, rule logic, or export logic, you MUST carefully read and inspect BOTH dataset files:

- @reviews_clean.csv
- @reviews_clean.json

You must not assume the schema.

You must analyze:

- all columns
- data types
- missing values
- duplicate records
- rating distributions
- text length distribution
- common Vietnamese review phrases
- aspect score distributions
- image-related fields
- relationship between `avg_rating` and aspect scores
- consistency between CSV and JSON

Generate a **Dataset Understanding Report** before any transformation.

---

# PROJECT CONTEXT

I am developing a thesis project:

**"Explainable Multi-modal Deep Learning System for Product Quality Assessment using Image and Text Data"**

Current aspect ratings:

- food_score
- service_score
- atmosphere_score
- price_score
- position_score

Current Foody overall rating:

- avg_rating

The original Foody `avg_rating` is derived from aspect ratings, so it is not an independent overall satisfaction label.

---

# IMPORTANT DATASET RESTRUCTURING REQUIREMENT

The Position aspect is intentionally excluded from this research.

Reason:

Position convenience is weakly observable from review images and is often not explicitly discussed in review text.

Therefore, `position_score` introduces label noise and reduces explainability quality.

Do NOT delete `position_score`.

Preserve it for traceability, but exclude it from:

- avg_rating computation
- training targets
- overall_satisfaction generation
- evaluation metrics
- target visualizations

---

# REQUIRED RATING TRANSFORMATION

Preserve original Foody avg_rating as:

`foody_original_avg_rating`

Then recompute:

```text
avg_rating =
(
food_score +
service_score +
atmosphere_score +
price_score
) / 4
```

All subsequent processing must use the recomputed `avg_rating`.

Final training targets:

- food_score
- service_score
- atmosphere_score
- price_score
- overall_satisfaction

Excluded target:

- position_score

---

# DATASET-DRIVEN RULE DISCOVERY

Before generating `overall_satisfaction_rules.json`, analyze the review corpus from the real dataset.

Distinguish:

## Aspect-specific expressions

These should generally NOT modify overall_satisfaction:

- món ngon
- đồ ăn ngon
- phục vụ tốt
- nhân viên thân thiện
- không gian đẹp
- giá rẻ

## Global satisfaction expressions

These MAY modify overall_satisfaction:

- sẽ quay lại
- chắc chắn sẽ quay lại
- không quay lại nữa
- không bao giờ quay lại
- rất hài lòng
- cực kỳ hài lòng
- thất vọng
- không có gì để chê
- đáng để thử
- quán ruột
- đi như cơm bữa
- giới thiệu cho bạn bè
- ủng hộ dài lâu

---

# REQUIRED RULE FILE

Create:

`overall_satisfaction_rules.json`

The rule file must be generated from:

- corpus analysis
- expert-designed rules
- manual review

Each rule must contain:

```json
{
  "rule_name": {
    "score": 0.5,
    "description": "...",
    "patterns": [...]
  }
}
```

Required categories:

Positive:

1. revisit_intention
2. repeat_purchase
3. recommendation
4. value_for_money
5. strong_satisfaction
6. no_complaint
7. loyalty
8. advocacy

Negative:

1. no_revisit
2. strong_dissatisfaction
3. not_worth_it
4. waiting_problem
5. frustration
6. regret

Rules must be loaded dynamically, not hardcoded into the rule engine.

---

# REQUIRED NEW COLUMNS

Append these fields to BOTH CSV and JSON outputs:

- overall_adjustment
- overall_satisfaction
- overall_rules_triggered
- overall_evidence

For CSV, store `overall_evidence` as a serialized JSON string.

For JSON, store `overall_evidence` as a structured JSON array.

---

# REQUIRED COMPUTATION

```text
overall_adjustment = sum(all matched rule scores)

overall_satisfaction =
clip(
    avg_rating + overall_adjustment,
    0,
    10
)
```

Every adjustment must be:

- traceable
- explainable
- reproducible

---

# REQUIRED NOTEBOOK STRUCTURE

## 1. Environment Setup

- imports
- configuration
- random seed

## 2. Load and Understand Dataset

Load:

- reviews_clean.csv
- reviews_clean.json

Validate:

- file existence
- schema
- duplicates
- missing values
- data consistency between CSV and JSON

Generate Dataset Understanding Report.

## 3. Rating and Aspect Analysis

Analyze:

- foody original avg_rating
- food_score
- service_score
- atmosphere_score
- price_score
- position_score

Show distributions and summary statistics.

## 4. Dataset Restructuring

Create:

- foody_original_avg_rating

Recompute:

- avg_rating from 4 aspects only

Generate comparison statistics:

- average difference
- max difference
- min difference
- distribution shift
- affected review percentage

## 5. Corpus Analysis

Analyze:

- review length distribution
- common phrases
- common positive phrases
- common negative phrases
- Vietnamese satisfaction signals

## 6. Generate Rule Candidates

Discover candidate patterns from the dataset.

Separate:

- positive global satisfaction signals
- negative global satisfaction signals
- aspect-specific expressions

## 7. Create and Validate overall_satisfaction_rules.json

Generate rules.

Validate schema.

Display rule summary.

## 8. Text Normalization

Implement:

- lowercase
- unicode normalization
- whitespace cleanup
- punctuation cleanup

## 9. Rule Engine

Implement reusable functions:

- normalize_text()
- find_matching_rules()
- calculate_adjustment()
- generate_evidence()

Use:

- modular design
- type hints
- docstrings

## 10. Generate Overall Satisfaction

Generate:

- overall_adjustment
- overall_satisfaction
- overall_rules_triggered
- overall_evidence

for every review.

## 11. Dataset Statistics

Compute:

- total reviews
- adjusted reviews
- unchanged reviews
- positive adjustments
- negative adjustments
- average adjustment
- max adjustment
- min adjustment

## 12. Rule Coverage Analysis

For every rule, compute:

- trigger count
- percentage coverage
- most common matched phrases

Identify:

- unused rules
- redundant rules
- conflicting rules

## 13. Visualization

Create publication-quality charts:

- original Foody avg_rating distribution
- recomputed avg_rating distribution
- overall_satisfaction distribution
- adjustment distribution
- rule frequency distribution
- top rule coverage

## 14. Quality Checks

Verify:

- no score above 10
- no score below 0
- no NaN values
- evidence consistency
- JSON validity
- CSV validity

## 15. Sample Auditing

Display 20 adjusted reviews with:

- review text
- foody_original_avg_rating
- recomputed avg_rating
- overall_adjustment
- overall_satisfaction
- triggered rules
- evidence

## 16. Research Analysis

Generate a section:

**Impact of Removing Position from Rating Computation**

Discuss:

- why Position was excluded
- statistical impact
- rating shifts
- implications for model training
- implications for explainability

## 17. Export Results

Save:

- reviews_clean_enhanced.csv
- reviews_clean_enhanced.json
- overall_satisfaction_rules.json
- overall_satisfaction_rule_analysis.md

Do NOT overwrite original files.

---

# CONSTRAINTS

Use:

- Python
- Pandas
- NumPy
- JSON
- Matplotlib

Do NOT:

- overwrite original files
- modify original Foody ratings
- use random score generation
- use undocumented heuristics
- use black-box score modification

Every score adjustment must be explainable.

---

# OUTPUT FORMAT

Generate:

1. Complete notebook source code.
2. Complete `overall_satisfaction_rules.json`.
3. Complete `overall_satisfaction_rule_analysis.md`.
4. Folder structure.
5. Example outputs.
6. Dataset schema documentation.
7. Rule documentation.

---

# SELF-REVIEW PROCESS

After generating the notebook, perform a complete engineering review.

Check:

1. Dataset reading correctness.
2. Dataset understanding completeness.
3. Dataset restructuring correctness.
4. Avg rating recomputation correctness.
5. Rule correctness.
6. CSV export correctness.
7. JSON export correctness.
8. Evidence generation correctness.
9. Auditability.
10. Reproducibility.
11. Maintainability.
12. Research validity.

Then fix every issue found.

Repeat:

review → fix → review → fix

until the notebook is:

- production-ready
- research-ready
- audit-ready
- thesis-ready
- publication-ready
