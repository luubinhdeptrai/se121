# Overall Satisfaction Rule Analysis

_Generated: 2026-06-13T07:26:05.628291+00:00_

## 1. Overview

- Dataset: `data_raw/reviews_clean.csv` / `data_raw/reviews_clean.json` (9946 reviews, 40 original columns)
- Rating scale: 0-10
- Rule categories: 14 (8 positive, 6 negative), defined in `overall_satisfaction_rules.json`
- No randomness is used anywhere in the rating/adjustment computation (RANDOM_SEED=42 is used only for reproducible sampling in audits/examples).

## 2. Pipeline

```
comment_clean
  -> normalize_text()                 (NFC normalize, lowercase, collapse whitespace)
  -> find_matching_rules(text, RULES) (regex search per rule pattern)
  -> calculate_adjustment(matches)    (sum of DISTINCT triggered rules' scores)
  -> generate_evidence(matches)       (one record per distinct triggered rule)

overall_satisfaction = clip(avg_rating + overall_adjustment, 0, 10)
```

## 3. Dataset Restructuring (Section 4)

- `foody_original_avg_rating`: preserved original Foody rating (mean of food/service/atmosphere/price/position).
- `avg_rating`: recomputed as mean(food_score, service_score, atmosphere_score, price_score) - **position_score excluded** (see Section 8).
- `avg_rating` recomputed from 4 aspects for 9943 rows; fell back to `foody_original_avg_rating` for 3 rows (all 5 aspect scores missing).
- `avg_rating - foody_original_avg_rating`: mean=-0.0655, std=0.2780, min=-1.8000, max=+1.3500
- 1044 reviews (10.50%) changed by >= 0.5 points after excluding position_score.

## 4. Rule Catalogue and Coverage

Coverage = number/percentage of the 9946 reviews for which the rule's `score` contributes to `overall_adjustment` (each rule contributes at most once per review, regardless of how many of its patterns match).

| rule | polarity | score | n_patterns | n_reviews | pct_reviews |
|---|---|---|---|---|---|
| strong_satisfaction | positive | 0.6000 | 8 | 669 | 6.7300 |
| strong_dissatisfaction | negative | -0.6000 | 8 | 666 | 6.7000 |
| revisit_intention | positive | 0.5000 | 5 | 552 | 5.5500 |
| recommendation | positive | 0.4000 | 5 | 523 | 5.2600 |
| no_revisit | negative | -0.5000 | 5 | 331 | 3.3300 |
| advocacy | positive | 0.5000 | 4 | 330 | 3.3200 |
| frustration | negative | -0.4000 | 5 | 289 | 2.9100 |
| repeat_purchase | positive | 0.4000 | 5 | 195 | 1.9600 |
| waiting_problem | negative | -0.4000 | 6 | 148 | 1.4900 |
| loyalty | positive | 0.5000 | 5 | 113 | 1.1400 |
| no_complaint | positive | 0.3000 | 3 | 73 | 0.7300 |
| value_for_money | positive | 0.3000 | 3 | 66 | 0.6600 |
| regret | negative | -0.4000 | 4 | 27 | 0.2700 |
| not_worth_it | negative | -0.5000 | 5 | 22 | 0.2200 |

**`revisit_intention`** (score `+0.5`): Reviewer explicitly expresses intention to return or visit the restaurant again in the future. This is a global satisfaction signal independent of any single aspect (food/service/atmosphere/price). The '(quay lai|tro lai|ghe lai) lan (sau|toi|nua)' pattern additionally carries '(khong|chang|ko) bao gio' guards because corpus analysis found '(khong|ko) bao gio quay lai lan nua' (=will never come back again) is a common NEGATIVE idiom that would otherwise be misclassified as a revisit intention. The 'se (quay lai|...)' pattern carries '(khong|ko) nghi' guards because corpus analysis found '(khong|ko) nghi se quay lai' (=don't think [I'll] come back) is a NEGATIVE idiom that would otherwise be misclassified as a revisit intention.

**`repeat_purchase`** (score `+0.4`): Reviewer indicates they will order, buy, or eat the same item again (repeat-purchase intention for the product/dish, distinct from revisiting the venue itself). All patterns carry '(khong|chang|ko) bao gio' guards because corpus analysis found '(khong|chac chan khong|ko) bao gio an/mua/dat lai' (=will never order/buy/eat again) are common NEGATIVE idioms that would otherwise be misclassified as repeat-purchase intention.

**`recommendation`** (score `+0.4`): Reviewer recommends the restaurant or its food/service to other people - a word-of-mouth endorsement of the overall experience.

**`value_for_money`** (score `+0.3`): Reviewer explicitly states the price paid is worth the overall value/quality received. This is a global price-value judgment, distinct from aspect-specific price comments such as 'gia re' or 'gia hop ly' which are excluded from this rule set.

**`strong_satisfaction`** (score `+0.6`): Reviewer expresses strong overall satisfaction or delight with the experience as a whole, not tied to a single aspect (food/service/atmosphere/price). 'xuat sac'/'hoan hao' patterns carry extra lookbehind guards for hedged-negation idioms such as 'khong qua xuat sac' (=not too excellent) discovered during corpus analysis. The bare 'hai long' pattern additionally carries '(?<!ko )', '(?<!khong duoc )', '(?<!ko duoc )', '(?<!khong thay )' and '(?<!chua lam )' guards because corpus analysis found 'ko hai long' (=not satisfied) and '(khong|ko) duoc hai long' (=wasn't satisfied) are common NEGATIVE idioms that would otherwise be misclassified as strong satisfaction. Every 'khong X' guard on 'xuat sac' is mirrored by a 'ko X' guard (and 'tuyet voi' gains a 'khong/ko ngon' guard) because 'ko' is the same informal abbreviation of 'khong' found during corpus analysis of the 'hai long' pattern, e.g. 'ko qua xuat sac' (=not too excellent) and 'ko ngon tuyet voi' (=not wonderfully delicious) are common hedged-negation idioms.

**`no_complaint`** (score `+0.3`): Reviewer explicitly states they have no complaints or issues with the overall experience. The leading 'khong'/'chang'/'ko' here is part of the positive idiom itself, so it is intentionally not guarded against; the first pattern's '(khong|ko)' alternation adds the same 'ko'-abbreviation coverage used throughout this rule set (e.g. 'ko co gi de che', 19 additional reviews). A trailing word boundary on 'che' in all three patterns fixes a pre-existing bug where 'che' matched as a substring-prefix of the unrelated word 'chenh' (=differ/vary), e.g. 'khong chenh lech nhieu' (=doesn't vary much, a price-consistency remark unrelated to complaints) was previously misclassified as no_complaint (3 reviews).

**`loyalty`** (score `+0.5`): Reviewer identifies as a regular or loyal customer / long-term patron of the restaurant.

**`advocacy`** (score `+0.5`): Reviewer actively promotes or endorses the restaurant to others, going beyond a simple recommendation (active word-of-mouth advocacy / support for the business). The 'ung ho' pattern requires a leading word boundary because corpus analysis found it otherwise false-matches as a substring of unrelated words such as 'lung hop' (=the box leaks/breaks, about packaging); it also carries '(khong|ko) muon' guards because '(khong|ko) muon ung ho' (=don't want to support) is a NEGATIVE idiom that would otherwise be misclassified as advocacy.

**`no_revisit`** (score `-0.5`): Reviewer explicitly states they will not return to the restaurant in the future. The leading '(khong|ko)' alternation in the first and fourth patterns mirrors the 'ko'-abbreviation handling used throughout this rule set (see strong_satisfaction): corpus analysis found 'ko bao gio quay lai' / 'ko nen an o day' are common informal spellings of 'khong bao gio quay lai' / 'khong nen an o day' that the bare 'khong'-only patterns missed (76 additional reviews corpus-wide).

**`strong_dissatisfaction`** (score `-0.6`): Reviewer expresses strong overall dissatisfaction or disappointment with the experience as a whole, not tied to a single aspect. The 'that vong' and 'te' patterns carry extended lookbehind guards for common reversal idioms ('se khong lam ban that vong' = won't disappoint you; 'khong qua te' = not too bad) discovered during corpus analysis. A dedicated '(khong|ko) hai long' (=not satisfied) pattern is included because it is a direct negative-satisfaction statement, not a hedge of a positive word; the '(khong|ko)' alternation is the same 'ko'-abbreviation found during corpus analysis of strong_satisfaction's 'hai long' guard ('ko hai long' is the exact idiom that pattern guards AGAINST as a positive - here it is the trigger FOR strong_dissatisfaction, adding 14 reviews that previously triggered no rule at all).

**`not_worth_it`** (score `-0.5`): Reviewer states the price paid was not worth it relative to the overall quality received - a global price-value judgment (negative counterpart of value_for_money). The second pattern's '(khong|ko)' alternation adds the same 'ko'-abbreviation coverage used throughout this rule set (e.g. 'ko dang dong tien', 1 additional review).

**`waiting_problem`** (score `-0.4`): Reviewer reports significant waiting-time issues, indicating slow overall service delivery (not a specific food/service quality complaint, but an overall experience friction). The 'cho/doi (rat|qua)? lau' patterns carry '(khong|ko) (phai|bi|can|de)?', 'chua (phai)?' and 'dung' guards because corpus analysis found '(khong|ko) phai doi/cho (qua)? lau' (=don't have to wait long) is a common POSITIVE idiom that would otherwise be misclassified as a waiting-time problem.

**`frustration`** (score `-0.4`): Reviewer expresses frustration, anger, or annoyance with the overall experience. The 'kho chiu' pattern carries '(khong|chang|chua|ko)', '(khong|ko) he' and 'khong bi/thay' guards because corpus analysis found '(khong|ko) (he) kho chiu' (=not at all annoying) is a common POSITIVE idiom that would otherwise be misclassified as frustration.

**`regret`** (score `-0.4`): Reviewer expresses regret about having visited or spent money/time at the restaurant. 'hoi han' and 'uong cong/tien' carry negation guards because corpus analysis found 'khong uong cong' (=not a waste of effort, i.e. well worth it) and 'khong (chut) hoi han' (=no regrets) are common POSITIVE idioms that would otherwise be misclassified as regret.

## 5. Adjustment Statistics (Section 11)

- Reviews with `overall_adjustment != 0`: 3263 (32.81%) - 2058 net positive, 1205 net negative.
- `overall_adjustment`: mean=+0.0469, std=0.3401, min=-1.50, max=+2.00
- Clipping: 432 rows clipped down to 10, 26 rows clipped up to 0.

Most common `overall_adjustment` values:

| overall_adjustment | n_reviews | pct_reviews |
|---|---|---|
| -1.50 | 11 | 0.11 |
| -1.40 | 1 | 0.01 |
| -1.30 | 2 | 0.02 |
| -1.10 | 62 | 0.62 |
| -1.00 | 49 | 0.49 |
| -0.90 | 24 | 0.24 |
| -0.80 | 6 | 0.06 |
| -0.70 | 4 | 0.04 |
| -0.60 | 439 | 4.41 |
| -0.50 | 228 | 2.29 |

## 6. Rule Redundancy (Jaccard similarity, Section 12.1)

Highest-overlap rule pairs out of 91 pairs (none exceed Jaccard 0.3, i.e. no pair of rules is substantially redundant):

| rule_a | rule_b | jaccard | n_overlap |
|---|---|---|---|
| no_revisit | strong_dissatisfaction | 0.0825 | 76 |
| no_revisit | frustration | 0.0580 | 34 |
| strong_dissatisfaction | frustration | 0.0576 | 52 |
| revisit_intention | strong_satisfaction | 0.0562 | 65 |
| recommendation | strong_satisfaction | 0.0447 | 51 |

## 7. Positive/Negative Co-occurrence (Section 12.2)

- Positive rule(s) only: 2015 (20.26%)
- Negative rule(s) only: 1097 (11.03%)
- Both positive and negative rules fired: 191 (1.92%)
- No rule triggered: 6643 (66.79%)

Mixed-signal reviews are treated as **genuinely mixed-sentiment** (e.g. great food but a long wait), not as bugs: `overall_adjustment` sums both the positive and the negative contribution, which is the intended explainable behaviour.

## 8. Position Score Exclusion - Research Rationale (Section 16)

Correlation matrix (5 raw aspects + 3 rating variants):

| aspect | food_score | service_score | atmosphere_score | price_score | position_score | foody_original_avg_rating | avg_rating | overall_satisfaction |
|---|---|---|---|---|---|---|---|---|
| food_score | 1.000 | 0.835 | 0.760 | 0.828 | 0.756 | 0.923 | 0.934 | 0.933 |
| service_score | 0.835 | 1.000 | 0.811 | 0.791 | 0.762 | 0.926 | 0.936 | 0.932 |
| atmosphere_score | 0.760 | 0.811 | 1.000 | 0.752 | 0.808 | 0.903 | 0.897 | 0.886 |
| price_score | 0.828 | 0.791 | 0.752 | 1.000 | 0.769 | 0.910 | 0.915 | 0.906 |
| position_score | 0.756 | 0.762 | 0.808 | 0.769 | 1.000 | 0.892 | 0.839 | 0.828 |
| foody_original_avg_rating | 0.923 | 0.926 | 0.903 | 0.910 | 0.892 | 1.000 | 0.994 | 0.987 |
| avg_rating | 0.934 | 0.936 | 0.897 | 0.915 | 0.839 | 0.994 | 1.000 | 0.994 |
| overall_satisfaction | 0.933 | 0.932 | 0.886 | 0.906 | 0.828 | 0.987 | 0.994 | 1.000 |

`position_score` has the lowest correlation with the overall Foody rating (0.892) among the 5 aspects (range across the other 4: 0.903-0.926).

`position_score` measures location/accessibility, a property of the restaurant's location rather than of the food/service/atmosphere/price experience that is depicted in a review's image and described in its text. It is therefore excluded from `avg_rating` (and hence from `overall_satisfaction`) to improve the construct validity of the regression target for a multi-modal (image + text) quality-assessment model, while being preserved unchanged in the exported dataset for any downstream analysis that needs it.

## 9. Known Limitations

- **Phrase-level matching, no discourse resolution.** Rules match regex patterns against the normalized review text without resolving what/whom a phrase refers to. A review that contrasts the current restaurant with others (e.g. "o nhieu quan khac rat kho chiu nhung quan nay thi...") may trigger a rule (e.g. `frustration`) based on a clause that describes a *different* restaurant.
- **Generic statements vs personal intention.** Phrases such as "ung ho" (support/patronize) can appear in a generic recommendation to other customers rather than the reviewer's own repeat-visit intention; both are treated identically by the `advocacy` rule.
- **Negation handling is pattern-based, not syntactic.** Negation guards (`(?<!khong )`, `(?<!chang )`, ...) cover the negation idioms found during corpus analysis (Section 5) but cannot exhaustively cover every possible negation construction in Vietnamese.
- **Residual multi-word negation (documented, not fixed).** The guards added for `waiting_problem`'s 'doi/cho lau' and `frustration`'s 'kho chiu' patterns cover one- and two-word negation prefixes (e.g. 'khong phai doi lau', 'khong he kho chiu'). A small residual of three-or-more-word constructions, e.g. 'khong de khach doi lau' (=staff don't make customers wait long) or 'khong co thai do kho chiu' (=doesn't have an annoying attitude), remain unguarded; extending the lookbehind list to cover every such construction would add unbounded complexity for diminishing returns, so these rare cases (well under 0.5% of reviews) are accepted as label noise rather than chased with ever-more-specific regex.
- **Single-letter 'k' abbreviation of 'khong' not guarded.** Corpus analysis (Section 5) found 1,598 standalone 'k' tokens used as an informal abbreviation of 'khong', in addition to the two-letter 'ko' abbreviation that IS guarded against throughout the rule set (e.g. in `strong_satisfaction`'s 'xuat sac' and 'hai long' patterns, Section 4). Constructions such as 'k xuat sac' (=not excellent) and 'k hai long' (=not satisfied) therefore still trigger `strong_satisfaction` (+0.6) as if positive, in 7 reviews for these two patterns alone. A third '(?<!k )'-style guard alongside every existing '(?<!khong )'/'(?<!ko )' pair across all patterns was judged to add disproportionate regex complexity for a single-character token that is also highly ambiguous in other contexts (e.g. as the currency unit in '30k' VND, or as a typo for 'ok'/'oke'); this is therefore accepted as a documented limitation rather than guarded against.
- These limitations are inherent to a transparent, rule-based, regex-driven approach and are an explicit design trade-off in favour of auditability and reproducibility over recall.

## 10. Output Files

- `overall_satisfaction_rules.json` - the 14 rule definitions with metadata (15,719 bytes)
- `reviews_clean_enhanced.csv` - enhanced dataset, CSV (19,668,962 bytes, 9946 rows x 47 cols)
- `reviews_clean_enhanced.json` - enhanced dataset, JSON (31,217,070 bytes, 9946 records)
- `overall_satisfaction_rule_analysis.md` - this report
