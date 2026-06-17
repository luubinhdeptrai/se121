# CODEBASE_OVERVIEW

## 1. Executive Summary

### Project purpose

This repository implements a multi-modal regression pipeline for restaurant quality assessment from:

- review text
- review images

The core prediction task in the current Python code is multi-target score regression over:

- `food_score`
- `price_score`
- `atmosphere_score`
- `service_score`
- `overall_satisfaction`

The repository also contains a substantial data-engineering pipeline:

1. crawl Foody restaurant, review, and image metadata
2. clean and validate reviews/images
3. generate an explainable `overall_satisfaction` label from rule-based text evidence
4. build image-text training splits
5. train unimodal text, unimodal image, and fused models

### Current development status

Implemented and evidenced in code:

- Foody data crawling notebook with checkpoint/resume support
- cleaning pipeline that produces `restaurants_clean`, `review_images_clean`, `multimodal_reviews`, and `text_only_reviews`
- rule-based label-generation notebook that produces `reviews_clean_enhanced` and `overall_satisfaction_rules.json`
- reusable PyTorch training/evaluation code for text-only, image-only, and fusion models
- executed experiment evidence for:
  - `xlm-roberta-base + ConvNeXt`
  - `microsoft/mdeberta-v3-base + SigLIP ViT`

Partially implemented or inconsistent:

- documentation frequently describes a "joint loss" with alpha weighting, but current training code uses plain vector MSE
- notebook outputs do not always match the current source code at `HEAD`
- some intermediate artifacts referenced by notebooks are not committed
- no committed checkpoints or final `data/text/*.csv` split files are present

Not implemented in runnable project code:

- Grad-CAM
- SHAP
- LIME
- attention visualization
- advanced fusion methods such as cross-attention, FiLM, GMU
- robust losses such as Huber or weighted multi-task losses

### Main research direction

The repository is currently strongest as:

- a baseline multimodal regression system
- a reproducible data-preparation workflow
- a rule-explainable label-engineering workflow for `overall_satisfaction`

It is not yet a complete end-to-end explainable multimodal research platform at model-explanation level.

### Overall architecture summary

Current training architecture in code:

```text
review CSV row
  -> tokenize comment_clean with Hugging Face tokenizer
  -> load image from local MD5(url).jpg cache, or fetch from URL on demand
  -> process image with Hugging Face image processor

Text branch:
  AutoModel(text_model_name)
  -> pooled text feature
  -> FC( hidden -> 256 ) + ReLU + Dropout
  -> Linear(256 -> 5 scores)

Image branch:
  timm.create_model(image_model_name, pretrained=True, num_classes=0)
  -> pooled image feature
  -> FC( hidden -> 256 ) + ReLU + Dropout
  -> Linear(256 -> 5 scores)

Fusion branch:
  freeze text and image branches
  -> extract raw encoder features with no_grad
  -> concatenate(text_feature, image_feature)
  -> MLP( fusion -> 512 -> 256 )
  -> Linear(256 -> 5 scores)
```

Important implementation detail:

- fusion uses raw encoder features returned by the unimodal models, not the 256-d branch projections

## 2. Repository Structure

### Repository tree

```text
SE365/
|-- data_processed/
|   |-- overall_satisfaction_rule_analysis.md
|   |-- overall_satisfaction_rules.json
|   |-- reviews_clean_enhanced.csv
|   `-- reviews_clean_enhanced.json
|-- data_raw/
|   |-- cleaning_report.json
|   |-- multimodal_reviews.csv
|   |-- restaurants_clean.csv
|   |-- restaurants_clean.json
|   |-- review_images_clean.csv
|   |-- review_images_clean.json
|   `-- text_only_reviews.csv
|-- doc/
|   |-- ARCHITECTURE_AND_METRICS.md
|   |-- COLAB_GUIDE.md
|   |-- CREATE_FINAL_DATASET.md
|   |-- DATA_SETUP.md
|   |-- EXPERIMENTAL_PLAN.md
|   |-- Explainable_AI_for_Multimodal_Product_Quality_Assessment.md
|   |-- Multimodal_Learning_Handbook.md
|   |-- Proposal_Multimodel.md
|   `-- XAI_Survival_Guide.md
|-- draft/
|   |-- architechture.png
|   `-- plan.md
|-- Models/
|   |-- FusionModel.py
|   |-- ImageModel.py
|   `-- TextModel.py
|-- notebook/
|   |-- 01_generate_overall_satisfaction.ipynb
|   |-- baseline_mbert_resnet50.ipynb
|   |-- clean_foody_dataset.ipynb
|   |-- colab.ipynb
|   |-- crawl_data_from_foody.ipynb
|   |-- mdeberta-siglip.ipynb
|   |-- ViDeBERTa_SwinB.ipynb
|   `-- xlm-roberta-convnext.ipynb
|-- src/
|   `-- dataset.py
|-- .gitignore
|-- Config.py
|-- download_images.py
|-- LICENSE
|-- main.py
|-- preprocess_data.py
|-- prompt.md
|-- README.md
|-- report.md
|-- requirements.txt
|-- test.py
`-- Trainer.py
```

### Folder-by-folder purpose

| Folder | Purpose | Important files | How it is used |
| --- | --- | --- | --- |
| `data_raw/` | Cleaned-but-pre-enhancement datasets used as upstream inputs | `multimodal_reviews.csv`, `review_images_clean.csv`, `text_only_reviews.csv`, `cleaning_report.json` | Consumed by `preprocess_data.py` and documented by the cleaning notebook |
| `data_processed/` | Label-enhanced dataset and rule artifacts | `reviews_clean_enhanced.csv`, `overall_satisfaction_rules.json`, `overall_satisfaction_rule_analysis.md` | `preprocess_data.py` merges this with `multimodal_reviews.csv` |
| `Models/` | Runtime PyTorch model definitions | `TextModel.py`, `ImageModel.py`, `FusionModel.py` | Imported by `main.py` and `test.py` |
| `src/` | Data loading utilities | `dataset.py` | Defines `MultimodalDataset` for all train/test modes |
| `notebook/` | Data collection, cleaning, labeling, and experiment notebooks | crawling, cleaning, label generation, experiment launch notebooks | Primary source of data-generation logic and historical experiment evidence |
| `doc/` | Human-readable design/reference docs | architecture, setup, proposal, XAI guides | Informational only; not imported by runtime code |
| `draft/` | Early planning artifacts | `plan.md`, `architechture.png` | Planning/reference only |

## 3. Dataset Analysis

### Dataset files

| File | Location | Format | Rows | Notes |
| --- | --- | --- | ---: | --- |
| `restaurants_clean.csv` | `data_raw/` | CSV | 300 | cleaned restaurant metadata |
| `restaurants_clean.json` | `data_raw/` | JSON | 300 records | JSON mirror of restaurant metadata |
| `review_images_clean.csv` | `data_raw/` | CSV | 22,150 | one row per image |
| `review_images_clean.json` | `data_raw/` | JSON | 22,150 records | JSON mirror of image metadata |
| `multimodal_reviews.csv` | `data_raw/` | CSV | 22,150 | one row per `(review, image)` pair |
| `text_only_reviews.csv` | `data_raw/` | CSV | 9,946 | one row per valid review |
| `cleaning_report.json` | `data_raw/` | JSON | n/a | summary statistics from cleaning notebook |
| `reviews_clean_enhanced.csv` | `data_processed/` | CSV | 9,946 | cleaned review dataset plus engineered overall-satisfaction fields |
| `reviews_clean_enhanced.json` | `data_processed/` | JSON | 9,946 records | JSON mirror of enhanced review dataset |
| `overall_satisfaction_rules.json` | `data_processed/` | JSON | 14 rule groups | rule configuration for label generation |
| `overall_satisfaction_rule_analysis.md` | `data_processed/` | Markdown | n/a | generated rule coverage report |

Referenced by notebooks but not present in the committed repository:

- `data_raw/reviews_clean.csv`
- `data_raw/reviews_clean.json`
- generated training splits under `data/text/`
- downloaded image cache under `data/image/`

### Dataset schema

#### `data_raw/restaurants_clean.csv`

| Column | Type | Meaning |
| --- | --- | --- |
| `restaurant_id` | int64 | Foody restaurant identifier |
| `restaurant_name` | object/string | restaurant name |
| `restaurant_url` | object/string | relative Foody path |
| `restaurant_full_url` | object/string | absolute restaurant URL |
| `address` | object/string | restaurant address |
| `avg_rating` | float64 | restaurant-level average rating |
| `total_reviews` | int64 | total reviews reported by Foody |
| `latitude` | float64 | latitude from crawl result |
| `longitude` | float64 | longitude from crawl result |
| `crawl_timestamp` | object/string | crawl timestamp |

#### `data_raw/review_images_clean.csv`

| Column | Type | Meaning |
| --- | --- | --- |
| `image_id` | int64 | Foody image identifier |
| `review_id` | int64 | parent review identifier |
| `restaurant_id` | int64 | parent restaurant identifier |
| `restaurant_name` | object/string | restaurant name |
| `image_url` | object/string | image URL |
| `image_description` | object/string | optional image caption/description |
| `width` | int64 | image width |
| `height` | int64 | image height |
| `bg_color` | object/string | background color metadata from crawl |
| `total_likes` | int64 | likes on image |
| `photo_detail_url` | object/string | image detail URL |
| `crawl_timestamp` | object/string | crawl timestamp |

#### `data_raw/multimodal_reviews.csv`

| Column | Type | Meaning |
| --- | --- | --- |
| `review_id` | int64 | review identifier |
| `restaurant_id` | int64 | restaurant identifier |
| `restaurant_name` | object/string | restaurant name |
| `comment_clean` | object/string | cleaned review text used in multimodal pairing |
| `avg_rating` | float64 | review-level regression target before enhancement |
| `image_id` | int64 | image identifier |
| `image_url` | object/string | image URL |
| `width` | int64 | image width |
| `height` | int64 | image height |
| `created_datetime` | object/string | normalized review creation datetime |

Important behavior:

- this is an image-level dataset, not a review-level dataset
- one review with multiple images appears in multiple rows

#### `data_raw/text_only_reviews.csv`

| Column | Type | Meaning |
| --- | --- | --- |
| `review_id` | int64 | review identifier |
| `restaurant_id` | int64 | restaurant identifier |
| `comment_clean` | object/string | cleaned review text |
| `avg_rating` | float64 | original regression target |
| `created_datetime` | object/string | normalized creation datetime |

#### `data_processed/reviews_clean_enhanced.csv`

This file is the most important analytic artifact. It extends the cleaned review dataset with label-engineering outputs.

| Column group | Columns | Meaning |
| --- | --- | --- |
| Identity | `review_id`, `restaurant_id`, `restaurant_name`, `user_id`, `user_name` | review, restaurant, and user identifiers |
| User/profile | `user_avatar_url` | optional avatar URL |
| Raw text | `title`, `comment` | original title and review text |
| Raw score fields | `avg_rating`, `food_score`, `service_score`, `atmosphere_score`, `position_score`, `price_score` | review scores from Foody; `avg_rating` is later recomputed |
| Review metadata | `review_type`, `review_type_name`, `created_date_raw`, `created_on`, `updated_on`, `device_name`, `device_type`, `total_views`, `total_like`, `total_comment` | source metadata from Foody |
| Image linkage | `has_images`, `image_count`, `review_url`, `source_restaurant_url` | whether the review has images and where it came from |
| Crawl/time metadata | `crawl_timestamp`, `created_datetime`, `updated_datetime`, `created_year`, `created_month` | normalized timestamps and derived temporal fields |
| Clean-text features | `comment_clean`, `emoji_count`, `comment_length`, `word_count` | cleaned text and shallow NLP features |
| Cleaning flags | `is_ad_or_spam`, `is_too_short`, `is_valid_content` | cleaning notebook outputs |
| Rating traceability | `foody_original_avg_rating`, `avg_rating_recomputed` | original 5-aspect mean and boolean showing recomputation from 4 aspects |
| Rule-engine text | `comment_normalized` | normalized text matched by regex rules |
| Rule-engine outputs | `overall_adjustment`, `overall_rules_triggered`, `overall_evidence` | rule scores, triggered categories, and structured evidence |
| Final target | `overall_satisfaction` | `clip(avg_rating + overall_adjustment, 0, 10)` |

Important export detail:

- in CSV, `overall_rules_triggered` and `overall_evidence` are JSON-serialized strings
- in JSON, they remain native arrays/objects

#### `data_processed/overall_satisfaction_rules.json`

| Field | Meaning |
| --- | --- |
| `metadata.version` | rule-set version |
| `metadata.generated_at` | generation timestamp |
| `metadata.scale_min`, `metadata.scale_max` | label range |
| `metadata.total_categories` | total rule groups |
| `metadata.positive_categories`, `metadata.negative_categories` | polarity counts |
| `metadata.language` | language, `vi` |
| `metadata.notes` | explanation of normalization and regex behavior |
| `rules.<rule_name>.score` | additive score contribution |
| `rules.<rule_name>.description` | human explanation |
| `rules.<rule_name>.patterns` | regex patterns for that rule |

### Dataset statistics

Verified from committed artifacts:

| Statistic | Value |
| --- | ---: |
| Restaurants | 300 |
| Valid review-level rows | 9,946 |
| Reviews with at least one image | 6,082 |
| Image-level rows | 22,150 |
| Unique image URLs | 22,150 |
| Unique restaurants in `multimodal_reviews.csv` | 268 |
| Unique restaurants in `reviews_clean_enhanced.csv` | 298 |
| Image coverage over valid reviews | 61.15% |
| Mean original Foody rating | 6.7544 |
| Mean recomputed `avg_rating` | 6.6889 |
| Mean `overall_satisfaction` | 6.7133 |
| Reviews with non-zero `overall_adjustment` | 3,263 |
| Rows clipped to 10 after adjustment | 432 |
| Rows clipped to 0 after adjustment | 26 |
| Reviews with all five aspect scores missing | 3 |

Cleaning-stage statistics from `cleaning_report.json`:

| Stage | Reviews | Images | Notes |
| --- | ---: | ---: | --- |
| Raw crawl | 11,111 | 24,599 | checkpoint crawl outputs before filtering |
| After technical cleaning | 11,111 | 24,599 | no duplicates/empty IDs removed in committed report |
| After content cleaning | 9,946 valid content | n/a | 535 spam, 630 too short, 20 emoji-only flagged |
| After ML filtering | 9,946 | 22,150 | valid reviews and image rows used downstream |

### Train / Validation / Test Split

Implemented split logic in `preprocess_data.py`:

1. merge `data_raw/multimodal_reviews.csv` with `data_processed/reviews_clean_enhanced.csv` on `review_id`
2. drop rows missing `comment_clean`, `image_url`, `overall_satisfaction`, or aspect scores
3. drop duplicate `image_url`
4. keep final columns needed for training
5. randomly sample `n=5500` rows with `random_state=42`
6. split into:
   - train: 4,400
   - validation: 550
   - test: 550

Then `download_images.py` is designed to trim the final usable dataset to exactly:

- train: 4,000
- validation: 500
- test: 500

after verifying which images were actually downloaded.

Observed reproducibility caveats:

- no stratification is used
- the `data/text/*.csv` outputs are not committed
- notebook logs show historical runs with different realized counts such as `3971 / 501 / 528`, so the committed repository does not contain a single canonical final split artifact

## 4. Model Architecture Analysis

### Reusable model classes in the current Python code

| Model | Purpose | Input | Output | Backbone | Feature dimensions | Training objective | File |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `TextModel` | text-only regression | `input_ids`, `attention_mask` | 5 scores + raw text feature | `transformers.AutoModel(model_name)` | dynamic via `encoder.config.hidden_size`; projected to 256 for prediction head | multi-target regression | `Models/TextModel.py` |
| `ImageModel` | image-only regression | `pixel_values` | 5 scores + raw image feature | `timm.create_model(model_name, pretrained=True, num_classes=0)` | dynamic via `encoder.num_features`; projected to 256 for prediction head | multi-target regression | `Models/ImageModel.py` |
| `FusionModel` | image+text regression | text tensors + image tensor | 5 scores | frozen `TextModel` + frozen `ImageModel` + fusion MLP | `text_hidden + image_hidden -> 512 -> 256 -> 5` | multi-target regression | `Models/FusionModel.py` |

### Backbone combinations observed in notebooks

| Notebook | Text backbone | Image backbone | Evidence of execution | Notes |
| --- | --- | --- | --- | --- |
| `xlm-roberta-convnext.ipynb` | `xlm-roberta-base` | `convnext_base_in22k` | Yes | strongest committed experiment evidence |
| `mdeberta-siglip.ipynb` | `microsoft/mdeberta-v3-base` | `vit_base_patch16_siglip_224` | Yes | second committed experiment evidence |
| `baseline_mbert_resnet50.ipynb` | `bert-base-multilingual-cased` | `resnet50` | No output cells | launch template only |
| `ViDeBERTa_SwinB.ipynb` | `FPTAI/vibert-base-cased` | `swin_base_patch4_window7_224` | No output cells | notebook name says ViDeBERTa but command uses ViBERT |
| `colab.ipynb` | default `xlm-roberta-base` | default `convnext_base_in22k` | Yes | 1-epoch smoke run on Colab |

Important inconsistency:

- notebook names and documentation are not always reliable indicators of the actual backbone used

## 5. Image Branch Analysis

### Current image encoder

The runtime image branch is generic and driven by `args.image_model_name`, defaulting to:

- `convnext_base_in22k`

Implementation:

```text
timm.create_model(model_name, pretrained=True, num_classes=0)
-> pooled feature from timm encoder
-> Linear(hidden_size -> 256)
-> ReLU
-> Dropout(0.2)
-> Linear(256 -> 5 scores)
```

### Feature extraction strategy

- `num_classes=0` in timm means the classifier head is removed and the encoder returns features
- `self.encoder(pixel_values)` is used directly
- the code assumes timm returns a pooled feature vector, not a spatial feature map

### Pooling strategy

- pooling is delegated to the timm model implementation
- there is no custom GAP/GMP layer in `ImageModel.py`

### Output dimension

- dynamic via `self.encoder.num_features`
- the prediction head compresses the encoder feature to 256 before regressing 5 scores
- in fusion mode, the code uses the raw encoder feature returned by `ImageModel.forward`, not the 256-d projection

### Important implementation caveat

`main.py` and `test.py` build the image preprocessor with:

```python
AutoImageProcessor.from_pretrained(args.image_model_name)
```

and if that fails they fall back to:

```python
facebook/convnext-base-224-22k
```

Implication:

- ConvNeXt baseline preprocessing is probably fine
- non-Hugging-Face timm backbones such as `resnet50`, `swin_base_patch4_window7_224`, and `vit_base_patch16_siglip_224` likely use the ConvNeXt image processor fallback, which is a preprocessing mismatch risk

## 6. Text Branch Analysis

### Current text encoder

The runtime text branch is generic and driven by `args.text_model_name`, defaulting to:

- `xlm-roberta-base`

Implementation:

```text
AutoModel(model_name)
-> pooled text feature
-> Linear(hidden_size -> 256)
-> ReLU
-> Dropout(0.2)
-> Linear(256 -> 5 scores)
```

### Tokenization

- tokenization uses `transformers.AutoTokenizer.from_pretrained(args.text_model_name)`
- default maximum sequence length is `128`
- padding is `max_length`
- truncation is enabled

### Pooling strategy

The code uses:

1. `outputs.pooler_output` if the model provides it
2. otherwise `outputs.last_hidden_state[:, 0, :]`

For XLM-RoBERTa specifically, this usually means first-token pooling rather than a learned pooler.

### Output dimension

- dynamic via `encoder.config.hidden_size`
- projected to 256 for the branch prediction head
- fusion uses the raw pooled encoder feature, not the 256-d branch projection

## 7. Fusion Layer Analysis

Only one fusion approach is implemented in runnable code.

| Fusion method | Architecture | Inputs | Outputs | File location |
| --- | --- | --- | --- | --- |
| Concatenation + MLP | freeze text/image branches, concatenate raw pooled features, pass through `Linear -> ReLU -> Dropout -> Linear -> ReLU -> Linear` | raw text feature + raw image feature | 5 regression scores | `Models/FusionModel.py` |

Detailed flow:

```text
text_features  = TextModel(...)[1]
image_features = ImageModel(...)[1]
fused = cat(text_features, image_features)
-> Linear(fusion_size, 512)
-> ReLU
-> Dropout(0.2)
-> Linear(512, 256)
-> ReLU
-> Linear(256, 5)
```

Not found in codebase:

- GMU
- FiLM
- cross-attention
- gated fusion
- late averaging ensemble

## 8. Loss Functions

### Implemented losses in current code

| Loss / metric | Formula | Purpose | File location |
| --- | --- | --- | --- |
| `MSELoss` | `mean((y_hat - y)^2)` | training loss and per-target validation/test loss | `Trainer.py`, `test.py` |
| `L1Loss` / MAE | `mean(abs(y_hat - y))` | validation/test reporting only | `Trainer.py`, `test.py` |

### What is actually trained

Current training loss:

```text
loss = MSELoss(pred_factors, true_factors)
```

This is plain vector MSE across all 5 outputs.

### Important documentation mismatch

Several docs describe a joint loss such as:

```text
alpha * MSE_overall + (1 - alpha) * mean(MSE_factor_i)
```

This is **not** what the current training code implements.

Also not found in code:

- Huber loss
- weighted loss
- uncertainty-weighted multi-task loss
- MAE as training objective

## 9. Training Pipeline

### Data loading

- `main.py` builds `MultimodalDataset` for train and validation CSVs
- `DataLoader(..., batch_size=args.batch_size, shuffle=True/False, num_workers=0)`
- dataset returns:
  - tokenized text
  - processed image tensor
  - target vector of 5 scores

### Image loading behavior

`src/dataset.py` uses MD5 of `image_url` as local filename:

```text
image_dir/<md5(image_url)>.jpg
```

Fallback order:

1. local cached JPG
2. download from URL via `requests.get`
3. black `224x224` image if loading fails

### Augmentation

Explicit image augmentation is **not** implemented.

What happens instead:

- preprocessing is delegated entirely to `AutoImageProcessor`
- text preprocessing is tokenization only

### Training loop

Implemented in `Trainer.py`:

1. zero gradients
2. forward pass according to `args.mode`
3. compute vector MSE loss
4. `loss.backward()`
5. gradient clipping with `max_norm=1.0`
6. `optimizer.step()`

### Optimizer

- `torch.optim.AdamW`
- parameters filtered by `requires_grad`
- defaults from `Config.py`:
  - `lr=1e-5`
  - `weight_decay=1e-2`

### Scheduler

- Unknown / Not Found in Codebase

### Early stopping

- Unknown / Not Found in Codebase

### Checkpointing

- best validation loss checkpoint only
- saved to `./checkpoints/best_model_<mode>.pth`

### Missing training robustness features

- no seed-setting for PyTorch, NumPy, or DataLoader workers in the runtime training scripts
- no mixed precision
- no learning-rate scheduler
- no resume-from-checkpoint training
- no experiment logger integration

## 10. Evaluation Pipeline

### Metrics

Current evaluation code computes:

- MSE per target
- RMSE per target
- MAE per target

### Evaluation flow

1. build test dataset from `args.test_path`
2. instantiate model based on `args.mode`
3. load `best_model_<mode>.pth`
4. run inference over test loader
5. aggregate per-target losses and print metrics

### Prediction generation

- predictions are generated batchwise in memory
- no CSV/JSON export of predictions is implemented
- no confusion/error analysis notebook is present beyond printed metrics

### Reporting inconsistency

Current `test.py` expects 5 targets including `overall_satisfaction`, but historical notebook outputs only show 4 visible targets in several runs. Treat archived notebook metrics as historical evidence, not exact reproduction of the current source tree.

## 11. Experiment Inventory

| Experiment ID | Notebook/File | Model configuration | Fusion method | Loss function | Metrics produced | Current status |
| --- | --- | --- | --- | --- | --- | --- |
| `DATA-CRAWL-001` | `notebook/crawl_data_from_foody.ipynb` | Foody API crawl over 300 target restaurants | n/a | n/a | crawl counts and exported datasets | Implemented; notebook defines full pipeline |
| `DATA-CLEAN-001` | `notebook/clean_foody_dataset.ipynb` | technical cleaning + content filtering + ML filtering | n/a | n/a | cleaning report, rating distribution, image coverage | Implemented; committed outputs exist in `data_raw/` |
| `LABEL-OVERALL-001` | `notebook/01_generate_overall_satisfaction.ipynb` | 14-rule regex engine over normalized `comment_clean` | n/a | additive rule scores over recomputed `avg_rating` | coverage, adjustment stats, clipping stats, correlations | Implemented; committed outputs exist in `data_processed/` |
| `EXP-XLM-CONVNEXT-001` | `notebook/xlm-roberta-convnext.ipynb` | `xlm-roberta-base` + `convnext_base_in22k`; text 5 ep, image 10 ep, fusion 15 ep | concat + MLP | MSE | Final test: Food MAE 1.0098, Price MAE 1.0483, Atmos MAE 1.0135, Service MAE 1.0330; corresponding MSE/RMSE also logged | Completed historical run |
| `EXP-MDEBERTA-SIGLIP-001` | `notebook/mdeberta-siglip.ipynb` | `microsoft/mdeberta-v3-base` + `vit_base_patch16_siglip_224`; text 5 ep, image 10 ep, fusion 15 ep | concat + MLP | MSE | Final test: Food MAE 1.1016, Price MAE 1.0694, Atmos MAE 1.0547, Service MAE 1.1185; corresponding MSE/RMSE also logged | Completed historical run |
| `EXP-MBERT-RESNET50-001` | `notebook/baseline_mbert_resnet50.ipynb` | `bert-base-multilingual-cased` + `resnet50` | concat + MLP | MSE | None committed | Launch template only |
| `EXP-VIBERT-SWIN-001` | `notebook/ViDeBERTa_SwinB.ipynb` | actually `FPTAI/vibert-base-cased` + `swin_base_patch4_window7_224` | concat + MLP | MSE | None committed | Launch template only; notebook title is misleading |
| `EXP-COLAB-SMOKE-001` | `notebook/colab.ipynb` | default baseline, 1 epoch each stage | concat + MLP | MSE | 1-epoch val/test metrics printed | Smoke test / demo run |

Notes:

- `README.md` and `report.md` summarize only the two completed backbone experiments
- no committed evidence exists for robust losses, advanced fusion, or XAI experiments

## 12. XAI Analysis

### What is actually implemented

Implemented:

- rule-based explainability for `overall_satisfaction`
- per-review evidence via `overall_evidence`

This is label-side explainability, not model-side explainability.

### What exists only as documentation

| Technique | File location | Current status | How it works in repo context |
| --- | --- | --- | --- |
| Grad-CAM | `doc/XAI_Survival_Guide.md`, `doc/Explainable_AI_for_Multimodal_Product_Quality_Assessment.md` | Documentation only | described conceptually for image localization |
| SHAP | same docs | Documentation only | described for fusion-level contribution analysis |
| LIME | same docs | Documentation only | described for local perturbation analysis |
| Attention visualization | same docs | Documentation only | described for text-token inspection |

Not found in runnable codebase:

- no Grad-CAM implementation
- no SHAP wrapper
- no LIME wrapper
- no attention extraction/visualization pipeline

## 13. Configuration Analysis

### Runtime CLI configuration from `Config.py`

| Parameter | Default | Meaning |
| --- | --- | --- |
| `--mode` | `train_text` | one of `train_text`, `train_image`, `train_fusion` |
| `--train_path` | `./data/text/train.csv` | train CSV |
| `--val_path` | `./data/text/val.csv` | validation CSV |
| `--test_path` | `./data/text/test.csv` | test CSV |
| `--image_dir` | `./data/image` | local image cache directory |
| `--save_path` | `./checkpoints` | checkpoint directory |
| `--text_model_name` | `xlm-roberta-base` | Hugging Face text model id |
| `--image_model_name` | `convnext_base_in22k` | timm image model name |
| `--max_length` | `128` | max token length |
| `--batch_size` | `16` | batch size |
| `--epochs` | `5` | epochs |
| `--lr` | `1e-5` | learning rate |
| `--weight_decay` | `1e-2` | AdamW weight decay |

### Notebook configuration highlights

| Source | Key settings |
| --- | --- |
| `crawl_data_from_foody.ipynb` | `TARGET_RESTAURANTS=300`, `TEST_MODE=False`, `CHECKPOINT_EVERY_RESTAURANTS=10`, `sleep_between_requests=0.8`, `sleep_between_restaurants=2.0` |
| `clean_foody_dataset.ipynb` | `MIN_COMMENT_LENGTH=15`, `MIN_WORD_COUNT=3`, valid rating range `[0,10]` |
| `01_generate_overall_satisfaction.ipynb` | `RANDOM_SEED=42`, 14 rule categories, 4-aspect recomputation excluding `position_score` |
| `preprocess_data.py` | `sample(n=5500, random_state=42)` |
| `download_images.py` | target sizes `(4000, 500, 500)` and `ThreadPoolExecutor(max_workers=20)` |

### YAML / TOML / environment files

- YAML configs: Unknown / Not Found in Codebase
- `.yml` / `.yaml` experiment configs: Unknown / Not Found in Codebase
- pinned dependency lockfile: Unknown / Not Found in Codebase

## 14. File-by-File Summary

| File | Purpose | Notes |
| --- | --- | --- |
| `Config.py` | CLI argument definitions | runtime defaults for paths, backbones, and optimization |
| `main.py` | training entrypoint | selects mode, builds datasets/processors/models, launches `Trainer` |
| `Trainer.py` | training and validation loop | plain MSE training, AdamW, gradient clipping, best-checkpoint save |
| `test.py` | evaluation entrypoint | loads saved checkpoint and prints MSE/RMSE/MAE |
| `src/dataset.py` | `MultimodalDataset` | loads CSV rows, tokenizes text, loads cached or remote images |
| `Models/TextModel.py` | text branch wrapper | Hugging Face encoder plus shallow regression head |
| `Models/ImageModel.py` | image branch wrapper | timm encoder plus shallow regression head |
| `Models/FusionModel.py` | fusion branch wrapper | frozen unimodal branches plus concatenation MLP |
| `preprocess_data.py` | build train/val/test CSVs | merges enhanced review labels with multimodal image rows |
| `download_images.py` | download and trim image cache | hashes URLs to JPG names and overwrites split CSVs after validation |
| `requirements.txt` | Python dependencies | unpinned package list only |
| `README.md` | short project readme | contains setup plus historical benchmark tables |
| `report.md` | project progress report | useful state summary, but not executable |
| `prompt.md` | task prompt artifact | not part of project runtime |
| `data_raw/restaurants_clean.csv` | cleaned restaurant metadata | crawl output |
| `data_raw/restaurants_clean.json` | JSON mirror of restaurant metadata | data mirror only |
| `data_raw/review_images_clean.csv` | cleaned image metadata | one row per image |
| `data_raw/review_images_clean.json` | JSON mirror of image metadata | data mirror only |
| `data_raw/multimodal_reviews.csv` | multimodal training precursor | one row per review-image pair |
| `data_raw/text_only_reviews.csv` | text-only training precursor | one row per review |
| `data_raw/cleaning_report.json` | cleaning statistics report | produced by cleaning notebook |
| `data_processed/reviews_clean_enhanced.csv` | enhanced review dataset | adds rule-based overall satisfaction features |
| `data_processed/reviews_clean_enhanced.json` | JSON mirror of enhanced dataset | keeps nested evidence fields as arrays/objects |
| `data_processed/overall_satisfaction_rules.json` | rule configuration | 14 explainable regex-based rule categories |
| `data_processed/overall_satisfaction_rule_analysis.md` | generated rule analysis report | summarizes rule coverage and target properties |
| `notebook/crawl_data_from_foody.ipynb` | Foody crawler notebook | includes HTTP session logic, discovery, pagination, detail enrichment, checkpoint/resume |
| `notebook/clean_foody_dataset.ipynb` | data cleaning notebook | technical cleaning, content filtering, ML dataset construction |
| `notebook/01_generate_overall_satisfaction.ipynb` | label-engineering notebook | creates explainable `overall_satisfaction` |
| `notebook/xlm-roberta-convnext.ipynb` | executed baseline experiment | strongest historical experiment evidence |
| `notebook/mdeberta-siglip.ipynb` | executed alternative experiment | second historical experiment evidence |
| `notebook/baseline_mbert_resnet50.ipynb` | experiment launcher | command-only template |
| `notebook/ViDeBERTa_SwinB.ipynb` | experiment launcher | command-only template; title/model mismatch |
| `notebook/colab.ipynb` | Colab quickstart | end-to-end 1-click smoke run |
| `doc/ARCHITECTURE_AND_METRICS.md` | design rationale doc | partially stale versus current loss implementation |
| `doc/DATA_SETUP.md` | data layout guide | documents expected `data/` folder not committed here |
| `doc/CREATE_FINAL_DATASET.md` | dataset-generation guide | high-level lifecycle across crawl, clean, label, split, image download |
| `doc/COLAB_GUIDE.md` | Colab usage guide | points users to `notebook/colab.ipynb` |
| `doc/EXPERIMENTAL_PLAN.md` | proposed future experiments | proposal only, not implemented |
| `doc/Proposal_Multimodel.md` | research proposal | conceptual architecture document |
| `doc/Multimodal_Learning_Handbook.md` | educational handbook | background/reference only |
| `doc/Explainable_AI_for_Multimodal_Product_Quality_Assessment.md` | XAI study guide | reference only |
| `doc/XAI_Survival_Guide.md` | XAI defense/reference guide | reference only |
| `draft/plan.md` | early project plan | planning artifact only |
| `draft/architechture.png` | draft diagram | not used by code |

## 15. Current Progress Assessment

### Already Completed

- data crawl pipeline with checkpoint/resume design
- dataset cleaning and ML filtering
- multimodal pair construction
- explainable `overall_satisfaction` target generation
- baseline text/image/fusion training code
- baseline evaluation code
- at least two historical backbone experiment runs with reported metrics

### Partially Completed

- reproducible experiment packaging
- documentation consistency across notebooks, README, and runtime code
- use of `overall_satisfaction` across all historical experiments
- experiment inventory for alternative backbones beyond the two completed runs

### Missing Components

- committed final `data/text/` split files
- committed image cache under `data/image/`
- committed checkpoints
- robust-loss experiments
- advanced fusion experiments
- model-side XAI implementations
- version-pinned environment specification
- automated tests

### Technical Debt

- docs claim joint loss, code uses plain vector MSE
- notebook outputs appear to come from older code revisions
- image preprocessing fallback may mismatch non-ConvNeXt backbones
- training/evaluation average metrics per batch rather than exact samplewise aggregation
- no deterministic training seed management
- no scheduler or early stopping

### Risks

- exact historical metrics may be hard to reproduce from current `HEAD`
- image download failures can silently change final split sizes
- fusion performance depends on frozen encoder features and inherited unimodal checkpoints
- external Foody endpoints may drift or block crawling

## 16. Future Experiment Opportunities

### Image Branch

| Experiment | Why it is valuable |
| --- | --- |
| Fix backbone-specific preprocessing for ResNet, Swin, and SigLIP | current fallback to ConvNeXt preprocessing can distort comparisons |
| Compare ConvNeXt vs Swin vs SigLIP under identical split and seed control | current evidence mixes backbone changes with historical pipeline drift |
| Add explicit image augmentation | current code relies only on model processor normalization/resizing |

### Text Branch

| Experiment | Why it is valuable |
| --- | --- |
| Re-run XLM-R, mDeBERTa, mBERT, and ViBERT on the same exact committed split | isolates backbone effect from split drift |
| Test longer context windows than `max_length=128` | restaurant reviews can be long and may lose sentiment evidence under truncation |
| Inspect first-token pooling vs mean pooling | current code uses first token when no pooler exists; this may underuse sequence information |

### Fusion Layer

| Experiment | Why it is valuable |
| --- | --- |
| GMU or gated fusion | helps learn modality weighting rather than fixed concatenation |
| FiLM-style conditioning | lets text guide image feature modulation |
| Cross-attention over token and patch features | better models interaction than pooled-vector concatenation |
| Unfreeze selected encoder layers during fusion | current fusion head may underfit cross-modal relationships because encoders are frozen and wrapped in `no_grad` |

### Loss Function

| Experiment | Why it is valuable |
| --- | --- |
| Implement true joint loss with separate overall and aspect terms | aligns code with documented research intent |
| Huber loss | more robust to noisy ratings and review outliers |
| Uncertainty-weighted multi-task loss | lets the model learn output-specific weighting instead of equal MSE over all targets |
| Weighted loss emphasizing `overall_satisfaction` | useful if the final thesis target prioritizes holistic judgment |

### XAI

| Experiment | Why it is valuable |
| --- | --- |
| Grad-CAM on image branch | shows whether the model attends to food regions or background artifacts |
| Attention or saliency analysis for text branch | identifies influential phrases for individual predictions |
| SHAP on fusion embeddings | quantifies modality contribution and branch dominance |
| LIME on image and text inputs | adds local perturbation-based explanation and sanity checks |

## 17. Reproducibility Checklist

### Required data artifacts

- `data_raw/multimodal_reviews.csv`
- `data_processed/reviews_clean_enhanced.csv`
- generated split files:
  - `data/text/train.csv`
  - `data/text/val.csv`
  - `data/text/test.csv`
- downloaded image cache:
  - `data/image/<md5(image_url)>.jpg`

Current repository state:

- split CSVs: Not committed
- image cache: Not committed
- checkpoints: Not committed

### Seeds and deterministic controls

Found:

- `RANDOM_SEED=42` in `01_generate_overall_satisfaction.ipynb`
- `random_state=42` in `preprocess_data.py`

Missing from runtime training code:

- PyTorch seed
- CUDA determinism flags
- DataLoader worker seed handling

### Config files and constants

- runtime CLI defaults: `Config.py`
- cleaning thresholds: `notebook/clean_foody_dataset.ipynb`
- crawl behavior: `notebook/crawl_data_from_foody.ipynb`
- label rules: `data_processed/overall_satisfaction_rules.json`

### Checkpoints

Expected paths:

- `./checkpoints/best_model_train_text.pth`
- `./checkpoints/best_model_train_image.pth`
- `./checkpoints/best_model_train_fusion.pth`

Current repository state:

- Unknown / Not Found in Codebase

### Dependencies

Committed dependency list:

- `numpy`
- `pandas`
- `Pillow`
- `scikit-learn`
- `torch`
- `torchvision`
- `tqdm`
- `transformers`
- `timm`
- `requests`

Reproducibility gap:

- versions are not pinned

### Minimal end-to-end reproduction path

1. run `notebook/crawl_data_from_foody.ipynb` or provide equivalent raw data
2. run `notebook/clean_foody_dataset.ipynb`
3. run `notebook/01_generate_overall_satisfaction.ipynb`
4. run `python preprocess_data.py`
5. run `python download_images.py`
6. train:
   - `python main.py --mode train_text`
   - `python main.py --mode train_image`
   - `python main.py --mode train_fusion`
7. evaluate:
   - `python test.py --mode train_fusion`

### Final assessment

The repository is reproducible at workflow level, but not yet fully reproducible at artifact level. The missing committed split files, missing checkpoints, unpinned dependencies, and historical notebook/code drift are the main blockers to exact experiment replay.
