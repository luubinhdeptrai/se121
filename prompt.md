# ROLE

You are a Senior AI Research Scientist, Multimodal Deep Learning Engineer, and Research Codebase Architect.

You specialize in:

- Computer Vision
- Natural Language Processing
- Multimodal Deep Learning
- Regression Modeling
- Explainable AI
- PyTorch
- Hugging Face Transformers
- timm
- Research Experiment Design
- Reproducibility Engineering
- Technical Proposal Writing
- Research Codebase Refactoring

Your task is to read the entire existing codebase and create a professional experiment proposal file.

---

# GOAL

Read and analyze the entire codebase first.

Then create a Markdown file named:

```text
proposal.md
```

The goal of `proposal.md` is to propose a complete experiment roadmap for this multimodal deep learning project.

The proposal must be:

- professional
- logically organized
- scientifically defensible
- easy to understand
- implementation-ready
- suitable for explaining to a university lecturer

The proposal should not only list experiments. It must explain the **methodology**, **research philosophy**, and **reasoning behind each design choice**.

A future AI coding agent should be able to read `proposal.md` and implement the experiments with minimal follow-up questions.

---

# IMPORTANT REFERENCE FILE

Before writing `proposal.md`, read the existing file:

```text
@EXPERIMENTAL_PLAN.md
```

or any equivalent @EXPERIMENTAL_PLAN.md file in the codebase.

This file was proposed by my teammate and may contain useful ideas about:

- backbone experiments
- text-image model combinations
- loss function experiments
- fusion experiments
- benchmark expectations
- references and rationale

You must analyze this file carefully.

If any ideas in `EXPERIMENTAL_PLAN.md` are useful, scientifically valid, and feasible, incorporate them into `proposal.md`.

However, do not copy it blindly.

You must critically evaluate it and improve it where necessary.

For example:

- If a proposed model pair is suitable, include it.
- If a proposed claim is too strong, rewrite it more carefully.
- If the plan mixes classification results with my regression task, adapt the rationale properly.
- If the plan proposes useful fusion/loss ideas, integrate them into the new phase-based roadmap.
- If any cited pair is not directly suitable for Vietnamese regression, explain it as a candidate rather than a guaranteed best choice.

The final proposal should be stronger, more systematic, and more reproducible than the original experimental plan.

---

# PROJECT CONTEXT

This project is an Explainable Multimodal Deep Learning System for restaurant/product quality assessment using:

- review text
- review images

The current dataset consists of train/validation/test CSV files with columns similar to:

```text
review_id
comment_clean
image_url
overall_satisfaction
food_score
service_score
atmosphere_score
price_score
```

Important dataset characteristics:

- The dataset is based on Foody-style Vietnamese restaurant reviews.
- Text is mainly Vietnamese and may contain informal language, slang, noisy encoding, and user-generated content.
- `image_url` may contain multiple images for one review.
- One review can have several images.
- Images are highly heterogeneous and noisy:
  - food photos
  - drink photos
  - menu screenshots
  - receipts/order screenshots
  - takeaway packaging
  - restaurant environment photos
  - staff/person photos
  - blurry or low-quality images

- Therefore, image features may be useful but also noisy.
- Text is expected to be a strong signal.
- Fusion should ideally learn when to trust text more and when to trust image more.

The current codebase already includes a baseline pipeline with:

- dataset loading
- text-only model
- image-only model
- fusion model
- training loop
- evaluation loop
- at least some historical experiments

---

# CURRENT CODEBASE ISSUES TO ADDRESS

When writing the proposal, explicitly consider and address these technical issues:

```text
image preprocessing fallback may mismatch non-ConvNeXt backbones
training/evaluation average metrics per batch rather than exact samplewise aggregation
no deterministic training seed management
no scheduler or early stopping
no seed-setting for PyTorch, NumPy, or DataLoader workers in runtime training scripts
no mixed precision
no learning-rate scheduler
no resume-from-checkpoint training
no experiment logger integration
docs may claim joint loss but code may use plain vector MSE
notebook outputs may come from older code revisions
final split files may not be committed
image cache may not be committed
checkpoints may not be committed
historical metrics may be hard to reproduce
image download failures may silently change final split sizes
```

The proposal must include a dedicated infrastructure phase to fix these issues before running serious experiments.

---

# REQUIRED RESEARCH PHILOSOPHY

Use this methodology:

```text
Controlled Sequential Ablation + Promising Combination Validation
```

The flow is:

```text
1. Choose reasonable fixed components first
2. Ablate one component
3. Pick the best variant
4. Replace the old component with the best one
5. Move to the next component
6. Finally compare with several promising full combinations
```

Explain this clearly in `proposal.md`.

The proposal must explain why this approach is better than randomly trying many model pairs.

Also explain why full factorial search is expensive and not necessary for this project.

The proposal should clearly distinguish:

```text
Engineering hygiene:
seed control, scheduler, early stopping, sample-wise metrics, logging, preprocessing correctness

Research contribution:
image backbone, text backbone, fusion method, loss function, XAI analysis
```

---

# REQUIRED PHASE STRUCTURE

Organize `proposal.md` into phases.

Recommended structure:

```text
Phase 0: Research Infrastructure and Reproducibility Setup
Phase 1: Baseline Establishment
Phase 2: Image Branch Ablation
Phase 3: Text Branch Ablation
Phase 4: Fusion Layer Ablation
Phase 5: Loss Function Ablation
Phase 6: Promising Full-Combination Validation
Phase 7: Final Model Selection
Phase 8: Explainable AI Analysis
Phase 9: Final Experiment Packaging and Thesis-Ready Reporting
```

For each phase, include:

- goal
- motivation
- fixed components
- variable component
- experiment list
- implementation notes
- expected output files
- decision rule
- expected claim/conclusion

---

# MULTIMODAL ARCHITECTURE CONTEXT

Assume the general architecture is:

```text
Review Text
   ↓
Text Encoder
   ↓
Text Feature Extraction / Pooling
   ↓
Text Embedding
                    ↘
                     Fusion Layer
                    ↗
Image(s)
   ↓
Image Encoder
   ↓
Image Feature Extraction / Pooling
   ↓
Image Embedding
                    ↓
              Regression Heads
                    ↓
overall_satisfaction
food_score
service_score
atmosphere_score
price_score
```

The proposal must identify ablation opportunities in:

---

## A. Image Branch

Model-level candidates:

- ConvNeXt
- Swin-B
- EfficientNet-B3
- CLIP visual encoder
- SigLIP / SigLIP2
- EVA-CLIP if feasible
- ViT-L
- MobileViT if resource-constrained

Component-level variants may include:

- Global Average Pooling
- Spatial Attention
- Attention Pooling
- multi-image mean pooling across review images
- multi-image attention pooling across review images
- image quality filtering
- backbone-specific preprocessing

Important note:

Do not incorrectly call CLS pooling a ConvNeXt feature strategy unless the model actually has a CLS token. CLS-style extraction is appropriate for ViT-like models, not CNN-like ConvNeXt.

---

## B. Text Branch

Model-level candidates:

- XLM-RoBERTa
- PhoBERT
- ViDeBERTa / ViBERT if available
- ViSoBERT
- mDeBERTa-v3
- RoBERTa as a reference baseline if needed

Component-level variants may include:

- first-token / CLS pooling
- mean pooling
- attention pooling
- layer aggregation
- longer max length
- text normalization strategy

Important note:

Because the dataset is mainly Vietnamese and user-generated, Vietnamese-specific or social-media-oriented models such as PhoBERT, ViDeBERTa/ViBERT, and ViSoBERT should be strongly considered.

---

## C. Fusion Layer

Fusion candidates:

- Concatenation + MLP
- GMU
- Gated Cross-Modal Fusion
- FiLM
- Cross-Attention
- late weighted averaging as a simple comparison if appropriate

The proposal should explain:

- Concatenation is the baseline.
- GMU is useful when modality reliability differs across samples.
- Gated Cross-Modal Fusion is useful because this dataset has noisy images.
- FiLM allows one modality to modulate another.
- Cross-Attention provides fine-grained token-patch interaction but is more complex and expensive.

---

## D. Loss Function

Loss candidates:

- MSE
- MAE
- Huber
- SmoothL1
- Log-Cosh if feasible
- Weighted Huber
- true multi-task weighted loss
- homoscedastic uncertainty-weighted multi-task loss if feasible

The proposal must clarify:

- MSE is the baseline.
- Huber/SmoothL1/Log-Cosh are useful for noisy labels and outliers.
- Weighted multi-task loss addresses task balancing.
- Huber itself does not solve multi-task balancing; task weights solve balancing.
- Focal loss and weighted cross-entropy are not suitable for the main regression task unless an auxiliary classification head is added.

---

# REQUIRED BASELINE DESIGN

The proposal must clearly define baseline experiments.

At minimum include:

```text
Baseline 0.1: Text-only baseline
Baseline 0.2: Image-only baseline
Baseline 0.3: Multimodal baseline
```

Example:

```text
Text-only:
XLM-R / PhoBERT / current text baseline
Loss: MSE

Image-only:
ConvNeXt / current image baseline
Loss: MSE

Multimodal baseline:
ConvNeXt + XLM-R
Fusion: Concatenation + MLP
Loss: MSE
```

Explain that every later experiment should be compared against these baselines.

---

# REQUIRED IMAGE-TEXT PAIR RECOMMENDATIONS

The proposal must recommend several promising image-text pairs.

Use the current codebase, dataset characteristics, and `EXPERIMENTAL_PLAN.md` as references.

Possible pairs include:

```text
ConvNeXt + XLM-R
ConvNeXt + PhoBERT
Swin-B + PhoBERT
ViSoBERT + EfficientNet-B3
CLIP visual encoder + RoBERTa / Vietnamese text encoder
SigLIP / SigLIP2 + mDeBERTa / DeBERTa-v3
SigLIP / EVA-CLIP + ViDeBERTa
MobileViT + PhoBERT
```

The proposal should explain:

- XLM-R is a multilingual baseline.
- PhoBERT / ViDeBERTa / ViSoBERT are suitable for Vietnamese text.
- ConvNeXt is a strong CNN-style visual baseline.
- EfficientNet-B3 is a strong and efficient CNN alternative.
- Swin-B provides hierarchical transformer-based multi-scale visual features.
- CLIP / SigLIP / EVA-CLIP may be strong because their visual features are pretrained with image-text alignment.
- MobileViT is useful if lightweight training or deployment matters.

Do not claim that a pair is guaranteed to be best. Present it as a research candidate with a clear hypothesis.

---

# REQUIRED EXPERIMENT DETAILS

For each experiment, include full details:

- experiment ID
- experiment name
- research question
- image branch
- image internal variant
- text branch
- text internal variant
- fusion method
- loss function
- fixed components
- variable component
- training settings
- expected artifacts
- evaluation metrics
- selection criterion
- expected conclusion

Example format:

```markdown
## EXP_003_image_swinb_vs_convnext

### Research Question

Does Swin-B improve image representation compared to ConvNeXt under the same text, fusion, and loss settings?

### Fixed Components

- Text branch: PhoBERT with mean pooling
- Fusion: Concatenation + MLP
- Loss: Huber
- Dataset split: frozen v1
- Seed: 42

### Variable Component

- Image backbone:
  - ConvNeXt
  - Swin-B
  - SigLIP / EVA-CLIP

### Why This Experiment Matters

...

### Expected Outputs

- config.yaml
- metrics.json
- predictions.csv
- train.log
- loss_curve.png
- README.md
```

---

# REQUIRED REPRODUCIBILITY REQUIREMENTS

The proposal must include a reproducibility checklist.

Include:

```text
fixed dataset version
committed train/val/test split files
fixed random seed
Python random seed
NumPy seed
PyTorch seed
CUDA deterministic settings when feasible
DataLoader worker seed
saved config.yaml for each experiment
saved metrics.json
saved predictions.csv
saved checkpoint or checkpoint link
reported hyperparameters
reported package versions
backbone-specific preprocessing
sample-wise metric aggregation
same evaluation script for all experiments
similar metrics when re-run
```

Also include:

- dependency pinning
- environment export
- clear data versioning
- no silent re-splitting
- no silent image re-downloading that changes sample counts

---

# REQUIRED METRICS

The proposal must require sample-wise metrics, not batch-average metrics.

Use:

```text
MAE
RMSE
R²
per-target MAE
per-target RMSE
overall_satisfaction MAE
average aspect MAE
```

Targets:

```text
overall_satisfaction
food_score
service_score
atmosphere_score
price_score
```

Metrics must be computed from all predictions and ground-truth labels after concatenating the full validation/test predictions.

Do not average metric values across batches unless weighted by batch size.

The proposal should require exporting:

```text
predictions.csv
```

with columns such as:

```text
review_id
split
y_true_overall
y_pred_overall
y_true_food
y_pred_food
y_true_service
y_pred_service
y_true_atmosphere
y_pred_atmosphere
y_true_price
y_pred_price
absolute_error_overall
absolute_error_food
absolute_error_service
absolute_error_atmosphere
absolute_error_price
```

---

# REQUIRED TRAINING INFRASTRUCTURE

The proposal must require improving training infrastructure before serious experiments.

Include:

- deterministic seed utility
- sample-wise metric computation
- early stopping
- learning-rate scheduler
- mixed precision training if GPU supports it
- checkpoint saving
- resume-from-checkpoint
- experiment logging
- metrics export
- predictions export
- training curves
- error analysis output

Recommended defaults:

```text
optimizer: AdamW
scheduler: ReduceLROnPlateau or linear warmup + cosine decay
early stopping: patience 3-5
gradient clipping: max_norm=1.0
seed: 42
mixed precision: enabled when using CUDA
```

Clarify that scheduler, early stopping, seed control, logging, and exact metric aggregation are infrastructure requirements, not primary research contributions.

---

# REQUIRED HYBRID .PY + NOTEBOOK WORKFLOW

The proposal must recommend a hybrid implementation style:

```text
.py files contain reusable core logic
.ipynb notebooks only call scripts, inspect outputs, visualize results, and perform XAI/error analysis
```

## Shared `.py` code should contain:

```text
dataset loading
image preprocessing
text tokenization
model definitions
fusion modules
loss functions
training loop
evaluation loop
metrics computation
seed utilities
checkpoint utilities
experiment logger
XAI utilities
```

## Notebooks should contain:

```text
Google Drive mount
environment setup
copy dataset/artifacts
run training commands
display metrics
draw charts
inspect predictions
visualize XAI results
```

The notebook must not contain the only copy of core training logic.

---

# REQUIRED PROJECT STRUCTURE

The proposal must propose a professional repository structure:

```text
project/
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── splits/
│   │   ├── train.csv
│   │   ├── val.csv
│   │   └── test.csv
│   └── README.md
│
├── src/
│   ├── datasets/
│   ├── models/
│   ├── fusion/
│   ├── losses/
│   ├── training/
│   ├── evaluation/
│   ├── metrics/
│   ├── xai/
│   └── utils/
│
├── configs/
├── notebooks/
├── experiments/
├── reports/
├── scripts/
├── requirements.txt
├── environment.yml
└── README.md
```

The proposal should explain which files are shared and which files belong to individual experiments.

---

# REQUIRED GOOGLE COLAB + DRIVE + GITHUB ARTIFACT STRATEGY

Training will be done on Google Colab.

Use this principle:

```text
Google Drive = heavy artifact storage
GitHub repo = code + configs + lightweight results + documentation
```

## Google Drive should store:

```text
large checkpoints
full image cache
large XAI arrays
large intermediate tensors
large logs if needed
full artifact backup
```

## GitHub should store:

```text
source code
configs
proposal.md
README files
metrics.json
predictions.csv if not too large
prediction samples if full file is too large
training logs if reasonable
figures
leaderboard
experiment summaries
checkpoint links
environment files
```

## Each Colab experiment should save to Drive first:

```text
/content/drive/MyDrive/<project_name>/experiments/EXP_XXX/
```

Then copy lightweight artifacts back into the repository:

```text
repo/experiments/EXP_XXX/
```

Files to copy from Drive to repo after each experiment:

```text
config.yaml
metrics.json
predictions.csv or predictions_sample.csv
train.log
README.md
loss_curve.png
prediction_vs_groundtruth.png
error_analysis.csv
xai figures if available
checkpoint_link.md
```

Large files should remain on Drive:

```text
best_model.pt
last_model.pt
optimizer.pt
scheduler.pt
large .npy SHAP arrays
large image cache
```

The repository should contain links to heavy artifacts.

---

# REQUIRED EXPERIMENT FOLDER TEMPLATE

Every experiment folder must follow this template:

```text
experiments/
└── EXP_XXX_short_name/
    ├── README.md
    ├── config.yaml
    ├── metrics.json
    ├── predictions.csv
    ├── error_analysis.csv
    ├── train.log
    ├── figures/
    │   ├── loss_curve.png
    │   ├── mae_by_target.png
    │   └── prediction_vs_groundtruth.png
    ├── xai/
    │   ├── gradcam/
    │   ├── attention/
    │   ├── shap/
    │   └── lime/
    └── checkpoint_link.md
```

The proposal must define what each file is used for.

---

# REQUIRED CONFIG TEMPLATE

Include a reusable `config.yaml` template in `proposal.md`.

It should contain at least:

```yaml
experiment:
  id: EXP_XXX
  name: short_name
  description: ""
  seed: 42
  dataset_version: frozen_split_v1

data:
  train_path: data/splits/train.csv
  val_path: data/splits/val.csv
  test_path: data/splits/test.csv
  image_dir: data/images
  max_text_length: 128
  multi_image_strategy: mean_pooling

image_branch:
  backbone: convnext_base
  processor: official_backbone_processor
  feature_strategy: gap
  trainable: true

text_branch:
  backbone: xlm-roberta-base
  tokenizer: official_tokenizer
  feature_strategy: cls_pooling
  trainable: true

fusion:
  method: concat_mlp
  hidden_dims: [512, 256]
  dropout: 0.2

loss:
  name: mse
  multitask_weights:
    overall_satisfaction: 1.0
    food_score: 1.0
    service_score: 1.0
    atmosphere_score: 1.0
    price_score: 1.0

training:
  epochs: 20
  batch_size: 16
  optimizer: adamw
  learning_rate: 1e-5
  weight_decay: 1e-2
  scheduler: reduce_on_plateau
  early_stopping_patience: 5
  gradient_clip_norm: 1.0
  mixed_precision: true
  resume_from_checkpoint: false

evaluation:
  metrics:
    - mae
    - rmse
    - r2
  aggregation: sample_wise

artifacts:
  output_dir: experiments/EXP_XXX_short_name
  save_checkpoint: true
  save_predictions: true
  save_metrics: true
  save_figures: true
```

---

# REQUIRED README TEMPLATE FOR EACH EXPERIMENT

Include this template:

```markdown
# EXP_XXX - Experiment Name

## Goal

## Research Question

## Configuration Summary

## Fixed Components

## Variable Component

## Dataset Version

## Training Settings

## Results

## Comparison Against Baseline

## Interpretation

## Limitations

## Reproduction Command

## Artifact Links
```

---

# REQUIRED FINAL TABLES IN PROPOSAL

The proposal must include these tables:

## Experiment Roadmap Table

Columns:

```text
Phase
Experiment ID
Experiment Name
Variable Component
Fixed Components
Purpose
Expected Claim
Priority
```

## Image Branch Candidate Table

Columns:

```text
Image Backbone
Family
Why Try It
Expected Strength
Expected Risk
```

## Text Branch Candidate Table

Columns:

```text
Text Backbone
Family
Why Try It
Expected Strength
Expected Risk
```

## Fusion Candidate Table

Columns:

```text
Fusion Method
Why Try It
Complexity
Expected Benefit
Risk
```

## Loss Function Candidate Table

Columns:

```text
Loss
Why Try It
Suitable For
Risk
```

## Artifact Management Table

Columns:

```text
Artifact
Generated By
Stored in Drive?
Committed to GitHub?
Reason
```

## Reproducibility Checklist Table

Columns:

```text
Requirement
Status in current codebase
Required improvement
Priority
```

---

# REQUIRED XAI STRATEGY

The proposal must not require full XAI for every experiment.

Instead:

```text
Run lightweight sanity-check XAI early on 1-2 baseline models.
Run full XAI only after selecting best baseline and best proposed model.
```

Recommended XAI targets:

```text
best unimodal text model
best unimodal image model
best multimodal baseline
best final proposed multimodal model
```

Recommended XAI methods:

```text
Grad-CAM for image branch
attention/saliency for text branch
SHAP for modality contribution at fusion level
LIME for local perturbation-based explanation
```

Also require saving raw numeric explanation outputs, not only figures.

---

# REQUIRED PROMISING COMBINATION VALIDATION

After sequential ablation, include a final validation phase.

This phase should test several promising full configurations because greedy sequential ablation may not find the global best configuration.

Example:

```text
Best sequential model:
SigLIP + ViDeBERTa + GMU + Huber

Promising alternatives:
Swin-B + PhoBERT + GMU + Huber
ViSoBERT + EfficientNet-B3 + FiLM + Huber
CLIP visual encoder + Vietnamese text encoder + Cross-Attention + Weighted Huber
EVA-CLIP + mDeBERTa + Cross-Attention + Weighted Huber
ConvNeXt + PhoBERT + FiLM + Huber
```

Explain that this validates whether the selected components work well together or whether another combination has better synergy.

---

# REQUIRED WRITING QUALITY

The final `proposal.md` must be written in a professional, coherent, and logical manner.

It should clearly explain:

- the research methodology
- why each phase exists
- why each component is chosen
- what each ablation proves
- how conclusions will be drawn
- how experiments will be reproduced
- how results will be presented to a lecturer

Do not write a shallow checklist.

Write it as a real research engineering proposal.

The reader should understand not only **what** will be done, but also **why** it will be done.

---

# CONSTRAINTS

Do not modify existing code.

Do not create notebooks.

Do not create training scripts.

Only create:

```text
proposal.md
```

Do not hallucinate current implementation details.

If something is unknown, write:

```text
Unknown / Must be verified in codebase
```

Distinguish clearly between:

```text
Implemented
Partially implemented
Not implemented
Recommended future work
```

---

# FORMAT PRINCIPLE

The final `proposal.md` must be self-contained.

A future AI coding agent should be able to read it and understand:

- the project goal
- current weaknesses
- teammate’s experimental plan ideas worth reusing
- required infrastructure fixes
- experiment roadmap
- experiment folder structure
- Colab/Drive/GitHub workflow
- reproducibility requirements
- exact artifacts to generate
- final model selection process
- XAI strategy
- methodology and reasoning behind each choice

The proposal should be detailed but not bloated.

Prioritize clarity, reproducibility, traceability, and scientific defensibility.
