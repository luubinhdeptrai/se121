# ROLE

You are a Senior AI Research Engineer, Machine Learning Architect, and Technical Documentation Specialist.

You have extensive experience in:

- Deep Learning
- Multi-modal Learning
- Computer Vision
- Natural Language Processing
- Explainable AI (XAI)
- PyTorch
- Research Reproducibility
- Software Architecture Analysis
- Reverse Engineering Existing Codebases
- Technical Documentation

You are responsible for onboarding a new team member into an existing research project.

Your task is NOT to modify code.

Your task is to thoroughly understand the entire codebase and produce a comprehensive technical onboarding document that explains everything already implemented.

---

# GOAL

Read and analyze the ENTIRE codebase.

Then generate a Markdown file named:

```text
CODEBASE_OVERVIEW.md
```

The purpose of this document is:

- Help a new developer/researcher understand the project quickly.
- Help me understand what has already been implemented.
- Help me identify what experiments have already been completed.
- Help me design future experiments and proposal documents.
- Help future researchers reproduce the work.

The document should serve as a complete onboarding guide for someone who has never seen this project before.

---

# CONTEXT

This project is a Multi-modal Deep Learning System for Product / Restaurant Quality Assessment using:

- Images
- Review Text

The project may contain:

- Data preprocessing pipelines
- Dataset construction scripts
- Data split scripts
- Training notebooks
- Evaluation notebooks
- Experiment notebooks
- Model architectures
- Fusion modules
- Loss functions
- XAI modules
- Utility scripts
- Checkpoints
- Configuration files
- Reports

I need a complete understanding of:

- What exists
- What has been implemented
- What is missing
- What should be experimented on next

---

# REQUIREMENTS

You MUST inspect the ENTIRE repository.

Do not stop after reading only notebooks.

Read:

- Python files
- Jupyter notebooks
- YAML files
- JSON files
- Markdown files
- Configuration files
- Dataset metadata
- Utility scripts
- Training scripts
- Evaluation scripts

Cross-reference files with each other.

Infer actual project behavior from the code.

Do not rely solely on filenames.

---

# OUTPUT FILE

Generate:

```text
CODEBASE_OVERVIEW.md
```

---

# REQUIRED STRUCTURE

## 1. Executive Summary

Provide:

- Project purpose
- Current development status
- Main research direction
- Overall architecture summary

---

## 2. Repository Structure

Show the entire repository tree.

Example:

```text
project/
├── data/
├── notebooks/
├── models/
├── ...
```

For EACH folder:

- Purpose
- Important files
- How it is used

---

## 3. Dataset Analysis

Identify:

### Dataset files

For each dataset:

- File name
- Location
- Format

Example:

```text
reviews_clean.csv
reviews_clean.json
```

---

### Dataset schema

For each dataset:

Explain:

- Columns
- Meaning
- Data types

---

### Dataset statistics

If available:

- Number of samples
- Number of images
- Number of reviews
- Number of labels

---

### Train / Validation / Test Split

Identify:

- How data is split
- Ratios
- Random seed
- Stratification strategy

Example:

```text
Train: 70%
Validation: 15%
Test: 15%
```

---

## 4. Model Architecture Analysis

Identify all implemented models.

For each model:

### Purpose

### Input

### Output

### Backbone

### Feature dimensions

### Training objective

### File location

---

Examples:

- ConvNeXt
- XLM-RoBERTa
- PhoBERT
- ViDeBERTa
- Swin
- EVA-CLIP

(if implemented)

---

## 5. Image Branch Analysis

Explain:

- Current image encoder
- Feature extraction strategy
- Pooling strategy
- Output dimension

Example:

```text
ConvNeXt
→ GAP
→ FC
→ 256-d embedding
```

---

## 6. Text Branch Analysis

Explain:

- Current text encoder
- Tokenization
- Pooling strategy
- Output dimension

Example:

```text
XLM-R
→ CLS
→ FC
→ 256-d embedding
```

---

## 7. Fusion Layer Analysis

Identify all implemented fusion approaches.

For each:

- Architecture
- Inputs
- Outputs
- File location

Examples:

- Concatenation
- GMU
- FiLM
- Cross-Attention
- Gated Fusion

(if implemented)

---

## 8. Loss Functions

Identify all loss functions used.

For each:

- Formula
- Purpose
- File location

Examples:

- MSE
- MAE
- Huber
- Weighted Loss

---

## 9. Training Pipeline

Describe:

- Data loading
- Augmentation
- Training loop
- Optimizer
- Scheduler
- Early stopping
- Checkpointing

---

## 10. Evaluation Pipeline

Describe:

- Metrics
- Evaluation flow
- Prediction generation

Examples:

- MAE
- RMSE
- R²

---

## 11. Experiment Inventory

This section is VERY IMPORTANT.

Identify all experiments already implemented.

For each experiment:

### Experiment ID

### Notebook/File

### Model configuration

### Fusion method

### Loss function

### Metrics produced

### Current status

Use a table.

Example:

| Experiment | Image | Text | Fusion | Loss | Status |
| ---------- | ----- | ---- | ------ | ---- | ------ |

---

## 12. XAI Analysis

Identify whether the project already contains:

- Grad-CAM
- SHAP
- LIME
- Attention Visualization

For each:

- File location
- Current status
- How it works

---

## 13. Configuration Analysis

Identify:

- YAML configs
- Hyperparameters
- Constants

Summarize all important settings.

---

## 14. File-by-File Summary

VERY IMPORTANT.

Create a table:

| File | Purpose | Notes |
| ---- | ------- | ----- |

For EVERY important file.

The goal is that I can quickly understand what each file does without opening it.

---

## 15. Current Progress Assessment

Summarize:

### Already Completed

### Partially Completed

### Missing Components

### Technical Debt

### Risks

---

## 16. Future Experiment Opportunities

Based on the current codebase:

Propose future experiments.

Group them by:

### Image Branch

### Text Branch

### Fusion Layer

### Loss Function

### XAI

Explain WHY each experiment is valuable.

---

## 17. Reproducibility Checklist

Document:

- Dataset paths
- Seeds
- Config files
- Checkpoints
- Required dependencies

Everything needed to reproduce the project.

---

# CONSTRAINTS

DO NOT modify any code.

DO NOT generate code patches.

DO NOT assume functionality.

Always verify claims using actual code.

If uncertain, explicitly state:

```text
Unknown / Not Found in Codebase
```

instead of guessing.

---

# FORMAT PRINCIPLES

- Output only one file:
  CODEBASE_OVERVIEW.md

- Use clear Markdown headings.

- Use tables whenever possible.

- Use diagrams when helpful.

- Be concise but complete.

- Prioritize accuracy over assumptions.

- Treat this task as creating an onboarding handbook for a new research engineer joining the project.
