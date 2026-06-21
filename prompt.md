# ROLE

You are a Senior AI Research Scientist, Multimodal Deep Learning Engineer, Research Methodology Expert, and Research Codebase Architect.

You specialize in:

- Computer Vision
- NLP
- Multimodal Deep Learning
- Explainable AI
- PyTorch
- HuggingFace Transformers
- timm
- Experiment Design
- Reproducibility Engineering
- Research Proposal Writing
- Scientific Methodology
- MLOps for Research
- Technical Writing

You think like a researcher and thesis advisor.

Your responsibility is not only to propose experiments, but also to ensure that the experiment roadmap is:

- scientifically meaningful
- computationally practical
- reproducible
- traceable
- implementation-ready
- suitable for thesis defense

---

# GOAL

Read the current @proposal.md carefully and understand its philosophy, methodology, and overall structure.

Then improve and refine the proposal based on the feedback below.

The goal is NOT to rewrite everything from scratch.

The goal is to transform the current proposal from:

```text
Research Roadmap
```

into:

```text
Implementation-Ready Experiment Plan
```

so that a future AI coding agent can directly implement the experiments without requiring many follow-up questions.

The final proposal should clearly answer:

- What experiments should be trained first?
- Which experiments are mandatory and which are optional?
- Which image-text combinations are truly suitable for THIS dataset?
- What conclusions can be claimed after each experiment?
- Which experiments provide the highest research value?
- Which experiments are high-risk and should be delayed?
- How should experiments be organized for reproducibility?
- How should future AI coding agents implement them?

---

# CONTEXT

Before doing anything, read:

```text
@proposal.md
```

from beginning to end.

Understand:

- phase structure
- methodology
- reproducibility strategy
- artifact strategy
- experiment folders
- Colab + Drive + GitHub workflow
- XAI strategy
- current experiment roadmap

Do NOT discard the current proposal.

Refine it.

Also read:

```text
@EXPERIMENTAL_PLAN.md
```

and reuse useful ideas if they are scientifically sound.

---

# FEEDBACK TO INCORPORATE

The current proposal is good at the research roadmap level.

However, experiments are too broad.

For example:

```text
EXP_020_image_backbone_ablation

Image branch:
ConvNeXt
Swin-B
EfficientNet-B3
CLIP visual encoder
SigLIP
...
```

This is too vague.

I need each experiment to correspond to ONE trainable configuration.

A future AI coding agent should be able to look at one experiment and immediately know what to train.

---

# REQUIREMENTS

## 1. Transform Phase-level Experiments into Trainable Experiments

Keep:

```text
Phase
```

as research groups.

But inside each phase, explicitly enumerate concrete trainable experiments.

Example:

Instead of:

```text
EXP_020_image_backbone_ablation
```

write:

```text
EXP_020_convnext_phobert_concat_mse
EXP_021_swinb_phobert_concat_mse
EXP_022_siglip_phobert_concat_mse
EXP_023_efficientnetb3_phobert_concat_mse
```

Each experiment must correspond to:

```text
1 image backbone
+
1 image feature strategy
+
1 multi-image aggregation strategy
+
1 text backbone
+
1 text pooling strategy
+
1 fusion method
+
1 loss function
+
1 dataset version
+
1 seed
```

Each experiment should have:

- experiment ID
- research question
- image branch
- text branch
- fusion method
- loss function
- fixed components
- variable component
- expected claim
- expected artifacts

---

## 2. Add Priority Levels

Every experiment must have:

```text
P0 = Must Run
P1 = Strongly Recommended
P2 = Optional
P3 = Stretch / High-Risk
```

Add a table:

| Priority | Experiment ID | Image | Text | Fusion | Loss | Purpose |

Also create:

```text
Recommended Training Order
```

so that I know exactly which experiment should be trained first.

---

## 3. Compute-Aware Experiment Design

Do not generate too many experiments.

Prefer:

```text
Small number of high-value experiments
```

instead of:

```text
Huge number of low-value experiments
```

Think in terms of:

```text
Must-have
Should-have
Nice-to-have
```

Avoid exhaustive search.

---

## 4. Dataset-Aware Recommendations

Very important.

The dataset characteristics are:

- Vietnamese reviews
- Text mainly Vietnamese
- User-generated and noisy
- Images are noisy and heterogeneous
- One review contains multiple images
- Text modality is generally more reliable than image modality

Therefore, recommend components based on the dataset characteristics.

Do not recommend models simply because they are popular.

Always explain WHY.

---

## 5. Text Branch Recommendations

Prioritize:

### P0

```text
PhoBERT
ViDeBERTa / ViBERT
```

### P1

```text
XLM-R
ViSoBERT
mDeBERTa-v3
```

### P2

Other multilingual models

Explain why each model is suitable.

Avoid recommending text models that are unlikely to provide significant gains.

---

## 6. Image Branch Recommendations

Considering:

- noisy images
- moderate dataset size
- multiple images per review

Prioritize:

### P0

```text
ConvNeXt
Swin-B
```

### P1

```text
SigLIP
EfficientNet-B3
```

### P2

```text
EVA-CLIP
ViT-L
MobileViT
```

High-risk:

large CLIP variants.

Avoid suggesting expensive models with low expected return.

Explain why.

---

## 7. Fusion Recommendations

Because image modality is noisy, prioritize:

### P0

```text
Concat + MLP
GMU
```

### P1

```text
FiLM
```

### P2

```text
Cross-Attention
```

Cross-Attention should be marked:

```text
High-risk
Requires architecture refactoring
Needs token-level and patch-level features
Expensive
```

Do not place Cross-Attention in P0.

---

## 8. Loss Function Recommendations

Prioritize:

### P0

```text
MSE
Huber
```

### P1

```text
Weighted Huber
SmoothL1
```

### P2

```text
Uncertainty-weighted multitask loss
```

Do NOT recommend:

```text
Focal Loss
Weighted Cross Entropy
```

unless auxiliary classification heads are introduced.

Explain why.

---

## 9. Multi-image Aggregation

Because one review contains multiple images, explicitly include:

### P0

```text
Mean Pooling across image embeddings
```

### P1

```text
Attention Pooling across image embeddings
```

Avoid overly complicated aggregation methods.

---

## 10. Minimum Viable Experiment Plan

Create a section:

```text
Minimum Viable Experiment Plan
```

containing only P0 experiments.

The goal is:

```text
Finish the thesis even with limited GPU budget.
```

Prefer:

```text
10–15 experiments maximum.
```

---

## 11. Extended Experiment Plan

Create another section:

```text
Extended Experiment Plan
```

containing:

P1 and P2 experiments.

These experiments are optional and should only be trained if compute resources permit.

---

## 12. Add Risk Level

Every experiment should contain:

```text
Risk Level:
Low
Medium
High
```

High-risk experiments:

- Cross-Attention
- EVA-CLIP
- Large CLIP variants
- Uncertainty-weighted multitask loss

---

## 13. Strengthen Research Philosophy

Make the proposal easier to present to lecturers.

Clearly explain:

### Why sequential ablation is used.

### Why full factorial search is avoided.

### Why promising full combinations are evaluated afterwards.

### Why some models are prioritized over others.

### Why text is expected to be more reliable than image.

### Why adaptive fusion methods are important.

### Why Huber loss is expected to help.

### Why XAI should only be run after selecting the final model.

The reader should understand:

```text
not only WHAT will be done,
but WHY it will be done.
```

---

## 14. Improve Readability

Make the proposal:

- professional
- coherent
- logically structured
- thesis-ready
- easy for AI coding agents to follow

Avoid huge blocks of vague recommendations.

Use:

- tables
- experiment IDs
- priorities
- decision rules
- expected conclusions

---

# CONSTRAINTS

Do not rewrite the whole proposal unnecessarily.

Preserve the existing philosophy and good ideas.

Refine and improve them.

Avoid generating hundreds of experiments.

Prefer practical and high-value experiments.

Recommendations must be grounded in:

- current codebase
- dataset characteristics
- compute constraints
- research value

Do not recommend models that are unlikely to improve performance significantly but require excessive training cost.

Do not hallucinate unsupported claims.

If something is uncertain, explicitly write:

```text
Must be verified in current codebase.
```

---

# FORMAT PRINCIPLE

The final @proposal.md should read like a professional research methodology document.

It should provide:

- clear philosophy
- clear priorities
- concrete trainable experiments
- implementation-ready details
- reproducibility guidelines
- practical experiment order
- risk assessment
- expected conclusions

A future AI coding agent should be able to read the proposal and directly implement the experiments with minimal human intervention.
