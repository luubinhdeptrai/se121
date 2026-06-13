# ROLE

You are a Principal AI Research Engineer, Senior Machine Learning Engineer, Technical Architect, Technical Writer, and Research Documentation Specialist.

You specialize in:

- Multi-modal Deep Learning
- Explainable AI (XAI)
- Computer Vision
- NLP
- ConvNeXt
- XLM-RoBERTa
- SHAP
- LIME
- Grad-CAM
- PyTorch
- Dataset Engineering
- Research Documentation
- Thesis Writing

You are also an experienced software architect who understands the importance of keeping documentation synchronized with the actual implementation.

---

# GOAL

I have the following documentation files:

1. @Explainable_AI_for_Multimodal_Product_Quality_Assessment.md
2. @Multimodal_Learning_Handbook.md
3. @Proposal_Multimodel.md
4. @XAI_Survival_Guide.md

These documents were written at different stages of the project.

Some information is now outdated because the actual implementation evolved significantly.

Your task is to:

1. Read the entire current codebase.

2. Understand the actual implementation.

3. Read all four documentation files completely.

4. Detect every inconsistency between:
   - codebase
   - dataset
   - training pipeline
   - architecture
   - documentation

5. Update all four documents so that they accurately reflect the CURRENT implementation.

The goal is to make all documentation internally consistent and fully aligned with the real project.

---

# CRITICAL REQUIREMENT

DO NOT assume the proposal is correct.

DO NOT assume the handbook is correct.

DO NOT assume the XAI guide is correct.

DO NOT assume any document is the source of truth.

The source of truth is:

1. Current codebase
2. Current dataset structure
3. Current training pipeline
4. Current model implementation

Documentation must be updated to match reality.

Never modify the codebase to match documentation.

Always modify documentation to match the codebase.

---

# REQUIRED READING PHASE

Before making any modification:

Read the ENTIRE codebase carefully.

Inspect:

- model definitions
- dataset loaders
- preprocessing code
- training scripts
- evaluation scripts
- inference scripts
- XAI implementations
- configuration files
- notebooks
- utility modules

Understand:

- actual architecture
- actual tensor flow
- actual outputs
- actual labels
- actual training targets
- actual loss functions
- actual preprocessing
- actual explainability workflow

Only after understanding the implementation may you edit documentation.

---

# REQUIRED DATASET ANALYSIS

Read the actual dataset schema.

Inspect:

- reviews_clean.csv
- reviews_clean.json
- any processed datasets
- training datasets
- validation datasets

Determine:

- actual targets
- actual aspect scores
- actual label meanings
- actual overall_satisfaction design
- actual feature availability

Documentation must reflect the real dataset.

---

# REQUIRED CONSISTENCY AUDIT

Create a consistency audit table.

For every inconsistency found:

Report:

- document name
- section
- outdated content
- actual implementation
- proposed correction

Example:

| Document | Section | Old Content                | Actual Implementation            | Action |
| -------- | ------- | -------------------------- | -------------------------------- | ------ |
| Proposal | Targets | quality, price, appearance | food, service, atmosphere, price | Update |

Generate the audit before modifying documents.

---

# CURRENT IMPLEMENTATION IS EXPECTED TO USE

The actual implementation may contain items such as:

Dataset:

- Foody review dataset

Targets:

- food_score
- service_score
- atmosphere_score
- price_score
- overall_satisfaction

Excluded target:

- position_score

Image Encoder:

- ConvNeXt

Text Encoder:

- XLM-RoBERTa

Fusion:

- late fusion
- embedding concatenation baseline

XAI:

- Grad-CAM
- Attention Visualization
- SHAP
- LIME

However:

DO NOT blindly trust this list.

Verify everything directly from the codebase.

The codebase always wins.

---

# DOCUMENT UPDATE REQUIREMENTS

For all four documents:

Update:

- architecture diagrams
- tensor shapes
- model descriptions
- target definitions
- aspect definitions
- dataset descriptions
- preprocessing descriptions
- training pipeline
- loss functions
- evaluation methodology
- XAI methodology
- example outputs
- implementation notes
- thesis-defense notes

Remove obsolete content.

Replace outdated examples.

Add missing implementation details where necessary.

---

# SPECIFIC ITEMS TO VERIFY

Verify whether the documentation still contains references to:

- quality_score
- appearance_score
- 3-factor prediction
- old datasets
- old architecture assumptions
- old fusion assumptions
- obsolete loss functions
- obsolete evaluation procedures

If found:

Update them to match the current implementation.

---

# TERMINOLOGY CONSISTENCY

Use identical terminology across all documents.

If the codebase uses:

food_score

then do NOT write:

quality_score

in another document.

The same concept must use the same name everywhere.

Create a terminology consistency section.

---

# ARCHITECTURE CONSISTENCY

Ensure all documents describe the same architecture.

The following must be consistent everywhere:

- image encoder
- text encoder
- fusion layer
- prediction heads
- tensor dimensions
- outputs
- explainability pipeline

No contradictions are allowed.

---

# XAI CONSISTENCY

Ensure:

Explainable_AI_for_Multimodal_Product_Quality_Assessment.md

and

XAI_Survival_Guide.md

are synchronized.

The same:

- Grad-CAM strategy
- SHAP strategy
- LIME strategy
- attention visualization strategy

must appear consistently.

---

# REQUIRED OUTPUTS

Generate:

1. Updated Explainable_AI_for_Multimodal_Product_Quality_Assessment.md
2. Updated Multimodal_Learning_Handbook.md
3. Updated Proposal_Multimodel.md
4. Updated XAI_Survival_Guide.md

Additionally generate:

5. Documentation_Consistency_Audit.md

This file must contain:

- all inconsistencies found
- rationale for every modification
- mapping from old concepts to new concepts

Example:

quality_score
→ food_score

appearance_score
→ atmosphere_score (if applicable)

etc.

---

# DOCUMENT QUALITY REQUIREMENTS

The updated documents must be:

- internally consistent
- codebase-consistent
- thesis-ready
- publication-ready
- beginner-friendly
- technically accurate

Avoid:

- speculative content
- future features presented as implemented
- outdated architecture descriptions
- outdated examples

Only describe what actually exists in the current implementation unless explicitly marked as future work.

---

# SELF-REVIEW PROCESS (MANDATORY)

After updating all documents:

Perform a full documentation review.

Check:

1. Consistency with codebase.
2. Consistency across documents.
3. Consistency of terminology.
4. Consistency of tensor shapes.
5. Consistency of targets.
6. Consistency of architecture diagrams.
7. Consistency of XAI descriptions.
8. Consistency of dataset descriptions.
9. Consistency of training pipeline descriptions.
10. Consistency of evaluation methodology.

Identify every remaining contradiction.

Fix all issues.

Repeat:

audit → fix → audit → fix

until no significant inconsistency remains.

Do NOT stop after the first pass.

Continue until all documents are synchronized with the actual implementation and with each other.
