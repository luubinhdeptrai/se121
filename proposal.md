# Experiment Proposal: Explainable Multimodal Deep Learning for Vietnamese Restaurant Review Quality Assessment

## 1. Purpose

This proposal defines an implementation-ready experiment roadmap for the current multimodal regression project. The project predicts restaurant review quality scores from:

- review text, mainly Vietnamese user-generated content
- one or more review images
- five regression targets: `overall_satisfaction`, `food_score`, `service_score`, `atmosphere_score`, and `price_score`

The goal is not to randomly try many model pairs. The goal is to build a scientifically defensible research path that first makes the experiment system reproducible, then isolates the contribution of image encoders, text encoders, fusion modules, loss functions, and explainability analysis.

The guiding methodology is:

```text
Controlled Sequential Ablation + Promising Combination Validation
```

This means:

1. Choose reasonable fixed components first.
2. Ablate one component while holding the others constant.
3. Select the best variant using a predefined validation rule.
4. Replace the old component with the selected variant.
5. Move to the next component.
6. Finally test several promising full combinations because greedy ablation may miss component synergy.

This approach is stronger than trying many model pairs without structure because every result answers a clear research question. It is also more practical than full factorial search. For example, testing 8 image encoders, 6 text encoders, 5 fusion methods, 8 loss choices, 3 pooling choices, and 3 seeds would require 17,280 runs before hyperparameter tuning. That is unnecessary for a university-scale project and unrealistic on Google Colab.

## 2. Current Codebase Evidence

The repository already contains a baseline multimodal research pipeline, but it is not yet ready for serious comparative experiments.

### Implemented

- Data artifacts in `data_raw/` and `data_processed/`.
- Rule-based `overall_satisfaction` generation with `overall_satisfaction_rules.json` and `overall_satisfaction_rule_analysis.md`.
- `preprocess_data.py` merges multimodal rows with enhanced review labels and groups multiple image URLs by `review_id`.
- `MultimodalDataset` loads text, up to `max_images=4` images, `num_images`, and a five-target score vector.
- `TextModel`, `ImageModel`, and `FusionModel` exist.
- `ImageModel` supports multi-image mean pooling with a `num_images` mask during training and validation.
- `Trainer.py` trains with AdamW, plain vector MSE, gradient clipping, cosine warmup scheduler, gradient accumulation, early stopping, and best-checkpoint saving.
- Historical executed notebooks report results for `xlm-roberta-base + ConvNeXt` and `microsoft/mdeberta-v3-base + SigLIP`.

### Partially Implemented

- Early stopping and scheduler exist in `Trainer.py`, but they are not part of a reproducible config/logging/export framework.
- Layer unfreezing exists for specific XLM-R/ConvNeXt-like structures, but support for Swin, ViT, SigLIP, CLIP, EfficientNet, and other models must be verified.
- Colab workflow exists, but outputs are inconsistent and not a complete artifact-management system.
- Multi-image mean pooling exists, but alternative multi-image pooling and image-quality filtering are not implemented.
- Documentation mentions strong experiment ideas, but some docs are stale relative to current code.

### Not Implemented

- Deterministic runtime seed utility for Python, NumPy, PyTorch, CUDA, and DataLoader workers.
- Sample-wise metric aggregation and `predictions.csv` export.
- Mixed precision training.
- Resume-from-checkpoint with optimizer and scheduler state.
- Experiment logger integration.
- Versioned `config.yaml` per experiment.
- Robust losses such as Huber, SmoothL1, Log-Cosh, weighted Huber, or uncertainty-weighted multi-task loss.
- Fusion modules beyond concatenation + MLP.
- Model-side XAI modules: Grad-CAM, attention/saliency, SHAP, and LIME.
- Committed final split files under `data/text/` or `data/splits/`.
- Committed image cache under `data/image/`.
- Committed checkpoints or checkpoint links.

### Unknown / Must be verified in codebase

- Exact canonical final train/validation/test split version.
- Exact final image cache and its failure manifest.
- Exact checkpoint artifacts for historical runs.
- Exact package versions used by historical notebooks.
- Whether every proposed timm backbone has correct preprocessing and feature shapes under current code.

## 3. Critical Review of `doc/EXPERIMENTAL_PLAN.md`

The teammate plan is useful but must be adapted carefully.

Ideas worth reusing:

- The backbone candidates are plausible: CLIP visual encoder, EfficientNet-B3, SigLIP2, Swin-B, ViSoBERT, and DeBERTa-family text models.
- Robust losses such as Huber and Log-Cosh are appropriate for noisy regression labels.
- Homoscedastic uncertainty weighting is a valid multi-task loss candidate.
- Gated fusion, FiLM, and cross-attention are relevant because images are noisy and modality reliability can vary sample by sample.

Ideas that must be softened:

- Several cited results are classification results, not regression results. They should support hypotheses, not guarantee performance on Vietnamese restaurant-score regression.
- RoBERTa + CLIP is a useful reference pair, but generic English RoBERTa is not automatically ideal for Vietnamese Foody reviews.
- DeBERTa + SigLIP2 may transfer from e-commerce/product classification, but it must be evaluated as a candidate, not as an expected winner.
- Benchmark expectations in the plan are too strong because historical metrics are batch-averaged, split/checkpoint artifacts are missing, and some notebook outputs come from older code.
- The docs use the term "joint loss", but current code trains with plain vector MSE over five targets.

This proposal integrates the useful hypotheses while restructuring them into a controlled phase-based roadmap.

## 4. Engineering Hygiene vs Research Contribution

Engineering hygiene must be fixed before claims are made. These items make experiments trustworthy but are not the main research contribution:

- deterministic seed control
- backbone-specific preprocessing
- sample-wise metric aggregation
- config and artifact saving
- logging
- early stopping and scheduler standardization
- mixed precision
- checkpoint/resume
- environment pinning
- no silent data re-splitting or image re-downloading

Research contribution begins after the infrastructure is stable:

- image backbone selection
- text backbone selection for noisy Vietnamese reviews
- fusion design for unreliable image evidence
- loss design for noisy multi-target regression
- XAI analysis of image, text, and modality contribution

## 5. Architecture Under Study

```text
Review Text
  -> Text Encoder
  -> Text Feature Extraction / Pooling
  -> Text Embedding
                                  \
                                   Fusion Layer
                                  /
Image(s)
  -> Image Encoder
  -> Image Feature Extraction / Pooling
  -> Image Embedding
  -> Regression Heads
  -> overall_satisfaction, food_score, service_score, atmosphere_score, price_score
```

The current code uses:

- Text branch: Hugging Face `AutoModel`, first-token or `pooler_output` pooling.
- Image branch: `timm.create_model(..., num_classes=0)`, then mean pooling across review images.
- Fusion branch: concatenation of raw text and image encoder features, then MLP.
- Loss: `nn.MSELoss()` over the five-output vector.

Important implementation note: do not describe ConvNeXt as using CLS pooling. ConvNeXt is CNN-style and does not have a CLS token in the same way ViT-like models do. CLS-style extraction is appropriate for ViT, CLIP ViT, SigLIP, and similar transformer encoders.

## 6. Required Metrics and Evaluation Protocol

All serious experiments must compute metrics sample-wise, not by unweighted averaging of batch metrics.

For validation and test:

1. Collect all predictions and ground truths across the full split.
2. Concatenate them into arrays of shape `[num_samples, 5]`.
3. Compute metrics from the complete arrays.
4. Export `predictions.csv` and `metrics.json`.

Required metrics:

- MAE
- RMSE
- R2
- per-target MAE
- per-target RMSE
- `overall_satisfaction` MAE
- average aspect MAE across `food_score`, `service_score`, `atmosphere_score`, and `price_score`
- mean five-target MAE

Do not average metric values across batches unless weighted by batch size.

`predictions.csv` must include at least:

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

Primary selection criterion:

- lowest validation mean five-target MAE

Tie breakers:

1. lower `overall_satisfaction` MAE
2. lower average aspect MAE
3. lower mean RMSE
4. smaller and cheaper model if performance is statistically similar

The test set must be used only after final model selection.

## 7. Experiment Roadmap Table

| Phase | Experiment ID | Experiment Name | Variable Component | Fixed Components | Purpose | Expected Claim | Priority |
|---|---|---|---|---|---|---|---|
| 0 | EXP_000 | Infrastructure rebuild | Engineering pipeline | Current baseline architecture | Make experiments reproducible | Future results are trustworthy and replayable | Critical |
| 0 | EXP_001 | Frozen dataset artifact | Dataset split and image cache | Data sources | Create frozen split v1 | All runs use identical samples and images | Critical |
| 0 | EXP_002 | Metric/export validation | Evaluation implementation | Baseline checkpoints or smoke model | Replace batch-average metrics | Metrics are exact sample-wise values | Critical |
| 1 | EXP_010 | Text-only baseline | Modality | Current text branch, MSE | Quantify text signal | Text is expected to be strong | Critical |
| 1 | EXP_011 | Image-only baseline | Modality | Current image branch, MSE | Quantify visual signal | Image-only is useful but noisy | Critical |
| 1 | EXP_012 | Multimodal concat baseline | Modality/fusion baseline | XLM-R + ConvNeXt + concat + MSE | Establish reference model | Fusion should beat or match unimodal baselines | Critical |
| 2 | EXP_020 | Image backbone ablation | Image backbone | Best text, concat, MSE/Huber baseline | Compare visual encoders | Some backbones handle noisy images better | High |
| 2 | EXP_021 | Multi-image pooling ablation | Image pooling | Selected image backbone | Compare mean vs attention pooling | Review-level image aggregation matters | High |
| 2 | EXP_022 | Image quality filtering | Image filtering | Selected image backbone/pooling | Reduce noisy visual evidence | Filtering may improve robustness | Medium |
| 3 | EXP_030 | Text backbone ablation | Text backbone | Selected image branch and concat | Test Vietnamese-specific encoders | Vietnamese/social text models may improve text signal | High |
| 3 | EXP_031 | Text pooling and length | Pooling/max length | Selected text backbone | Improve text representation | Mean/attention pooling or length 256 may help | High |
| 3 | EXP_032 | Text normalization ablation | Text normalization | Selected text backbone | Handle noisy Vietnamese text | Light normalization may improve robustness | Medium |
| 4 | EXP_040 | Gated fusion ablation | Fusion method | Selected text/image branches | Learn modality reliability | Gating helps with noisy images | High |
| 4 | EXP_041 | FiLM and cross-attention | Fusion method | Selected text/image branches | Test richer interaction | Cross-modal interaction may improve hard cases | Medium |
| 5 | EXP_050 | Robust loss ablation | Loss function | Selected architecture | Handle noisy labels/outliers | Huber/SmoothL1/Log-Cosh may beat MSE | High |
| 5 | EXP_051 | Multi-task loss balancing | Task weights | Selected architecture | Balance five targets | Weights, not Huber alone, address task imbalance | High |
| 6 | EXP_060 | Promising combination validation | Full configuration | Frozen split and infrastructure | Test component synergy | A non-greedy combination may outperform sequential best | High |
| 7 | EXP_070 | Final multi-seed selection | Random seed | Best candidates | Measure stability | Final model is robust across seeds | Critical |
| 7 | EXP_071 | Locked test evaluation | Final selected model | Frozen test split | Report final performance | Honest final metrics for thesis | Critical |
| 8 | EXP_080 | Lightweight XAI sanity check | XAI methods | 1-2 baseline models | Verify explanation tooling | XAI wrappers target correct tensors | Medium |
| 8 | EXP_081 | Full final XAI analysis | XAI case studies | Best baseline and final model | Explain model behavior | Final model is inspectable, not fully transparent | High |
| 9 | EXP_090 | Thesis-ready packaging | Reporting artifacts | Final results | Package reproducible evidence | Lecturer can audit the work | Critical |

## 8. Phase Details

### Phase 0: Research Infrastructure and Reproducibility Setup

Goal: create a reliable experiment platform before making research claims.

Motivation: current historical metrics are hard to reproduce because final split files, image cache, checkpoints, configs, predictions, and package versions are not committed or linked. Current evaluation also computes batch-averaged metrics. In addition, `test.py` does not pass `num_images` into image/fusion inference, so test-time multi-image pooling may differ from training/validation.

Fixed components:

- Current baseline architecture.
- Existing data sources in `data_raw/` and `data_processed/`.

Variable component:

- Engineering infrastructure, not model research.

Experiment list:

- `EXP_000_infrastructure_rebuild`
- `EXP_001_frozen_dataset_artifact`
- `EXP_002_metric_export_validation`

Implementation notes:

- Add a seed utility for Python `random`, NumPy, PyTorch CPU/GPU, CUDA determinism where feasible, and DataLoader worker seeds.
- Use a model/backbone preprocessing registry. For timm models, prefer timm data config and transforms unless a correct Hugging Face processor is explicitly available.
- Ensure test and validation pass `num_images` into `ImageModel` and `FusionModel`.
- Replace metric aggregation with sample-wise computation.
- Add `config.yaml`, `metrics.json`, `predictions.csv`, `error_analysis.csv`, `train.log`, training curves, and checkpoint links per run.
- Add mixed precision through `torch.cuda.amp` or the current PyTorch AMP API when CUDA is available.
- Save full checkpoints with model, optimizer, scheduler, epoch, best metric, config, and seed.
- Implement resume-from-checkpoint.
- Do not silently re-split data or redownload images. Save a split manifest and image failure manifest.

Expected output files:

- `data/splits/train.csv`
- `data/splits/val.csv`
- `data/splits/test.csv`
- `data/splits/split_manifest.json`
- `data/image_manifest.csv`
- `configs/base.yaml`
- `experiments/EXP_000_infrastructure_rebuild/README.md`
- `experiments/EXP_002_metric_export_validation/metrics.json`
- `experiments/EXP_002_metric_export_validation/predictions.csv`

Decision rule:

- No research experiment may start until the same config and seed can be rerun with matching validation metrics within a small tolerance.

Expected claim/conclusion:

- This phase does not claim model novelty. It establishes that later claims are reproducible and fair.

### Phase 1: Baseline Establishment

Goal: establish text-only, image-only, and multimodal baselines on the frozen split.

Motivation: multimodal learning is meaningful only if it is compared against both unimodal branches. Because text is expected to be strong and images are heterogeneous, the baseline must quantify whether images add signal or noise.

Fixed components:

- Frozen split v1.
- Seed 42 for first baseline run.
- AdamW.
- Same metric script.
- Plain vector MSE as the initial baseline loss.
- Backbone-specific preprocessing.

Variable component:

- Modality and baseline architecture.

Experiment list:

- `EXP_010_text_only_baseline`
- `EXP_011_image_only_baseline`
- `EXP_012_multimodal_concat_baseline`

Required baseline definitions:

- Baseline 0.1: Text-only baseline. Use XLM-R, PhoBERT, or the current text baseline with MSE first.
- Baseline 0.2: Image-only baseline. Use ConvNeXt or the current image baseline with MSE first.
- Baseline 0.3: Multimodal baseline. Use ConvNeXt + XLM-R, concatenation + MLP fusion, and MSE first.

Every later experiment must be compared against these baselines, not only against the immediately previous ablation.

Implementation notes:

- Text-only baseline should use current text default, likely `xlm-roberta-base`, unless code verification changes the default.
- Image-only baseline should use current image default, `convnext_base_in22k` or its current timm equivalent.
- Multimodal baseline should use ConvNeXt + XLM-R + concatenation MLP.
- Record historical notebook metrics only as background evidence, not as official baseline results.

Expected output files:

- one experiment folder per baseline
- `config.yaml`
- `metrics.json`
- `predictions.csv`
- `error_analysis.csv`
- `train.log`
- `figures/loss_curve.png`
- `figures/mae_by_target.png`
- `checkpoint_link.md`

Decision rule:

- Use validation mean five-target MAE as primary metric.
- A multimodal baseline is useful only if it improves over text-only or offers complementary error behavior without severe target degradation.

Expected claim/conclusion:

- The baseline phase should support a careful statement such as: "On frozen split v1, text-only provides the strongest unimodal signal; image-only is weaker but may add complementary information in selected cases; concatenation is the controlled multimodal baseline."

### Phase 2: Image Branch Ablation

Goal: identify a strong and practical image branch under fixed text, fusion, and loss settings.

Motivation: review images are noisy and diverse. They include food, drinks, menus, receipts, packaging, environment photos, people, and blurry images. A better visual encoder may help, but only if preprocessing and review-level image aggregation are correct.

Fixed components:

- Best Phase 1 text branch.
- Concatenation + MLP fusion.
- Baseline loss, initially MSE unless Phase 1 shows strong instability.
- Frozen split v1.
- Sample-wise metrics.

Variable component:

- Image backbone, image feature strategy, multi-image pooling, and filtering.

Experiment list:

- `EXP_020_image_backbone_ablation`
- `EXP_021_multi_image_pooling_ablation`
- `EXP_022_image_quality_filtering_ablation`

Implementation notes:

- Compare ConvNeXt, Swin-B, EfficientNet-B3, CLIP visual encoder, SigLIP/SigLIP2, EVA-CLIP if feasible, ViT-L if resources allow, and MobileViT if lightweight deployment matters.
- Use Global Average Pooling or timm-provided pooled features for CNN-like models.
- Use CLS/patch-token strategies only for ViT-like models that actually expose such tokens.
- Test multi-image mean pooling against attention pooling across images.
- Image quality filtering should be recorded with a manifest, not silent deletion.

Expected output files:

- per-backbone experiment folders
- `image_preprocessing_report.md`
- `image_failure_manifest.csv`
- `image_pooling_ablation.csv`

Decision rule:

- Select the image branch that improves validation mean MAE or reduces image-sensitive errors without harming text-dominant targets.
- If a larger model gives only tiny improvement, prefer the simpler model for Colab reproducibility.

Expected claim/conclusion:

- This phase should support a bounded claim: "Under fixed text/fusion/loss settings, image backbone X is the most reliable visual encoder for noisy Foody-style review images."

### Phase 3: Text Branch Ablation

Goal: identify the best text representation for noisy Vietnamese review text.

Motivation: the dataset is mainly Vietnamese, informal, and user-generated. Vietnamese-specific or social-media-oriented encoders may outperform general multilingual baselines.

Fixed components:

- Best image branch from Phase 2.
- Concatenation + MLP fusion.
- Baseline or currently selected loss.
- Frozen split v1.

Variable component:

- Text backbone, pooling, max length, and normalization.

Experiment list:

- `EXP_030_text_backbone_ablation`
- `EXP_031_text_pooling_length_ablation`
- `EXP_032_text_normalization_ablation`

Implementation notes:

- Compare XLM-RoBERTa, PhoBERT, ViDeBERTa/ViBERT if available, ViSoBERT, mDeBERTa-v3, and RoBERTa only as a reference baseline.
- Test first-token/CLS pooling, mean pooling over non-padding tokens, attention pooling, and last-layer or last-four-layer aggregation.
- Test max text length 128 vs 256. Current `Config.py` default is 256, while older docs and some templates mention 128.
- Text normalization must be conservative. Do not remove sentiment-bearing slang, negation, or emojis without an ablation.

Expected output files:

- per-text-backbone experiment folders
- `tokenization_report.md`
- `text_length_coverage.json`
- `normalization_examples.csv`

Decision rule:

- Select the text branch with best validation mean MAE and stable performance on `overall_satisfaction`.
- Require qualitative inspection of high-error text examples before finalizing.

Expected claim/conclusion:

- This phase should show whether Vietnamese-specific pretraining or social-media pretraining improves regression over XLM-R.

### Phase 4: Fusion Layer Ablation

Goal: test whether smarter fusion improves beyond concatenation.

Motivation: images are noisy and not always relevant. Fusion should learn when to trust text, when to trust images, and when the two modalities disagree.

Fixed components:

- Best image branch from Phase 2.
- Best text branch from Phase 3.
- Selected loss from current baseline, usually MSE until Phase 5.
- Frozen split v1.

Variable component:

- Fusion method.

Experiment list:

- `EXP_040_fusion_gating_ablation`
- `EXP_041_film_cross_attention_ablation`

Implementation notes:

- Concatenation + MLP remains the baseline.
- Late weighted averaging is a simple sanity comparison.
- GMU is useful because modality reliability differs by sample.
- Gated cross-modal fusion is useful because many review images are not aligned with the score target.
- FiLM lets one modality modulate another, for example text can condition image features.
- Cross-attention can model token-patch interaction, but requires token/patch features, careful memory control, and more implementation work.

Expected output files:

- fusion experiment folders
- `fusion_gate_statistics.csv` for gated methods
- `modality_weight_distribution.png`
- `fusion_error_cases.md`

Decision rule:

- Prefer a fusion method only if it beats concatenation on validation mean MAE and does not collapse to one modality without explanation.
- Cross-attention must justify its extra complexity with clear improvement or stronger XAI value.

Expected claim/conclusion:

- This phase should determine whether adaptive fusion improves noisy multimodal regression compared with fixed concatenation.

### Phase 5: Loss Function Ablation

Goal: improve robustness to noisy labels, outliers, and multi-task imbalance.

Motivation: MSE is sensitive to outliers. The targets may have different noise levels and difficulty. `overall_satisfaction` is engineered from aspect scores plus rule-based text evidence, so it should be evaluated carefully as a weak but auditable target.

Fixed components:

- Best architecture from Phases 2 to 4.
- Frozen split v1.
- Same optimizer and scheduler.

Variable component:

- Loss function and target weighting.

Experiment list:

- `EXP_050_robust_regression_loss_ablation`
- `EXP_051_multitask_loss_balancing`

Implementation notes:

- MSE is the baseline.
- MAE is robust but may optimize less smoothly.
- Huber, SmoothL1, and Log-Cosh are appropriate for noisy regression labels and outliers.
- Weighted Huber combines robustness with task weighting.
- True multi-task weighted loss should expose weights for each target.
- Homoscedastic uncertainty-weighted multi-task loss is feasible if implemented carefully and monitored for degenerate weights.
- Huber itself does not solve multi-task balancing. Task weights solve balancing.
- Focal loss and weighted cross-entropy are not suitable for the main task unless an auxiliary classification head is added.

Expected output files:

- loss experiment folders
- `loss_weight_history.csv` for learnable weighting
- `outlier_error_analysis.csv`
- `per_target_tradeoff.md`

Decision rule:

- Select the loss that improves validation mean MAE and reduces large-error tails without sacrificing `overall_satisfaction`.
- If a weighted loss improves one target while harming others, report it as a trade-off rather than a universal improvement.

Expected claim/conclusion:

- This phase should show whether robust regression and target balancing improve noisy multi-target score prediction.

### Phase 6: Promising Full-Combination Validation

Goal: test several complete configurations to catch component synergy that sequential ablation might miss.

Motivation: the best individual image encoder, text encoder, fusion module, and loss may not form the best full system. Some encoders pair better with specific fusion designs.

Fixed components:

- Frozen split v1.
- Reproducible infrastructure.
- Same metric script and artifact template.

Variable component:

- Full architecture combination.

Experiment list:

- `EXP_060_promising_combination_validation`

Promising combinations to test:

- Best sequential model from Phases 2 to 5.
- `ConvNeXt + XLM-R + concat MLP + MSE` as the official baseline anchor.
- `ConvNeXt + PhoBERT + FiLM + Huber`.
- `Swin-B + PhoBERT + GMU + Huber`.
- `ViSoBERT + EfficientNet-B3 + FiLM + Huber`.
- `CLIP visual encoder + Vietnamese text encoder + Gated Fusion + Weighted Huber`.
- `SigLIP/SigLIP2 + mDeBERTa-v3 or ViDeBERTa + GMU/Cross-Attention + Weighted Huber`.
- `EVA-CLIP + ViDeBERTa + Cross-Attention + Weighted Huber`, only if feasible on Colab.
- `MobileViT + PhoBERT + GMU + Huber`, if lightweight deployment is relevant.

Expected output files:

- full-combination experiment folders
- `combination_leaderboard.csv`
- `resource_usage.csv`

Decision rule:

- Select top 2 or 3 candidates for multi-seed validation.
- A combination must be judged by both performance and reproducibility cost.

Expected claim/conclusion:

- This phase validates whether selected components work well together or whether another combination has better synergy.

### Phase 7: Final Model Selection

Goal: select and evaluate the final model fairly.

Motivation: a single lucky seed can mislead conclusions, especially with small or noisy datasets.

Fixed components:

- Frozen split v1.
- Top candidates from Phase 6.
- Finalized training infrastructure.

Variable component:

- Random seed and candidate configuration.

Experiment list:

- `EXP_070_final_multiseed_selection`
- `EXP_071_locked_test_evaluation`

Implementation notes:

- Run at least seeds 42, 123, and 2026 for top candidates if compute allows.
- Select final model using validation metrics only.
- Run test evaluation once for the selected final model.
- Report mean and standard deviation across seeds on validation, and final test metrics for the selected checkpoint.

Expected output files:

- `final_leaderboard.csv`
- `final_test_metrics.json`
- `final_predictions.csv`
- `final_model_card.md`

Decision rule:

- Select the model with best mean validation performance and acceptable variance.
- Test set is locked until the final choice.

Expected claim/conclusion:

- The final selected model is not merely the best single run; it is stable enough to defend.

### Phase 8: Explainable AI Analysis

Goal: inspect what the models learned and whether the final system uses plausible evidence.

Motivation: the project is explainable multimodal learning, but full XAI for every experiment is unnecessary and expensive. XAI should be targeted.

Fixed components:

- Best unimodal text model.
- Best unimodal image model.
- Best multimodal baseline.
- Best final proposed multimodal model.

Variable component:

- XAI method and case-study set.

Experiment list:

- `EXP_080_xai_sanity_checks`
- `EXP_081_final_xai_analysis`

Implementation notes:

- Run lightweight sanity-check XAI early on 1-2 baseline models.
- Run full XAI only after selecting the best baseline and best final proposed model.
- Use Grad-CAM for image branch spatial evidence.
- Use attention/saliency for text branch evidence, with a caution that attention is diagnostic and not full causal explanation.
- Use SHAP for modality contribution at the fusion level.
- Use LIME for local perturbation-based checks.
- Explain one target score at a time.
- Hold the other modality fixed when explaining one modality.
- Save raw numeric outputs, not only figures.

Recommended XAI targets:

- best unimodal text model
- best unimodal image model
- best multimodal baseline
- best final proposed multimodal model

Expected output files:

- `xai/gradcam/*.png`
- `xai/gradcam/*.npy`
- `xai/attention/*.png`
- `xai/attention/*.csv`
- `xai/shap/*.json`
- `xai/shap/*.npy`
- `xai/lime/*.json`
- `xai/lime/*.png`
- `xai_case_studies.md`

Decision rule:

- XAI is not used to select the model unless two models are otherwise tied.
- XAI is used to validate plausibility, diagnose branch collapse, and prepare thesis defense.

Expected claim/conclusion:

- The final system is not fully transparent, but it is inspectable at image, text, fusion, and local perturbation levels.

### Phase 9: Final Experiment Packaging and Thesis-Ready Reporting

Goal: package results so a lecturer or future AI coding agent can audit and reproduce them.

Motivation: research quality depends on traceability, not only model performance.

Fixed components:

- Final selected model.
- Final artifacts.

Variable component:

- Reporting format and artifact organization.

Experiment list:

- `EXP_090_thesis_ready_packaging`

Implementation notes:

- Create a final report table with baselines, ablations, full combinations, and final model.
- Include model limitations and failure taxonomy.
- Include checkpoint links and exact environment export.
- Include a short reproduction command per experiment.

Expected output files:

- `reports/final_experiment_report.md`
- `reports/final_leaderboard.csv`
- `reports/error_taxonomy.md`
- `reports/xai_case_studies.md`
- `environment.yml`
- pinned `requirements.txt`

Decision rule:

- Packaging is complete only if a future agent can reproduce the final metric table from configs and linked artifacts.

Expected claim/conclusion:

- The project is ready for lecturer explanation and thesis-style reporting.

## 9. Experiment Specifications

The following specifications define the minimum required details for implementation. All experiments use frozen split v1, sample-wise metrics, seed 42 for first run, and the standard artifact folder unless otherwise stated.

### EXP_010_text_only_baseline

Research question: How strong is review text alone for predicting the five targets?

- Image branch: disabled.
- Image internal variant: not applicable.
- Text branch: `xlm-roberta-base` or current text baseline.
- Text internal variant: first-token or `pooler_output` pooling, max length from config.
- Fusion method: none.
- Loss function: MSE.
- Fixed components: frozen split v1, AdamW, scheduler, early stopping, seed 42.
- Variable component: modality is text only.
- Training settings: 20 epochs maximum, patience 3-5, batch size based on GPU memory, gradient clipping 1.0.
- Expected artifacts: `config.yaml`, `metrics.json`, `predictions.csv`, `train.log`, loss curve, checkpoint link.
- Evaluation metrics: MAE, RMSE, R2, per-target metrics.
- Selection criterion: establishes baseline, not selected against another text model yet.
- Expected conclusion: text should be a strong unimodal signal for Vietnamese reviews.

### EXP_011_image_only_baseline

Research question: How much predictive signal exists in review images alone?

- Image branch: ConvNeXt current baseline.
- Image internal variant: backbone pooled features plus multi-image mean pooling.
- Text branch: disabled.
- Text internal variant: not applicable.
- Fusion method: none.
- Loss function: MSE.
- Fixed components: frozen split v1, AdamW, scheduler, early stopping, seed 42.
- Variable component: modality is image only.
- Training settings: use smaller batch size if GPU memory requires it.
- Expected artifacts: same standard experiment files plus image failure summary.
- Evaluation metrics: sample-wise MAE, RMSE, R2, per-target metrics.
- Selection criterion: compare against text-only and multimodal baseline.
- Expected conclusion: image-only may be weaker because images are heterogeneous, but it quantifies visual contribution.

### EXP_012_multimodal_concat_baseline

Research question: Does simple multimodal fusion improve over text-only and image-only baselines?

- Image branch: ConvNeXt current baseline.
- Image internal variant: multi-image mean pooling.
- Text branch: XLM-R current baseline.
- Text internal variant: first-token or pooler pooling.
- Fusion method: concatenation + MLP.
- Loss function: MSE.
- Fixed components: frozen split v1, seed 42, same training infrastructure.
- Variable component: fusion of both modalities.
- Training settings: staged or joint training must be recorded exactly.
- Expected artifacts: standard experiment files.
- Evaluation metrics: sample-wise MAE, RMSE, R2, per-target metrics.
- Selection criterion: should beat or complement unimodal baselines.
- Expected conclusion: concatenation becomes the main reference for later fusion experiments.

### EXP_020_image_backbone_ablation

Research question: Which image encoder gives the best visual representation under fixed text, fusion, and loss settings?

- Image branch: ConvNeXt, Swin-B, EfficientNet-B3, CLIP visual encoder, SigLIP/SigLIP2, EVA-CLIP if feasible, ViT-L if feasible, MobileViT if resource-constrained.
- Image internal variant: correct backbone-specific preprocessing and pooled feature extraction.
- Text branch: best Phase 1 text branch.
- Text internal variant: fixed.
- Fusion method: concatenation + MLP.
- Loss function: MSE or current selected baseline loss.
- Fixed components: frozen split v1, seed 42, same training settings.
- Variable component: image backbone.
- Training settings: same epochs and early stopping; adjust batch size only if documented.
- Expected artifacts: standard files plus `image_preprocessing_report.md`.
- Evaluation metrics: sample-wise MAE, RMSE, R2, resource usage.
- Selection criterion: validation mean MAE with resource-aware tie breaker.
- Expected conclusion: identifies the most reliable image backbone for noisy review photos.

### EXP_021_multi_image_pooling_ablation

Research question: Does attention pooling across review images outperform simple mean pooling?

- Image branch: selected Phase 2 backbone.
- Image internal variant: multi-image mean pooling, multi-image attention pooling, optional top-k image pooling.
- Text branch: fixed selected text branch.
- Text internal variant: fixed.
- Fusion method: concatenation + MLP.
- Loss function: fixed.
- Fixed components: dataset, seed, optimizer, metrics.
- Variable component: review-level image aggregation.
- Training settings: identical where possible.
- Expected artifacts: standard files plus pooling weight visualizations.
- Evaluation metrics: per-target MAE/RMSE and high-error subset analysis.
- Selection criterion: validation mean MAE and interpretability of image weights.
- Expected conclusion: determines whether the model should learn which images matter.

### EXP_022_image_quality_filtering_ablation

Research question: Does filtering low-quality or irrelevant images improve multimodal regression?

- Image branch: selected image backbone and pooling.
- Image internal variant: no filtering, failed-image exclusion, blur/size filtering, optional menu/receipt heuristic filtering.
- Text branch: fixed selected text branch.
- Fusion method: fixed.
- Loss function: fixed.
- Fixed components: split identity must remain auditable.
- Variable component: image quality filtering policy.
- Training settings: same as selected architecture.
- Expected artifacts: `image_failure_manifest.csv`, `filtering_report.md`, standard files.
- Evaluation metrics: sample-wise metrics plus performance on reviews with many images.
- Selection criterion: improved validation MAE without silently changing sample counts.
- Expected conclusion: clarifies whether noisy images hurt or help.

### EXP_030_text_backbone_ablation

Research question: Do Vietnamese-specific or social-media-oriented language models improve over XLM-R?

- Image branch: selected Phase 2 branch.
- Image internal variant: fixed.
- Text branch: XLM-RoBERTa, PhoBERT, ViDeBERTa/ViBERT if available, ViSoBERT, mDeBERTa-v3, RoBERTa reference.
- Text internal variant: fixed pooling and length.
- Fusion method: concatenation + MLP.
- Loss function: fixed baseline loss.
- Fixed components: split, image branch, fusion, optimizer, metrics.
- Variable component: text backbone.
- Training settings: same maximum epochs, memory-adjusted batch size documented.
- Expected artifacts: standard files plus `tokenization_report.md`.
- Evaluation metrics: sample-wise metrics and high-error text analysis.
- Selection criterion: validation mean MAE and `overall_satisfaction` MAE.
- Expected conclusion: tests whether Vietnamese-domain pretraining improves regression.

### EXP_031_text_pooling_length_ablation

Research question: Does pooling strategy or longer text length improve text representation?

- Image branch: selected image branch.
- Text branch: selected text backbone.
- Text internal variant: first-token/CLS pooling, mean pooling, attention pooling, layer aggregation, max length 128 vs 256.
- Fusion method: fixed.
- Loss function: fixed.
- Fixed components: split, seed, optimizer.
- Variable component: text feature extraction.
- Training settings: same except sequence length and memory-adjusted batch size.
- Expected artifacts: standard files plus `text_length_coverage.json`.
- Evaluation metrics: sample-wise metrics and subset analysis by comment length.
- Selection criterion: validation mean MAE with special attention to long reviews.
- Expected conclusion: determines whether current first-token pooling underuses Vietnamese review context.

### EXP_032_text_normalization_ablation

Research question: Does controlled Vietnamese text normalization improve noisy user-generated review modeling?

- Image branch: fixed selected branch.
- Text branch: fixed selected backbone.
- Text internal variant: raw `comment_clean`, light Unicode/whitespace normalization, slang dictionary variant.
- Fusion method: fixed.
- Loss function: fixed.
- Fixed components: split, seed, metrics.
- Variable component: normalization strategy.
- Training settings: same.
- Expected artifacts: `normalization_examples.csv`, standard files.
- Evaluation metrics: sample-wise metrics and noisy-text subset metrics.
- Selection criterion: validation mean MAE and qualitative inspection.
- Expected conclusion: identifies whether extra normalization helps without destroying sentiment cues.

### EXP_040_fusion_gating_ablation

Research question: Can adaptive fusion learn when to trust text more than images?

- Image branch: selected branch.
- Text branch: selected branch.
- Fusion method: concat MLP, late weighted averaging, GMU, gated cross-modal fusion.
- Loss function: fixed baseline loss.
- Fixed components: split, encoders, optimizer.
- Variable component: fusion mechanism.
- Training settings: same fusion-stage budget; record trainable parameters.
- Expected artifacts: `fusion_gate_statistics.csv`, modality weight plots, standard files.
- Evaluation metrics: sample-wise metrics plus modality ablation on validation.
- Selection criterion: validation mean MAE and no unexplained branch collapse.
- Expected conclusion: tests whether gating is valuable for noisy image reliability.

### EXP_041_film_cross_attention_ablation

Research question: Do richer cross-modal interactions improve over gated or concatenation fusion?

- Image branch: selected branch, possibly patch features for cross-attention.
- Text branch: selected branch, token features required for cross-attention.
- Fusion method: FiLM and cross-attention.
- Loss function: fixed baseline loss.
- Fixed components: split, selected encoders, metrics.
- Variable component: interaction mechanism.
- Training settings: memory-managed batch size; document compute cost.
- Expected artifacts: attention maps if available, resource usage, standard files.
- Evaluation metrics: sample-wise metrics and hard-case analysis.
- Selection criterion: improvement large enough to justify complexity.
- Expected conclusion: decides whether fine-grained token-patch interaction is necessary.

### EXP_050_robust_regression_loss_ablation

Research question: Are robust losses better than MSE for noisy multi-target review scores?

- Image branch: selected architecture.
- Text branch: selected architecture.
- Fusion method: selected architecture.
- Loss function: MSE, MAE, Huber, SmoothL1, Log-Cosh.
- Fixed components: split, model, optimizer, seed.
- Variable component: scalar regression loss.
- Training settings: same.
- Expected artifacts: standard files plus `outlier_error_analysis.csv`.
- Evaluation metrics: MAE, RMSE, R2, tail-error percentiles.
- Selection criterion: validation mean MAE and reduced large-error tail.
- Expected conclusion: determines whether robust regression improves label-noise tolerance.

### EXP_051_multitask_loss_balancing

Research question: Does explicit task balancing improve five-target prediction?

- Image branch: selected architecture.
- Text branch: selected architecture.
- Fusion method: selected architecture.
- Loss function: equal weights, manual target weights, weighted Huber, homoscedastic uncertainty weighting.
- Fixed components: split, model, optimizer, seed.
- Variable component: target weighting.
- Training settings: monitor learned weights and per-target loss.
- Expected artifacts: `loss_weight_history.csv`, `per_target_tradeoff.md`, standard files.
- Evaluation metrics: per-target MAE/RMSE and overall mean MAE.
- Selection criterion: balanced improvement without hiding target degradation.
- Expected conclusion: tests task balancing separately from robust loss.

### EXP_060_promising_combination_validation

Research question: Do alternative full configurations outperform the greedy sequential best model?

- Image branch: selected and promising alternatives.
- Image internal variant: correct preprocessing and pooling per backbone.
- Text branch: selected and promising alternatives.
- Text internal variant: best pooling/length per candidate where feasible.
- Fusion method: selected, GMU, FiLM, or cross-attention.
- Loss function: selected robust or weighted loss.
- Fixed components: frozen split v1, sample-wise metrics, artifact template.
- Variable component: full model combination.
- Training settings: matched budget where feasible; document deviations.
- Expected artifacts: combination folders and `combination_leaderboard.csv`.
- Evaluation metrics: sample-wise metrics, resource usage, stability notes.
- Selection criterion: top validation mean MAE and feasible training cost.
- Expected conclusion: validates component synergy.

### EXP_070_final_multiseed_selection

Research question: Is the selected final model stable across random seeds?

- Image branch: top candidate(s).
- Text branch: top candidate(s).
- Fusion method: top candidate(s).
- Loss function: top candidate(s).
- Fixed components: frozen split v1 and infrastructure.
- Variable component: seed.
- Training settings: seeds 42, 123, 2026 if compute allows.
- Expected artifacts: per-seed folders and `final_leaderboard.csv`.
- Evaluation metrics: mean and standard deviation of validation metrics.
- Selection criterion: best mean validation MAE with acceptable variance.
- Expected conclusion: final model selection is robust, not a lucky run.

### EXP_071_locked_test_evaluation

Research question: What is the final honest test performance of the selected model?

- Image branch: final selected.
- Text branch: final selected.
- Fusion method: final selected.
- Loss function: final selected.
- Fixed components: frozen test split.
- Variable component: none.
- Training settings: use selected checkpoint only.
- Expected artifacts: `final_test_metrics.json`, `final_predictions.csv`, `final_model_card.md`.
- Evaluation metrics: all required sample-wise metrics.
- Selection criterion: no selection after this point.
- Expected conclusion: final thesis performance result.

### EXP_080_xai_sanity_checks

Research question: Are the XAI methods attached to the correct model tensors and targets?

- Image branch: baseline image model or multimodal baseline.
- Text branch: baseline text model or multimodal baseline.
- Fusion method: baseline concat.
- Loss function: baseline.
- Fixed components: small curated sample set.
- Variable component: XAI method.
- Training settings: no training; use saved checkpoints.
- Expected artifacts: raw heatmaps, token importance, modality contribution prototypes.
- Evaluation metrics: qualitative and shape/sanity checks.
- Selection criterion: XAI wrappers must be correct before final analysis.
- Expected conclusion: XAI tooling is usable and target-specific.

### EXP_081_final_xai_analysis

Research question: How does the final model use image, text, and fusion evidence?

- Image branch: best unimodal image and final multimodal model.
- Text branch: best unimodal text and final multimodal model.
- Fusion method: best baseline and final fusion.
- Loss function: final.
- Fixed components: selected case studies from correct, incorrect, high-error, and modality-conflict examples.
- Variable component: XAI method and target head.
- Training settings: no training; inference and explanation only.
- Expected artifacts: XAI folders with figures and raw numeric outputs.
- Evaluation metrics: explanation stability, modality contribution, failure taxonomy.
- Selection criterion: supports thesis interpretation, not model selection.
- Expected conclusion: final model is inspectable across modalities with stated limitations.

### EXP_090_thesis_ready_packaging

Research question: Can the full experiment record be audited and reproduced?

- Image branch: final and baseline records.
- Text branch: final and baseline records.
- Fusion method: final and baseline records.
- Loss function: final and baseline records.
- Fixed components: all completed artifacts.
- Variable component: reporting and packaging.
- Training settings: not applicable.
- Expected artifacts: final report, leaderboard, environment, artifact links.
- Evaluation metrics: completeness of reproduction checklist.
- Selection criterion: all required files exist or have documented Drive links.
- Expected conclusion: project is ready for lecturer review.

## 10. Image Branch Candidate Table

| Image Backbone | Family | Why Try It | Expected Strength | Expected Risk |
|---|---|---|---|---|
| ConvNeXt | Modern CNN | Current baseline and strong transfer model | Stable visual features, Grad-CAM compatible | May miss global context compared with transformers |
| Swin-B | Hierarchical vision transformer | Multi-scale visual features for scenes and food context | Good balance of local/global representation | Heavier and preprocessing must be correct |
| EfficientNet-B3 | Efficient CNN | Strong efficient CNN alternative | Good speed/performance trade-off | May underperform larger modern backbones |
| CLIP visual encoder | Image-text pretrained ViT | Visual features pretrained with language alignment | May reduce modality gap | CLIP text tokenizer is not used; visual preprocessing must be exact |
| SigLIP/SigLIP2 | Image-text pretrained ViT | Strong contrastive/image-text visual representation | Potentially strong product/review image features | May not work best with arbitrary text encoders |
| EVA-CLIP | Large image-text model | Strong visual representation if feasible | High feature quality | Heavy on Colab, implementation risk |
| ViT-L | Vision transformer | High-capacity global visual modeling | Strong if enough data/compute | Overfitting and memory cost |
| MobileViT | Lightweight hybrid | Deployment or low-resource baseline | Efficient and practical | Lower ceiling than larger backbones |

## 11. Text Branch Candidate Table

| Text Backbone | Family | Why Try It | Expected Strength | Expected Risk |
|---|---|---|---|---|
| XLM-RoBERTa | Multilingual transformer | Current multilingual baseline | Good Vietnamese coverage and robust baseline | Not specialized for Vietnamese social text |
| PhoBERT | Vietnamese RoBERTa-style model | Vietnamese-specific pretraining | Strong Vietnamese text understanding | Tokenization and preprocessing must match model expectations |
| ViDeBERTa / ViBERT | Vietnamese DeBERTa/BERT family | Vietnamese-focused encoder candidate | May improve local language semantics | Availability and exact model identity must be verified |
| ViSoBERT | Vietnamese social-media model | Designed for Vietnamese social text | Strong match to Foody-style informal reviews | May need careful tokenizer handling |
| mDeBERTa-v3 | Multilingual DeBERTa | Strong multilingual transformer alternative | Good generalization and strong encoder | Larger and may not beat Vietnamese-specific models |
| RoBERTa | English/reference baseline | Used in teammate plan with CLIP references | Useful reference for cross-modal literature | Not ideal for Vietnamese main dataset |

## 12. Fusion Candidate Table

| Fusion Method | Why Try It | Complexity | Expected Benefit | Risk |
|---|---|---:|---|---|
| Concatenation + MLP | Current baseline and easiest control | Low | Strong reference point | Cannot adaptively trust modalities |
| Late weighted averaging | Simple comparison | Low | Tests whether fusion head is necessary | Too simple for interaction |
| GMU | Learns gated modality mixture | Medium | Useful when modality reliability differs by sample | Gate collapse if not monitored |
| Gated Cross-Modal Fusion | Models reliability and interaction | Medium | Well matched to noisy images | More parameters and tuning |
| FiLM | Lets one modality modulate another | Medium | Text can condition image features or vice versa | Direction choice matters |
| Cross-Attention | Token-patch fine-grained interaction | High | Richest interaction and XAI potential | Expensive and requires feature-shape refactor |

## 13. Loss Function Candidate Table

| Loss | Why Try It | Suitable For | Risk |
|---|---|---|---|
| MSE | Current baseline | Clean regression and large-error penalty | Sensitive to outliers |
| MAE | Robust and interpretable | Noisy labels | Less smooth optimization |
| Huber | Combines MSE near zero and MAE for outliers | Noisy regression | Delta must be tuned |
| SmoothL1 | PyTorch-friendly Huber-like option | Robust regression | Similar tuning issue |
| Log-Cosh | Smooth robust regression | Label noise and outliers | Must implement and validate |
| Weighted Huber | Robustness plus target weighting | Noisy multi-task regression | Weights can bias conclusions |
| True multi-task weighted loss | Explicit task balancing | Different target difficulty | Requires defensible weights |
| Homoscedastic uncertainty weighting | Learns target uncertainty | Multi-task learning | Can learn degenerate weights without monitoring |

## 14. Recommended Image-Text Pair Hypotheses

These are research candidates, not guaranteed winners.

- `ConvNeXt + XLM-R`: current controlled baseline; strong and practical.
- `ConvNeXt + PhoBERT`: tests whether Vietnamese text improves while keeping stable visual features.
- `Swin-B + PhoBERT`: tests hierarchical transformer vision with Vietnamese text.
- `ViSoBERT + EfficientNet-B3`: adapts teammate plan to social Vietnamese text and efficient CNN features.
- `CLIP visual encoder + Vietnamese text encoder`: tests whether image-text-pretrained visual features help even with a separate Vietnamese text model.
- `SigLIP/SigLIP2 + mDeBERTa-v3`: tests image-text visual pretraining with a strong multilingual DeBERTa text branch.
- `SigLIP/EVA-CLIP + ViDeBERTa`: tests strong visual-language image features with Vietnamese text specialization.
- `MobileViT + PhoBERT`: lightweight candidate if training time or deployment matters.

## 15. Reproducibility Checklist Table

| Requirement | Status in current codebase | Required improvement | Priority |
|---|---|---|---|
| Fixed dataset version | Partially implemented via raw/processed artifacts | Create frozen split v1 manifest | Critical |
| Committed train/val/test split files | Not implemented locally | Commit lightweight splits or link exact artifact | Critical |
| Fixed random seed | Partially implemented in data split only | Add runtime seed utility | Critical |
| Python random seed | Not implemented | Set in seed utility | Critical |
| NumPy seed | Not implemented in training scripts | Set in seed utility | Critical |
| PyTorch seed | Not implemented | Set CPU and CUDA seeds | Critical |
| CUDA deterministic settings | Not implemented | Enable when feasible and document speed trade-off | High |
| DataLoader worker seed | Not implemented | Add worker init and generator | Critical |
| Saved `config.yaml` per experiment | Not implemented | Save resolved config | Critical |
| Saved `metrics.json` | Not implemented | Export sample-wise metrics | Critical |
| Saved `predictions.csv` | Not implemented | Export all validation/test predictions | Critical |
| Checkpoint or checkpoint link | Partially implemented local save path | Save full checkpoints to Drive and links to repo | Critical |
| Reported hyperparameters | Partially in notebook commands | Save in config and README | Critical |
| Reported package versions | Not pinned | Add pinned requirements and environment export | High |
| Backbone-specific preprocessing | Partially implemented, risky fallback | Build preprocessing registry | Critical |
| Sample-wise metric aggregation | Not implemented | Replace batch averaging | Critical |
| Same evaluation script | Partially implemented | Standardize evaluation with exports | Critical |
| Similar metrics on rerun | Unknown / Must be verified in codebase | Re-run smoke reproducibility check | High |
| Learning-rate scheduler | Partially implemented in `Trainer.py` | Save scheduler type/state in config and checkpoint | High |
| Early stopping | Partially implemented in `Trainer.py` | Standardize patience, monitored metric, and logs | High |
| Gradient clipping | Implemented in `Trainer.py` | Keep configurable and logged | Medium |
| Code/docs loss consistency | Partially inconsistent | Document that current code uses plain vector MSE unless changed | Critical |
| Historical notebook outputs | Partially reliable | Treat as historical only until rerun on frozen split | High |
| No silent re-splitting | Not implemented | Freeze split and manifest | Critical |
| No silent image redownload changes | Not implemented | Use image manifest and failure policy | Critical |
| Image download failures | Partially handled by filtering/black fallback | Record failures and avoid silent sample-count drift | Critical |
| Mixed precision | Not implemented | Enable when CUDA supports it | Medium |
| Resume-from-checkpoint | Not implemented | Save optimizer/scheduler/scaler state | High |
| Experiment logger | Not implemented | Add CSV/JSON logger or MLflow/W&B optional | Medium |

## 16. Artifact Management Table

| Artifact | Generated By | Stored in Drive? | Committed to GitHub? | Reason |
|---|---|---:|---:|---|
| Source code | Developer/agent | Optional backup | Yes | Required for reproduction |
| `proposal.md` | This proposal task | Optional backup | Yes | Main roadmap |
| `config.yaml` | Each experiment | Yes | Yes | Lightweight and required |
| `metrics.json` | Evaluation script | Yes | Yes | Lightweight result summary |
| `predictions.csv` | Evaluation script | Yes | Yes if not too large | Needed for exact metric audit |
| `predictions_sample.csv` | Evaluation script | Yes | Yes | Use if full predictions are too large |
| `train.log` | Training script | Yes | Yes if reasonable | Debug and audit trail |
| Training curves | Training script | Yes | Yes | Report figures |
| `error_analysis.csv` | Evaluation script | Yes | Yes | Failure analysis |
| Best checkpoint | Training script | Yes | No | Heavy artifact |
| Last checkpoint | Training script | Yes | No | Resume and audit |
| Optimizer/scheduler state | Training script | Yes | No | Resume training |
| Full image cache | Data pipeline | Yes | No | Heavy and ignored by Git |
| Image failure manifest | Data pipeline | Yes | Yes | Lightweight reproducibility record |
| Large XAI arrays | XAI scripts | Yes | No | Heavy numeric arrays |
| XAI figures | XAI scripts | Yes | Yes if selected | Thesis evidence |
| Checkpoint link | Training script or manual | Yes | Yes | Connects repo to heavy Drive artifact |
| Environment export | Setup script | Yes | Yes | Reproducibility |

## 17. Google Colab, Drive, and GitHub Strategy

Principle:

```text
Google Drive = heavy artifact storage
GitHub repo = code + configs + lightweight results + documentation
```

Google Drive should store:

- large checkpoints
- full image cache
- large XAI arrays
- large intermediate tensors
- large logs if needed
- full artifact backup

GitHub should store:

- source code
- configs
- `proposal.md`
- README files
- `metrics.json`
- `predictions.csv` if not too large
- prediction samples if full file is too large
- training logs if reasonable
- figures
- leaderboard
- experiment summaries
- checkpoint links
- environment files

Each Colab experiment should save to Drive first:

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

## 18. Recommended Project Structure

```text
project/
|
|-- data/
|   |-- raw/
|   |-- processed/
|   |-- splits/
|   |   |-- train.csv
|   |   |-- val.csv
|   |   `-- test.csv
|   `-- README.md
|
|-- src/
|   |-- datasets/
|   |-- models/
|   |-- fusion/
|   |-- losses/
|   |-- training/
|   |-- evaluation/
|   |-- metrics/
|   |-- xai/
|   `-- utils/
|
|-- configs/
|-- notebooks/
|-- experiments/
|-- reports/
|-- scripts/
|-- requirements.txt
|-- environment.yml
`-- README.md
```

Shared `.py` code should contain reusable logic:

- dataset loading
- image preprocessing
- text tokenization
- model definitions
- fusion modules
- loss functions
- training loop
- evaluation loop
- metrics computation
- seed utilities
- checkpoint utilities
- experiment logger
- XAI utilities

Notebooks should only:

- mount Google Drive
- set up the environment
- copy dataset/artifacts
- run training commands
- display metrics
- draw charts
- inspect predictions
- visualize XAI results

The notebook must not contain the only copy of core training logic.

## 19. Experiment Folder Template

```text
experiments/
`-- EXP_XXX_short_name/
    |-- README.md
    |-- config.yaml
    |-- metrics.json
    |-- predictions.csv
    |-- error_analysis.csv
    |-- train.log
    |-- figures/
    |   |-- loss_curve.png
    |   |-- mae_by_target.png
    |   `-- prediction_vs_groundtruth.png
    |-- xai/
    |   |-- gradcam/
    |   |-- attention/
    |   |-- shap/
    |   `-- lime/
    `-- checkpoint_link.md
```

File purposes:

- `README.md`: human-readable summary, result interpretation, reproduction command, and limitations.
- `config.yaml`: full resolved experiment configuration.
- `metrics.json`: exact sample-wise validation/test metrics.
- `predictions.csv`: per-sample predictions and absolute errors.
- `error_analysis.csv`: highest-error samples and grouped diagnostics.
- `train.log`: raw training log.
- `figures/loss_curve.png`: training and validation curves.
- `figures/mae_by_target.png`: per-target comparison.
- `figures/prediction_vs_groundtruth.png`: calibration/scatter plot.
- `xai/`: method-specific explanation outputs.
- `checkpoint_link.md`: Drive links for heavy model artifacts.

## 20. Reusable `config.yaml` Template

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

Note: the current `Config.py` default `max_length` is 256. The template starts at 128 because it is a conservative baseline. Phase 3 should explicitly ablate 128 vs 256.

## 21. Experiment README Template

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

## 22. Final Reporting Requirements

The final report to a lecturer should include:

- project goal and dataset description
- current codebase status and limitations
- frozen split version and image manifest
- baseline table: text-only, image-only, multimodal concat
- sequential ablation tables for image, text, fusion, and loss
- promising combination validation table
- final multi-seed selection table
- locked test performance table
- error taxonomy
- XAI case studies
- reproducibility checklist
- Drive links for heavy artifacts
- exact commands or configs used

Claims must be phrased defensibly. For example:

- Strong: "Under frozen split v1 and sample-wise evaluation, model A achieved lower validation MAE than model B."
- Too strong: "Model A is universally better for Vietnamese multimodal regression."

The final conclusion should distinguish prediction performance from explanation quality. XAI should be presented as architecture-aligned evidence for inspection and debugging, not proof of causality.
