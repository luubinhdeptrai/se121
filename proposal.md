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

For Phases 2 to 7, the roadmap table rows above are study groups. The concrete trainable run IDs inside each group are enumerated explicitly in Sections 8 and 9 so that each launched experiment corresponds to one exact configuration.

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

- `EXP_010_text_only_xlmr_mse`
- `EXP_011_image_only_convnext_meanpool_mse`
- `EXP_012_multimodal_convnext_xlmr_concat_mse`

Required baseline definitions:

- Baseline 0.1: `EXP_010_text_only_xlmr_mse`. Use `xlm-roberta-base`, first-token pooling, max length 256, and MSE.
- Baseline 0.2: `EXP_011_image_only_convnext_meanpool_mse`. Use `convnext_base_in22k`, pooled ConvNeXt features, masked multi-image mean pooling, and MSE.
- Baseline 0.3: `EXP_012_multimodal_convnext_xlmr_concat_mse`. Use ConvNeXt + XLM-R, concatenation + MLP fusion, and MSE.

Every later experiment must be compared against these baselines, not only against the immediately previous ablation.

Implementation notes:

- Text-only baseline should use `xlm-roberta-base` as the official Phase 1 anchor so that Vietnamese-specific backbones remain cleanly isolated for Phase 3.
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

- Image backbone runs:
  - `EXP_020A_convnext_xlmr_concat_mse` (reference; may reuse `EXP_012`)
  - `EXP_020B_swinb_xlmr_concat_mse`
  - `EXP_020C_siglip_xlmr_concat_mse`
  - `EXP_020D_efficientnetb3_xlmr_concat_mse`
- Multi-image pooling runs:
  - `EXP_021A_bestimage_meanpool_xlmr_concat_mse` (reference; may reuse the winning `EXP_020*` run)
  - `EXP_021B_bestimage_attentionpool_xlmr_concat_mse`
- Image quality filtering runs:
  - `EXP_022A_bestimage_bestpool_xlmr_concat_mse_nofilter` (reference; may reuse the winning `EXP_021*` run)
  - `EXP_022B_bestimage_bestpool_xlmr_concat_mse_decodefilter`
  - `EXP_022C_bestimage_bestpool_xlmr_concat_mse_decode_sizefilter`

Implementation notes:

- Use `xlm-roberta-base` as the fixed Phase 2 text branch so that image changes remain isolated. Vietnamese-specific text encoders are deferred to Phase 3 by design.
- Compare the four highest-value image backbones first: ConvNeXt, Swin-B, SigLIP, and EfficientNet-B3.
- Use Global Average Pooling or timm-provided pooled features for CNN-like models.
- Use CLS/patch-token strategies only for ViT-like models that actually expose such tokens.
- Test multi-image mean pooling against attention pooling across images.
- Image quality filtering should be recorded with a manifest, not silent deletion.
- CLIP, EVA-CLIP, ViT-L, and MobileViT remain optional extensions under the same template if the core four runs do not separate clearly and compute still permits expansion.

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

- Text backbone runs:
  - `EXP_030A_bestimage_bestpool_xlmr_concat_mse` (reference; may reuse the winning `EXP_022*` run)
  - `EXP_030B_bestimage_bestpool_phobert_concat_mse`
  - `EXP_030C_bestimage_bestpool_vibert_concat_mse`
  - `EXP_030D_bestimage_bestpool_visobert_concat_mse`
  - `EXP_030E_bestimage_bestpool_mdebertav3_concat_mse`
- Text pooling and length runs:
  - `EXP_031A_bestimage_besttext_firsttoken_len256_concat_mse`
  - `EXP_031B_bestimage_besttext_meanpool_len256_concat_mse`
  - `EXP_031C_bestimage_besttext_attentionpool_len256_concat_mse`
  - `EXP_031D_bestimage_besttext_bestpool_len128_concat_mse`
- Text normalization runs:
  - `EXP_032A_bestimage_besttext_bestpool_rawtext_concat_mse` (reference; may reuse the winning `EXP_031*` run)
  - `EXP_032B_bestimage_besttext_bestpool_unicode_whitespace_concat_mse`
  - `EXP_032C_bestimage_besttext_bestpool_light_slangmap_concat_mse`

Implementation notes:

- Compare XLM-RoBERTa, PhoBERT, ViDeBERTa/ViBERT if available, ViSoBERT, mDeBERTa-v3, and RoBERTa only as a reference baseline.
- Test first-token/CLS pooling, mean pooling over non-padding tokens, attention pooling, and last-layer or last-four-layer aggregation.
- Test max text length 128 vs 256. Current `Config.py` default is 256, while older docs and some templates mention 128.
- Text normalization must be conservative. Do not remove sentiment-bearing slang, negation, or emojis without an ablation.
- The minimum concrete Phase 3 set uses Vietnamese-priority backbones first. RoBERTa remains a literature reference and does not need to be in the core trainable set unless a lecturer specifically requests it.

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

- Gating runs:
  - `EXP_040A_bestimage_besttext_besttextpool_concat_mse` (reference; may reuse the winning `EXP_032*` run)
  - `EXP_040B_bestimage_besttext_besttextpool_gmu_mse`
  - `EXP_040C_bestimage_besttext_besttextpool_gatedcrossmodal_mse`
- Interaction runs:
  - `EXP_041A_bestimage_besttext_besttextpool_film_mse`
  - `EXP_041B_bestimage_besttext_besttextpool_crossattention_mse`

Implementation notes:

- Concatenation + MLP remains the baseline.
- Late weighted averaging is a simple sanity comparison, but it is not part of the minimum trainable set.
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

- Robust-loss runs:
  - `EXP_050A_bestimage_besttext_bestfusion_mse` (reference; may reuse the winning MSE-based Phase 4 run)
  - `EXP_050B_bestimage_besttext_bestfusion_huber`
  - `EXP_050C_bestimage_besttext_bestfusion_smoothl1`
  - `EXP_050D_bestimage_besttext_bestfusion_logcosh`
- Multitask-balancing runs:
  - `EXP_051A_bestimage_besttext_bestfusion_bestloss_equalweights` (reference; may reuse the winning `EXP_050*` run)
  - `EXP_051B_bestimage_besttext_bestfusion_bestloss_manualtaskweights`
  - `EXP_051C_bestimage_besttext_bestfusion_huber_manualtaskweights`
  - `EXP_051D_bestimage_besttext_bestfusion_uncertaintyweighted`

Implementation notes:

- MSE is the baseline.
- MAE is robust but may optimize less smoothly.
- Huber, SmoothL1, and Log-Cosh are appropriate for noisy regression labels and outliers.
- Weighted Huber combines robustness with task weighting.
- True multi-task weighted loss should expose weights for each target.
- Homoscedastic uncertainty-weighted multi-task loss is feasible if implemented carefully and monitored for degenerate weights.
- Huber itself does not solve multi-task balancing. Task weights solve balancing.
- Focal loss and weighted cross-entropy are not suitable for the main task unless an auxiliary classification head is added.
- If `EXP_050` selects Huber as the best scalar loss, `EXP_051B` becomes the weighted-Huber run and `EXP_051C` can be marked skipped as redundant.

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

- `EXP_060A_bestsequential_full_configuration`
- `EXP_060B_convnext_phobert_film_huber`
- `EXP_060C_swinb_vibert_gmu_huber`
- `EXP_060D_efficientnetb3_visobert_film_huber`
- `EXP_060E_siglip_mdebertav3_gmu_weightedhuber`

Promising combinations to test:

- `EXP_060A_bestsequential_full_configuration`: the exact winner from Phases 2 to 5.
- `EXP_012_multimodal_convnext_xlmr_concat_mse` as the official baseline anchor in the comparison table; it does not need to be retrained.
- `EXP_060B_convnext_phobert_film_huber`.
- `EXP_060C_swinb_vibert_gmu_huber`.
- `EXP_060D_efficientnetb3_visobert_film_huber`.
- `EXP_060E_siglip_mdebertav3_gmu_weightedhuber`.
- The original CLIP, EVA-CLIP, and MobileViT combinations remain optional extension runs if the core validation set leaves meaningful uncertainty and Colab budget permits.

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

- `EXP_070A_candidate1_seed42`
- `EXP_070B_candidate1_seed123`
- `EXP_070C_candidate1_seed2026`
- `EXP_070D_candidate2_seed42`
- `EXP_070E_candidate2_seed123`
- `EXP_070F_candidate2_seed2026`
- `EXP_071_locked_test_evaluation`

Implementation notes:

- Run seeds 42, 123, and 2026 for the top two Phase 6 candidates. Extend the same template to a third candidate only if the Phase 6 gap between second and third place is small enough to keep the decision ambiguous.
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

Whenever a trainable ID contains `bestimage`, `bestpool`, `besttext`, `besttextpool`, `bestfusion`, or `bestloss`, the placeholder must be replaced in `config.yaml` and the experiment `README.md` with the exact winning upstream experiment ID before launch. If a reference run is identical to an already completed earlier run, the later experiment folder may point to the earlier artifacts instead of retraining the same configuration.

### EXP_010_text_only_baseline

Research question: How strong is review text alone for predicting the five targets?

- Trainable configuration ID: `EXP_010_text_only_xlmr_mse`
- Image branch: disabled.
- Image internal variant: not applicable.
- Text branch: `xlm-roberta-base`.
- Text internal variant: first-token pooling, max length 256.
- Fusion method: none.
- Loss function: MSE.
- Fixed components: frozen split v1, AdamW, scheduler, early stopping, seed 42.
- Variable component: modality is text only.
- Training settings: 20 epochs maximum, patience 3-5, batch size based on GPU memory, gradient clipping 1.0.
- Motivation: establish the current-code multilingual text anchor before any Vietnamese-specific backbone changes.
- Expected artifacts: `config.yaml`, `metrics.json`, `predictions.csv`, `train.log`, loss curve, checkpoint link.
- Evaluation metrics: MAE, RMSE, R2, per-target metrics.
- Selection criterion: establishes the official text-only reference for all later comparisons.
- Expected conclusion: text should be a strong unimodal signal for Vietnamese reviews.

### EXP_011_image_only_baseline

Research question: How much predictive signal exists in review images alone?

- Trainable configuration ID: `EXP_011_image_only_convnext_meanpool_mse`
- Image branch: `convnext_base_in22k` or its resolved timm equivalent recorded in `config.yaml`.
- Image internal variant: timm pooled ConvNeXt features plus masked multi-image mean pooling over up to 4 images.
- Text branch: disabled.
- Text internal variant: not applicable.
- Fusion method: none.
- Loss function: MSE.
- Fixed components: frozen split v1, AdamW, scheduler, early stopping, seed 42.
- Variable component: modality is image only.
- Training settings: use smaller batch size if GPU memory requires it.
- Motivation: quantify the standalone value of the noisy image modality using the most stable current visual baseline.
- Expected artifacts: same standard experiment files plus image failure summary.
- Evaluation metrics: sample-wise MAE, RMSE, R2, per-target metrics.
- Selection criterion: compare against text-only and multimodal baseline.
- Expected conclusion: image-only may be weaker because images are heterogeneous, but it quantifies visual contribution.

### EXP_012_multimodal_concat_baseline

Research question: Does simple multimodal fusion improve over text-only and image-only baselines?

- Trainable configuration ID: `EXP_012_multimodal_convnext_xlmr_concat_mse`
- Image branch: `convnext_base_in22k` or its resolved timm equivalent recorded in `config.yaml`.
- Image internal variant: pooled ConvNeXt features plus masked multi-image mean pooling.
- Text branch: `xlm-roberta-base`.
- Text internal variant: first-token pooling, max length 256.
- Fusion method: concatenation + MLP.
- Loss function: MSE.
- Fixed components: frozen split v1, seed 42, same training infrastructure.
- Variable component: fusion of both modalities.
- Training settings: staged or joint training must be recorded exactly.
- Motivation: create the official multimodal anchor that later ablations will inherit and compare against.
- Expected artifacts: standard experiment files.
- Evaluation metrics: sample-wise MAE, RMSE, R2, per-target metrics.
- Selection criterion: should beat or complement unimodal baselines.
- Expected conclusion: concatenation becomes the main reference for later fusion experiments.

### EXP_020_image_backbone_ablation

Research question: Which image encoder gives the best visual representation under fixed text, fusion, and loss settings?

- Shared fixed components for all `EXP_020*` runs:
  - Text branch: `xlm-roberta-base`, first-token pooling, max length 256, inherited from `EXP_012`.
  - Fusion method: concatenation + MLP.
  - Loss function: MSE.
  - Frozen split v1, seed 42, same optimizer/scheduler/early stopping, sample-wise metrics, and backbone-specific preprocessing registry.
- Variable component: image backbone only.
- Concrete trainable experiments:

| Trainable ID | Image branch | Image internal variant | Motivation | Expected claim |
|---|---|---|---|---|
| `EXP_020A_convnext_xlmr_concat_mse` | `convnext_base_in22k` | pooled CNN features + masked multi-image mean pooling | reference run; same family as `EXP_012` | stable visual anchor for later comparison |
| `EXP_020B_swinb_xlmr_concat_mse` | `swin_base_patch4_window7_224` | pooled transformer features + masked multi-image mean pooling | test hierarchical transformer features for atmosphere and scene context | Swin-B may help when environment cues matter |
| `EXP_020C_siglip_xlmr_concat_mse` | `vit_base_patch16_siglip_224` | pooled ViT features + masked multi-image mean pooling | test image-text-pretrained visual features under the same text branch | SigLIP may help when semantic visual alignment matters |
| `EXP_020D_efficientnetb3_xlmr_concat_mse` | `efficientnet_b3` | pooled CNN features + masked multi-image mean pooling | add a cheaper strong CNN alternative for Colab | EfficientNet-B3 may approach stronger backbones at lower cost |

- Training settings: same epochs and early stopping across all runs; adjust batch size only if documented in `config.yaml`.
- Expected artifacts: standard files plus `image_preprocessing_report.md`.
- Evaluation metrics: sample-wise MAE, RMSE, R2, resource usage.
- Selection criterion: validation mean MAE with resource-aware tie breaker.
- Expected conclusion: identifies the most reliable image backbone for noisy review photos.

### EXP_021_multi_image_pooling_ablation

Research question: Does attention pooling across review images outperform simple mean pooling?

- Shared fixed components for all `EXP_021*` runs:
  - Image backbone: the winning `EXP_020*` backbone.
  - Text branch: `xlm-roberta-base`, first-token pooling, max length 256.
  - Fusion method: concatenation + MLP.
  - Loss function: MSE.
  - Frozen split v1, seed 42, same optimizer/scheduler/metrics.
- Variable component: review-level image aggregation only.
- Concrete trainable experiments:

| Trainable ID | Image internal variant | Motivation | Expected claim |
|---|---|---|---|
| `EXP_021A_bestimage_meanpool_xlmr_concat_mse` | masked mean pooling over valid image embeddings | reference policy already closest to current code | establishes whether simple averaging is sufficient |
| `EXP_021B_bestimage_attentionpool_xlmr_concat_mse` | learned attention pooling over valid image embeddings using the `num_images` mask | one review may contain both useful and irrelevant photos | attention pooling is adopted only if it learns better review-level image selection |

- Training settings: identical where possible. If `EXP_021A` is identical to the winning `EXP_020*` run, reuse earlier artifacts rather than retraining.
- Expected artifacts: standard files plus pooling weight visualizations.
- Evaluation metrics: per-target MAE/RMSE and high-error subset analysis.
- Selection criterion: validation mean MAE and interpretability of image weights.
- Expected conclusion: determines whether the model should learn which images matter.

### EXP_022_image_quality_filtering_ablation

Research question: Does filtering low-quality or irrelevant images improve multimodal regression?

- Shared fixed components for all `EXP_022*` runs:
  - Image backbone: winning `EXP_020*` backbone.
  - Image pooling: winning `EXP_021*` strategy.
  - Text branch: `xlm-roberta-base`, first-token pooling, max length 256.
  - Fusion method: concatenation + MLP.
  - Loss function: MSE.
  - Split identity and review row count must remain unchanged.
- Variable component: image filtering policy only.
- Concrete trainable experiments:

| Trainable ID | Filtering policy | Motivation | Expected claim |
|---|---|---|---|
| `EXP_022A_bestimage_bestpool_xlmr_concat_mse_nofilter` | no extra image filtering beyond the frozen dataset and existing decode behavior | reference policy | establishes whether filtering is needed at all |
| `EXP_022B_bestimage_bestpool_xlmr_concat_mse_decodefilter` | mask only missing, undecodable, or zero-byte images while keeping the review row | remove obviously broken visual evidence without changing the split | broken-image masking may reduce avoidable noise |
| `EXP_022C_bestimage_bestpool_xlmr_concat_mse_decode_sizefilter` | apply `EXP_022B` plus mask images with very small resolution, for example short side below 96 pixels | test a conservative quality threshold without semantic heuristics | conservative quality filtering may help if tiny images are mostly noise |

- Training settings: same as the selected architecture. Every per-image removal must be logged to `image_failure_manifest.csv`.
- Expected artifacts: `image_failure_manifest.csv`, `filtering_report.md`, standard files.
- Evaluation metrics: sample-wise metrics plus performance on reviews with many images.
- Selection criterion: improved validation MAE without silently changing sample counts.
- Expected conclusion: clarifies whether noisy images hurt or help.

### EXP_030_text_backbone_ablation

Research question: Do Vietnamese-specific or social-media-oriented language models improve over XLM-R?

- Shared fixed components for all `EXP_030*` runs:
  - Image branch: winning `EXP_020*` backbone, winning `EXP_021*` pooling, and winning `EXP_022*` filtering policy.
  - Fusion method: concatenation + MLP.
  - Loss function: MSE.
  - Text internal variant: first-token pooling, max length 256.
  - Frozen split v1, seed 42, same optimizer/scheduler/metrics.
- Variable component: text backbone only.
- Concrete trainable experiments:

| Trainable ID | Text branch | Motivation | Expected claim |
|---|---|---|---|
| `EXP_030A_bestimage_bestpool_xlmr_concat_mse` | `xlm-roberta-base` | reference multilingual baseline; may reuse the winning `EXP_022*` run if identical | preserves continuity with Phase 2 |
| `EXP_030B_bestimage_bestpool_phobert_concat_mse` | PhoBERT, exact checkpoint to be verified in codebase/environment and recorded in `config.yaml` | Vietnamese-specific pretraining should better match restaurant-review language | PhoBERT may improve local lexical and sentiment cues |
| `EXP_030C_bestimage_bestpool_vibert_concat_mse` | `FPTAI/vibert-base-cased` | existing notebook evidence already uses this Vietnamese checkpoint | ViBERT may offer a practical Vietnamese-focused alternative |
| `EXP_030D_bestimage_bestpool_visobert_concat_mse` | `uitnlp/visobert` | social-media pretraining matches informal Foody-style text | ViSoBERT may help on slang and casual review phrasing |
| `EXP_030E_bestimage_bestpool_mdebertav3_concat_mse` | `microsoft/mdeberta-v3-base` | strong multilingual DeBERTa-family reference already seen in historical runs | mDeBERTa-v3 may improve robustness even without Vietnamese-only pretraining |

- Training settings: same maximum epochs, memory-adjusted batch size documented.
- Expected artifacts: standard files plus `tokenization_report.md`.
- Evaluation metrics: sample-wise metrics and high-error text analysis.
- Selection criterion: validation mean MAE and `overall_satisfaction` MAE.
- Expected conclusion: tests whether Vietnamese-domain pretraining improves regression.

### EXP_031_text_pooling_length_ablation

Research question: Does pooling strategy or longer text length improve text representation?

- Shared fixed components for all `EXP_031*` runs:
  - Image branch: winning Phase 2 image branch.
  - Text backbone: winning `EXP_030*` backbone.
  - Fusion method: concatenation + MLP.
  - Loss function: MSE.
  - Frozen split v1, seed 42, same optimizer/scheduler/metrics.
- Variable component: text pooling strategy and sequence length only.
- Concrete trainable experiments:

| Trainable ID | Text internal variant | Motivation | Expected claim |
|---|---|---|---|
| `EXP_031A_bestimage_besttext_firsttoken_len256_concat_mse` | first-token pooling, max length 256 | reference configuration closest to the current code path | establishes the controlled text-feature baseline |
| `EXP_031B_bestimage_besttext_meanpool_len256_concat_mse` | masked mean pooling over non-padding tokens, max length 256 | test whether whole-sequence averaging captures review context better | mean pooling may reduce overreliance on the first token |
| `EXP_031C_bestimage_besttext_attentionpool_len256_concat_mse` | learned attention pooling over non-padding tokens, max length 256 | let the model learn which tokens matter most | attention pooling is useful only if it improves enough to justify extra complexity |
| `EXP_031D_bestimage_besttext_bestpool_len128_concat_mse` | winning pooling from `EXP_031A` to `EXP_031C`, max length 128 | explicitly test whether shorter input can match performance at lower cost | 128 tokens may be enough, or 256 may be justified for long reviews |

- Training settings: run `EXP_031A` to `EXP_031C` first, then launch `EXP_031D` using the winning pooling from the first three runs. Layer aggregation remains optional future work if pooling results remain ambiguous.
- Expected artifacts: standard files plus `text_length_coverage.json`.
- Evaluation metrics: sample-wise metrics and subset analysis by comment length.
- Selection criterion: validation mean MAE with special attention to long reviews.
- Expected conclusion: determines whether current first-token pooling underuses Vietnamese review context.

### EXP_032_text_normalization_ablation

Research question: Does controlled Vietnamese text normalization improve noisy user-generated review modeling?

- Shared fixed components for all `EXP_032*` runs:
  - Image branch: winning Phase 2 image branch.
  - Text backbone and pooling/length: winning `EXP_030*` and `EXP_031*` settings.
  - Fusion method: concatenation + MLP.
  - Loss function: MSE.
  - Frozen split v1, seed 42, same optimizer/scheduler/metrics.
- Variable component: normalization strategy only.
- Concrete trainable experiments:

| Trainable ID | Normalization strategy | Motivation | Expected claim |
|---|---|---|---|
| `EXP_032A_bestimage_besttext_bestpool_rawtext_concat_mse` | use current `comment_clean` exactly as stored | reference policy; may reuse the winning `EXP_031*` run if identical | establishes whether extra normalization is needed |
| `EXP_032B_bestimage_besttext_bestpool_unicode_whitespace_concat_mse` | apply Unicode normalization and whitespace collapsing only | test a minimal cleanup that should not alter sentiment content | minimal normalization may remove harmless noise |
| `EXP_032C_bestimage_besttext_bestpool_light_slangmap_concat_mse` | apply `EXP_032B` plus a conservative fixed slang map; preserve negation, intensifiers, and sentiment-bearing tokens | test whether a small domain lexicon helps with Foody-style text | light slang handling may improve robustness without over-cleaning |

- Training settings: same across all runs. The slang map must be versioned and saved with the experiment artifacts.
- Expected artifacts: `normalization_examples.csv`, standard files.
- Evaluation metrics: sample-wise metrics and noisy-text subset metrics.
- Selection criterion: validation mean MAE and qualitative inspection.
- Expected conclusion: identifies whether extra normalization helps without destroying sentiment cues.

### EXP_040_fusion_gating_ablation

Research question: Can adaptive fusion learn when to trust text more than images?

- Shared fixed components for all `EXP_040*` runs:
  - Image branch: winning Phase 2 image branch.
  - Text branch: winning Phase 3 text backbone, pooling/length, and normalization.
  - Loss function: MSE.
  - Frozen split v1, seed 42, same optimizer/scheduler/metrics.
- Variable component: gating-oriented fusion mechanism only.
- Concrete trainable experiments:

| Trainable ID | Fusion method | Motivation | Expected claim |
|---|---|---|---|
| `EXP_040A_bestimage_besttext_besttextpool_concat_mse` | concatenation + MLP | reference fusion; may reuse the winning `EXP_032*` run if identical | preserves the simple controlled baseline |
| `EXP_040B_bestimage_besttext_besttextpool_gmu_mse` | GMU | allow sample-level gating when image reliability varies | GMU may improve robustness to noisy or irrelevant photos |
| `EXP_040C_bestimage_besttext_besttextpool_gatedcrossmodal_mse` | gated cross-modal fusion | test a stronger reliability-aware interaction than plain concat | gated cross-modal fusion may outperform concat when modalities disagree |

- Training settings: same fusion-stage budget across runs; record trainable parameter count and gate statistics.
- Expected artifacts: `fusion_gate_statistics.csv`, modality weight plots, standard files.
- Evaluation metrics: sample-wise metrics plus modality ablation on validation.
- Selection criterion: validation mean MAE and no unexplained branch collapse.
- Expected conclusion: tests whether gating is valuable for noisy image reliability.

### EXP_041_film_cross_attention_ablation

Research question: Do richer cross-modal interactions improve over gated or concatenation fusion?

- Shared fixed components for all `EXP_041*` runs:
  - Image branch: same winning Phase 2 image branch used in `EXP_040*`.
  - Text branch: same winning Phase 3 text branch used in `EXP_040*`.
  - Loss function: MSE.
  - Frozen split v1, seed 42, same optimizer/scheduler/metrics.
- Variable component: richer interaction mechanism only.
- Concrete trainable experiments:

| Trainable ID | Fusion method | Motivation | Expected claim |
|---|---|---|---|
| `EXP_041A_bestimage_besttext_besttextpool_film_mse` | FiLM | let one modality modulate the other with moderate complexity | FiLM may capture useful conditioning without full cross-attention cost |
| `EXP_041B_bestimage_besttext_besttextpool_crossattention_mse` | cross-attention with token-level text features and patch-level image features | test the strongest fine-grained interaction idea from the literature | cross-attention is justified only if its gain clearly exceeds its implementation and memory cost |

- Training settings: memory-managed batch size; document compute cost. `EXP_041B` is high risk and should not proceed until token and patch features are verified in code.
- Expected artifacts: attention maps if available, resource usage, standard files.
- Evaluation metrics: sample-wise metrics and hard-case analysis.
- Selection criterion: improvement large enough to justify complexity.
- Expected conclusion: decides whether fine-grained token-patch interaction is necessary.

### EXP_050_robust_regression_loss_ablation

Research question: Are robust losses better than MSE for noisy multi-target review scores?

- Shared fixed components for all `EXP_050*` runs:
  - Image branch: winning Phase 2 image branch.
  - Text branch: winning Phase 3 text branch.
  - Fusion method: winning Phase 4 fusion method.
  - Frozen split v1, seed 42, same optimizer/scheduler/metrics.
- Variable component: scalar regression loss only.
- Concrete trainable experiments:

| Trainable ID | Loss function | Motivation | Expected claim |
|---|---|---|---|
| `EXP_050A_bestimage_besttext_bestfusion_mse` | MSE | reference loss; may reuse the winning MSE-based Phase 4 run | preserves the baseline objective |
| `EXP_050B_bestimage_besttext_bestfusion_huber` | Huber | robust alternative expected to handle outliers better | Huber may reduce large errors without losing smooth optimization |
| `EXP_050C_bestimage_besttext_bestfusion_smoothl1` | SmoothL1 | PyTorch-native Huber-like alternative | SmoothL1 may match Huber with simpler implementation |
| `EXP_050D_bestimage_besttext_bestfusion_logcosh` | Log-Cosh | smooth robust loss retained from the original proposal | Log-Cosh may offer another stable robust-loss option |

- Training settings: same across all runs. MAE is left out of the minimum trainable set because it changes optimization dynamics while usually overlapping conceptually with smoother robust losses.
- Expected artifacts: standard files plus `outlier_error_analysis.csv`.
- Evaluation metrics: MAE, RMSE, R2, tail-error percentiles.
- Selection criterion: validation mean MAE and reduced large-error tail.
- Expected conclusion: determines whether robust regression improves label-noise tolerance.

### EXP_051_multitask_loss_balancing

Research question: Does explicit task balancing improve five-target prediction?

- Shared fixed components for all `EXP_051*` runs:
  - Image branch: winning Phase 2 image branch.
  - Text branch: winning Phase 3 text branch.
  - Fusion method: winning Phase 4 fusion method.
  - Frozen split v1, seed 42, same optimizer/scheduler/metrics.
  - Manual target weights, when used, must be fixed once from the inverse per-target MAE of the Phase 5 reference run and normalized to mean 1.0.
- Variable component: target weighting strategy.
- Concrete trainable experiments:

| Trainable ID | Loss and weighting strategy | Motivation | Expected claim |
|---|---|---|---|
| `EXP_051A_bestimage_besttext_bestfusion_bestloss_equalweights` | winning `EXP_050*` scalar loss with equal target weights | reference policy; may reuse the winning `EXP_050*` run | isolates weighting from scalar-loss choice |
| `EXP_051B_bestimage_besttext_bestfusion_bestloss_manualtaskweights` | winning `EXP_050*` scalar loss plus fixed manual target weights | test whether explicit balancing helps without changing the scalar loss family | task weighting may improve weak targets without architectural changes |
| `EXP_051C_bestimage_besttext_bestfusion_huber_manualtaskweights` | Huber plus the same fixed manual target weights | explicit weighted-Huber candidate from the original proposal | weighted Huber may combine robustness and balancing effectively |
| `EXP_051D_bestimage_besttext_bestfusion_uncertaintyweighted` | homoscedastic uncertainty-weighted multitask loss | test learnable target balancing | uncertainty weighting may help if it stays well behaved |

- Training settings: monitor learned weights and per-target loss. If `EXP_050` already selects Huber as the best scalar loss, `EXP_051B` becomes the weighted-Huber run and `EXP_051C` can be marked skipped as redundant.
- Expected artifacts: `loss_weight_history.csv`, `per_target_tradeoff.md`, standard files.
- Evaluation metrics: per-target MAE/RMSE and overall mean MAE.
- Selection criterion: balanced improvement without hiding target degradation.
- Expected conclusion: tests task balancing separately from robust loss.

### EXP_060_promising_combination_validation

Research question: Do alternative full configurations outperform the greedy sequential best model?

- Shared fixed components for all `EXP_060*` runs:
  - Frozen split v1, sample-wise metrics, artifact template, and standardized training infrastructure.
  - Use the best pooling and normalization setting already selected for each text backbone, and the best image pooling/filtering setting already selected for each image backbone when that backbone appeared in Phase 2.
- Variable component: full architecture combination.
- Concrete trainable experiments:

| Trainable ID | Image branch | Text branch | Fusion | Loss | Motivation | Expected claim |
|---|---|---|---|---|---|---|
| `EXP_060A_bestsequential_full_configuration` | exact Phase 2 winner | exact Phase 3 winner | exact Phase 4 winner | exact Phase 5 winner | validate the greedy sequential winner as a complete system | confirms whether the sequential path already found the strongest practical model |
| `EXP_060B_convnext_phobert_film_huber` | `convnext_base_in22k` | PhoBERT, exact checkpoint verified in `config.yaml` | FiLM | Huber | keep a stable CNN image branch while pairing it with Vietnamese-specific text and moderate cross-modal conditioning | this combination may be stronger than the official baseline without large complexity jump |
| `EXP_060C_swinb_vibert_gmu_huber` | `swin_base_patch4_window7_224` | `FPTAI/vibert-base-cased` | GMU | Huber | combine hierarchical scene-aware image features with a Vietnamese-focused text encoder and adaptive gating | this pair may be especially useful when image reliability varies across reviews |
| `EXP_060D_efficientnetb3_visobert_film_huber` | `efficientnet_b3` | `uitnlp/visobert` | FiLM | Huber | preserve the original proposal's efficient social-text candidate | this run tests whether domain-matched text can offset a cheaper image encoder |
| `EXP_060E_siglip_mdebertav3_gmu_weightedhuber` | `vit_base_patch16_siglip_224` | `microsoft/mdeberta-v3-base` | GMU | Weighted Huber | convert the historically attempted SigLIP plus mDeBERTa direction into a stronger final-form candidate | this run checks whether image-text-pretrained vision plus robust gated fusion shows better synergy |

- Training settings: matched budget where feasible; document any deviations. `EXP_012_multimodal_convnext_xlmr_concat_mse` must still appear in the final comparison table as the official baseline anchor even though it is not retrained here.
- Expected artifacts: combination folders and `combination_leaderboard.csv`.
- Evaluation metrics: sample-wise metrics, resource usage, stability notes.
- Selection criterion: top validation mean MAE and feasible training cost.
- Expected conclusion: validates component synergy.

### EXP_070_final_multiseed_selection

Research question: Is the selected final model stable across random seeds?

- Shared fixed components for all `EXP_070*` runs:
  - Frozen split v1 and finalized infrastructure.
  - Candidate 1: best validation model from `EXP_060*`.
  - Candidate 2: strongest non-identical `EXP_060*` alternative that differs in backbone or fusion, not just a small loss variant.
- Variable component: random seed.
- Concrete trainable experiments:

| Trainable ID | Candidate | Seed | Motivation | Expected claim |
|---|---|---:|---|---|
| `EXP_070A_candidate1_seed42` | Candidate 1 | 42 | reference rerun | establishes the first point for stability |
| `EXP_070B_candidate1_seed123` | Candidate 1 | 123 | second seed for candidate 1 | checks whether performance holds beyond the original seed |
| `EXP_070C_candidate1_seed2026` | Candidate 1 | 2026 | third seed for candidate 1 | completes the minimal variance estimate |
| `EXP_070D_candidate2_seed42` | Candidate 2 | 42 | reference rerun | keeps the comparison symmetric |
| `EXP_070E_candidate2_seed123` | Candidate 2 | 123 | second seed for candidate 2 | tests robustness for the strongest alternative |
| `EXP_070F_candidate2_seed2026` | Candidate 2 | 2026 | third seed for candidate 2 | completes the minimal variance estimate for candidate 2 |

- Training settings: use the exact frozen config of each Phase 6 candidate and change only the seed. Extend the same template to a third candidate only if Phase 6 leaves the decision genuinely ambiguous.
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
