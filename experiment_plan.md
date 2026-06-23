# Optimized Experimental Plan

This document outlines the revised and streamlined experimental roadmap (condensed from `proposal.md`) down to approximately 18 critical experiments (Runs). The primary goal is to drastically reduce training time and Compute Unit consumption on A100/T4 instances while maintaining rigorous scientific methodology for the thesis.

*(Note: Infrastructure experiments `EXP_000 - EXP_002` are engineering prerequisites, not training runs).*

## Phase 1: Baselines - 3 Runs
Establishing ground-truth performance to demonstrate the superiority of the Multimodal approach over Unimodal architectures.
- `EXP_010_text_only_xlmr_mse`: Text-only baseline (XLM-R).
- `EXP_011_image_only_convnext_meanpool_mse`: Image-only baseline (ConvNeXt).
- `EXP_012_multimodal_convnext_xlmr_concat_mse`: Multimodal baseline (XLM-R + ConvNeXt + Concat + MSE).

## Phase 2 & 3: Backbone Ablation (Condensed) - 5 Runs
Instead of testing 10 different backbones, we focus on the most promising architectures backed by recent scientific literature:
- **Image Branch (Keeping XLM-R):**
  - `EXP_020B_swinb_xlmr_concat_mse`: Swin-B (Hierarchical Vision Transformer, highly effective for complex scenes).
  - `EXP_020D_efficientnetb3_xlmr_concat_mse`: EfficientNet-B3 (Based on the IJACSA 2024 paper showing strong synergy with text).
  - `EXP_020E_siglip_xlmr_concat_mse`: SigLIP (State-of-the-art vision encoder trained with sigmoid loss, exceptionally robust for zero-shot and transfer learning on real-world images).
- **Text Branch (Keeping the best Image backbone from above):**
  - `EXP_030B_bestimage_phobert_concat_mse`: PhoBERT (The gold standard for Vietnamese NLP).
  - `EXP_030D_bestimage_visobert_concat_mse`: ViSoBERT (Optimized for Vietnamese social media text, perfectly matching the Foody domain).

*(Excluded: ViBERT, mDeBERTa, and various Pooling/Filtering experiments to save runs).*

## Phase 4: Fusion Upgrades - 4 Runs
Replacing the rudimentary Concatenation with advanced cross-modal interaction mechanisms.
- `EXP_040B_bestimage_besttext_gmu_mse`: Gated Multimodal Unit (GMU - using a learnable gate to control modality noise).
- `EXP_040C_bestimage_besttext_gatedcrossmodal_mse`: Gated Cross-Modal (Enhancing GMU with cross-modal projections).
- `EXP_041A_bestimage_besttext_film_mse`: FiLM (Feature-wise Linear Modulation - conditioning visual features using textual representation).
- `EXP_041B_bestimage_besttext_crossattention_mse`: Cross-Attention (Allowing visual patches and text tokens to attend to each other).

## Phase 5: Robust Loss & Task Balancing - 2 Runs
Objective: Mitigate the impact of outliers (e.g., review bombing) and resolve the imbalance across the 5 target criteria.
- `EXP_050B_bestfusion_huber`: Huber Loss (Less sensitive to extreme outliers compared to standard MSE).
- `EXP_051D_bestfusion_uncertaintyweighted`: Homoscedastic Task Uncertainty (The model automatically learns to balance the 5 task weights $s_1 \dots s_5$).

## Phase 6: Promising Combinations - 3 Runs
Since combining the "best individual components" doesn't always yield the best overall model, we will evaluate 1 greedy combination and 2 alternative synergies.
- `EXP_060A_bestsequential`: Greedy combination of the absolute best components from Phases 1->5.
- `EXP_060B_swinb_visobert_gmu_uncertainty`: Alternative Candidate 1 (Computationally heavy, high expected accuracy).
- `EXP_060C_efficientnetb3_phobert_film_huber`: Alternative Candidate 2 (Optimized balance between speed and precision).

## Phase 7: Multi-seed Validation - 1 Additional Run
Proving that the final model's success is not due to random seed variance. (Note: The default model from Phase 6 is already trained with Seed 42, so we only need 1 extra training run).
- `EXP_070B_seed123`: Train the winning model from scratch using Seed 123 to check stability.
- `EXP_071_locked_test_evaluation`: Evaluate the best trained model on the Test Set. (This is NOT a training run. It just scores the model to get the final MAE/R2 for the thesis report. Takes ~2 minutes).


---
## COST ESTIMATION SUMMARY
- **Total Training Runs:** 18 Runs (Reduced significantly from the original proposal while keeping core contributions).
- **Estimated Time (A100 + AMP):** $\approx$ 45 minutes/Run $\rightarrow$ Total ~13.5 hours (Can be completed in < 1 day).
- **Estimated Time (T4 + AMP):** $\approx$ 3.5 hours/Run $\rightarrow$ Total ~63 hours (Can be completed in ~3 days, costing ~126 Compute Units).

*The theoretical background and citations from the original document (ViSoBERT, EfficientNet, GMU, FiLM) remain fully valid and should be included in the final thesis report.*
