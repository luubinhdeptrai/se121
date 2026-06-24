# Optimized Experimental Plan

This document outlines the final, streamlined experimental roadmap for the thesis. The primary goal is to ensure rigorous scientific methodology while optimizing execution time on A100/T4 instances.

*(Note: Infrastructure experiments `EXP_000 - EXP_002` are engineering prerequisites, not training runs).*

## Phase 1: Baselines - 3 Runs
Establishing ground-truth performance to demonstrate the superiority of the Multimodal approach over Unimodal architectures.
- `EXP_010_text_only_xlmr_mse`: Text-only baseline (XLM-R).
- `EXP_011_image_only_convnext_mse`: Image-only baseline (ConvNeXt).
- `EXP_012_multimodal_convnext_xlmr_concat_mse`: Multimodal baseline (XLM-R + ConvNeXt + Concat + MSE).

## Phase 2 & 3: Backbone Ablation - 5 Runs
Focusing on the most promising architectures backed by recent scientific literature. 
*Note: To ensure a fair comparison, any new Image/Text backbone is explicitly pre-trained for 20 epochs independently before being fused (15 epochs) within the same notebook.*

- **Image Branch (Keeping XLM-R fixed):**
  - `EXP_020B_swinb_xlmr_concat_mse`: Swin-B (Hierarchical Vision Transformer).
  - `EXP_020D_efficientnetb3_xlmr_concat_mse`: EfficientNet-B3 (Strong synergy with text).
  - `EXP_020E_siglip_xlmr_concat_mse`: SigLIP (SOTA vision encoder trained with sigmoid loss).
  
- **Text Branch (Keeping the best Image backbone from Phase 2):**
  - `EXP_030B_bestimage_phobert_concat_mse`: PhoBERT (Gold standard for Vietnamese NLP).
  - `EXP_030D_bestimage_visobert_concat_mse`: ViSoBERT (Optimized for Vietnamese social media text).

## Phase 4: Fusion Upgrades - 4 Runs
Replacing the rudimentary Concatenation with advanced cross-modal interaction mechanisms. Loads `best_model_train_image.pth` and `best_model_train_text.pth` from previous winners.
- `EXP_040B_bestimage_besttext_gmu_mse`: Gated Multimodal Unit (GMU).
- `EXP_040C_bestimage_besttext_gatedcrossmodal_mse`: Gated Cross-Modal.
- `EXP_041A_bestimage_besttext_film_mse`: FiLM (Feature-wise Linear Modulation).
- `EXP_041B_bestimage_besttext_crossattention_mse`: Cross-Attention.

## Phase 5: Robust Loss & Task Balancing - 3 Runs
Mitigating outliers and resolving imbalance across the 5 target criteria.
- `EXP_050B_bestfusion_huber`: Huber Loss (Less sensitive to extreme outliers).
- `EXP_050C_bestfusion_logcosh`: Log-Cosh Loss (Smoother than Huber, twice-differentiable).
- `EXP_051D_bestfusion_uncertaintyweighted`: Homoscedastic Task Uncertainty.

## Phase 6: Promising Combinations - 4 Runs
Evaluating alternative synergies outside of the greedy sequential path.
- `EXP_060A_bestsequential_full_configuration`: Baseline Candidate (Evaluates the Phase 5 winner on the Test Set, 0 additional training runs).
- `EXP_060B_swinb_visobert_gmu_uncertainty`: Alternative Candidate 1.
- `EXP_060C_efficientnetb3_phobert_film_huber`: Alternative Candidate 2.
- `EXP_060D_efficientnetb3_visobert_crossattention_logcosh`: Alternative Candidate 3 (EfficientNet-B3 + ViSoBERT + Cross-Attention + Log-Cosh).
- `EXP_060E_convnext_phobert_gatedcrossmodal_autoweight`: Alternative Candidate 4 (ConvNeXt + PhoBERT + Gated Cross-Modal + Auto-Weight).

## Phase 7: Multi-Seed Stability Validation - 1 Run
To ensure the robustness of the final model, the best chosen architecture (from Phase 5 or 6) will be re-trained with a different random seed.
- `EXP_070_bestmodel_seed123`: Train best model with `--seed 123`.

---
## COST & TIME ESTIMATION SUMMARY
- **Total Training Runs:** 20 Training Runs (19 architecture runs + 1 seed variant).
- **Total Estimated Epochs:** ~360 epochs (including independent 20-epoch pre-training steps in Phase 2/3 and 15-epoch fusion steps).
- **Estimated Compute Time (A100 + AMP):** $\approx$ 1.5 - 2 minutes/epoch $\rightarrow$ ~9 hours of pure compute. With `patience=5` Early Stopping, realistic runtime is **5-6 hours**.

*The theoretical background and citations from the original document (ViSoBERT, EfficientNet, GMU, FiLM) remain fully valid and should be included in the final thesis report.*
