# GD-Net package
# Re-implementation of:
#   Lin et al. (2024) "GD-Net: An Integrated Multimodal Information Model Based on
#   Deep Learning for Cancer Outcome Prediction and Informative Feature Selection"
#   Journal of Cellular and Molecular Medicine, 28:e70221. DOI: 10.1111/jcmm.70221
#
# The authors' official repo (https://github.com/JackiLin/GD-Net) only ships the
# contrastive encoder (pcl/builder.py) and augmentations (pcl/loader.py).
# This package vendors that code (with CPU fixes + speedups) and ADDS the missing
# pieces described in the paper: the training loop, the Cox-Elastic-Net risk
# model, and the XGBoost feature-selection module.
