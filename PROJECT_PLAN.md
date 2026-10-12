# Project Plan

## Project title
Evaluating the Robustness of Deep Learning Models for Breast Cancer Detection in Mammography Images Under Image Quality Variations

## Project description
This project will evaluate how image-quality changes affect the performance of deep-learning models that classify mammography images for breast cancer. We will compare model performance on original images and on images altered with controlled variations such as noise, blur, reduced contrast, and lower resolution.

## Problem statement
A deep-learning model may perform well on clear test images but become less reliable when image quality changes. This study will measure how selected image-quality variations affect cancer-detection performance and compare the robustness of the chosen models.

## Research question
How do controlled image-quality variations affect the performance of selected deep-learning models for breast cancer detection in mammography images?

## Objectives
1. Select a suitable mammography dataset and define the prediction task.
2. Prepare patient-level training, validation, and test splits.
3. Evaluate the selected models on original test images.
4. Evaluate the models on test images with controlled quality variations.
5. Compare performance using agreed evaluation metrics.

## Proposed dataset
CBIS-DDSM is the proposed primary dataset. The group will confirm dataset access, labels, and permitted use before beginning experiments. INbreast may be considered for additional evaluation if access and time allow.

## Proposed image-quality variations
- Noise
- Blur
- Reduced contrast
- Lower resolution

The group will agree on the exact methods and severity levels before running experiments.

## Proposed evaluation metrics
- Area under the ROC curve (AUC)
- Sensitivity
- Specificity
- F1-score

## Team
- Member 1: Literature review and research framing
- Member 2: Dataset and preprocessing
- Member 3: Model development and experiments
- Member 4: Evaluation and reporting

## Important project notes
- Split data by patient or case, not by individual image, to reduce data leakage.
- Do not upload mammography images or other restricted data to this repository.
- Record experiment settings and results in `EXPERIMENT_LOG.md`.
- Treat this as a research evaluation, not a clinical diagnostic tool.
- The group must confirm the final dataset, models, methods, and metrics before experiments begin.

## Current status
Project setup and study design are in progress.
