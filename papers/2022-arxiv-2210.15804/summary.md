# Handwashing Action Detection System for an Autonomous Social Robot (2022)

CNN with attention for real-time handwashing step recognition to drive robot feedback.

## Data
- **Dataset:** Kaggle Hand Wash Dataset (292 videos, 12 steps, 5 sink backgrounds).
- **Preparation:** videos converted to frames; left/right variants merged into 6 classes.
- **Splits:** environment-level split (train on 4 backgrounds, test on 1 unseen background).

## Method
- **Model:** ResNet-50 backbone with Channel Spatial Attention Bilinear (CSAB) module.
- **Temporal handling:** none; per-frame classification.

## Training
- **Framework:** Keras; transfer learning with data augmentation.
- **Hyperparameters:** not specified in the paper.

## Results
- **Findings:** attention module improves fine-grained step classification and generalization.

## Availability
- **Data/code:** dataset public via Kaggle; code/weights not released; no API.
