# Hand Hygiene Assessment via Joint Step Segmentation and Key Action Scorer (2023)

Fine-grained framework that segments WHO steps and scores key actions using a convolution-transformer.

## Data
- **Dataset:** HHA300 (300 videos; 60 participants; average >1000 frames per video).
- **Capture:** CCD cameras, multiple scenes.
- **Labels:** frame-level step labels + per-video quality scores (6 WHO steps).

## Method
- **Features:** RGB + optical flow; 1024-d I3D features pretrained on Kinetics.
- **Model:** multi-stage convolution-transformer with linear transformer blocks.
- **Temporal handling:** explicit step segmentation over full sequences, then key-action scoring.

## Training
- **Preprocessing:** resize frames to 224x224; extract optical flow.
- **Hyperparameters:** Adam lr 0.0005, weight decay 0.
- **Metrics:** Acc, Edit, F1@{10,25,50}, Spearman rho, relative L2 distance.

## Results
- **Findings:** improved step segmentation and scoring over multi-stage CNN baselines.

## Availability
- **Data/code:** HHA300 not public; code/weights not released; no API.
