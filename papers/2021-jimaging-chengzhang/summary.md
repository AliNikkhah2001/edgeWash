# Designing a Computer-Vision Application: Hand-Hygiene Assessment in an Open-Room Environment (J. Imaging 2021)

Open-room deployment study with a new side-view dataset and baseline action-recognition models.

## Data
- **Dataset:** Class-23 open-room set; 23 participants, 105 untrimmed videos, 3 camera views.
- **Capture:** GoPro side-view cameras, 1080p @ 30 FPS.
- **Labels:** per-frame actions; includes non-hygiene actions and occlusions.

## Method
- **Baseline:** ResNet50 single-frame classifier with majority vote over clips.
- **Temporal models:** ResNet50 + LSTM and ResNet50 + TRN over k=10 frames.
- **Other modalities:** RGB, optical flow, hand masks, skeleton joints, waterflow ROI explored.

## Training
- **Initialization:** ImageNet-pretrained ResNet50.
- **Hyperparameters:** 350 epochs, batch 32, SGD, lr 0.001 with drops at 200 and 300; multi-scale crop + horizontal flip.

## Results
- **Findings:** temporal models offer limited gains over spatial-only; cross-scenario performance drops due to background bias.

## Availability
- **Data/code:** dataset not public; code/weights not released; no API.
