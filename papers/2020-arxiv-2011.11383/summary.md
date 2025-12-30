# Automated Quality Assessment of Hand Washing Using Deep Learning (2020)

Frame-level WHO step recognition on hospital footage using compact CNNs.

## Data
- **Dataset:** PSKUS hospital videos; 1,854 episodes (1,094 double-annotated).
- **Capture:** 640x480 @ 30 FPS.
- **Training subset:** 378 videos → 309,315 frames; 70/20/10 train/val/test split.

## Method
- **Models:** MobileNetV2 and Xception pretrained on ImageNet (Keras).
- **Temporal handling:** none; per-frame classification with frame extraction.

## Training
- **MobileNetV2:** 50 epochs with early stopping after 10 epochs of no improvement.
- **Xception:** 10 epochs; Adam lr 0.001; categorical loss.

## Results
- **Accuracy:** ~0.64 (MobileNetV2) and ~0.67 (Xception) frame accuracy.

## Availability
- **Data:** not public in this paper; later released as PSKUS on Zenodo.
- **Code/weights:** not provided; no API.
