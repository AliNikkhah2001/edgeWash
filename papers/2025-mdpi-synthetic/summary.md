# Hand Washing Gesture Recognition Using Synthetic Dataset (J. Imaging 2025)

Introduces a large synthetic dataset and benchmarks InceptionV3, YOLOv8n, and PointNet for gesture recognition.

## Data
- **Synthetic:** 96,000 frames at 25 FPS (64 minutes); 8 gestures, 4 characters, 4 environments.
- **Resolution:** 960x540; includes RGB, depth, isolated depth, and hand masks.

## Method
- **InceptionV3:** ROI crops 96x96; Dropout 0.3; Dense 64; Adam lr 1e-4; 2 epochs; 5-fold CV.
- **YOLOv8n / YOLOv8n-seg:** 5-fold CV; 75/25 train/val split; 5 epochs; flipud/fliplr=0.
- **PointNet:** 512-point clouds; 6 epochs; Adam lr 1e-4; 5-fold CV.
- **Temporal handling:** frame-level models; no 3D/sequence network.

## Results
- **Real-world accuracy:** 56.9% (InceptionV3), 76.3% (YOLOv8n), 79.3% (YOLOv8n-seg).

## Availability
- **Data/code:** synthetic data and pretrained models public via upstream repo; no API.
