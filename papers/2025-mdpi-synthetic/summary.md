# Hand Washing Gesture Recognition Using Synthetic Dataset (J. Imaging 2025)

Introduces a 96k-frame synthetic dataset with RGB, depth, and masks covering 8 WHO-inspired gestures.

- Media/architecture: RGB/depth/mask frames; benchmarks InceptionV3 (image CNN), YOLOv8n (classification/segmentation), and PointNet (point clouds).
- Contribution: provides a large synthetic dataset and reports ~79% real-world accuracy when training YOLOv8n-seg on synthetic data.
- Availability: synthetic data public; code and pretrained models provided in the upstream repo; no external API.
