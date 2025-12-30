# Shadow Augmentation for Handwashing Action Recognition: From Synthetic to Real Datasets (MMSP 2024)

Study of shadow-induced domain shift and a shadow augmentation strategy for robust handwashing recognition.

## Data
- **Synthetic:** handwashing pose dataset with controlled shadow size/intensity/placement.
- **Real:** Portable51 (51 participants), Farm23 (23 participants), Kaggle Hand Wash Dataset (indoor).
- **Classes:** 7 WHO rubbing actions.

## Method
- **Models:** MobileNetV3 (synthetic experiments), ResNet50/ResNet152/ViT (real datasets), all ImageNet-pretrained.
- **Augmentation:** add synthetic shadows (e.g., 3,200 + 2,400 images) to shift breakdown points.
- **Temporal handling:** per-frame classification; no 3D/sequence model.

## Results
- **Findings:** heavier and larger shadows improve robustness; shadow augmentation boosts OoD performance on Farm23.

## Availability
- **Data/code:** Portable51/Farm23 not public; code/weights not released; no API.
