# Synthetic Hand-Washing Data Generation

## Overview
Rendering and augmentation toolkit for creating synthetic handwashing datasets plus training recipes for baseline models.

## Code Structure
- **Type**: Self-contained codebase (no external SDK)
- **Implementation**: Synthetic data generation + training scripts for 3 model types
- **Code Included**: Yes - data preparation and training scripts
- **Dependencies**: PyTorch, TensorFlow, Open3D, Blender (for rendering)

## Model Architecture
- **Primary Models**:
  - **Inception-V3** (RGB images, TensorFlow/Keras)
  - **YOLOv8** (RGB detection + segmentation, Ultralytics)
  - **PointNet** (point cloud/depth, PyTorch)
- **Transfer Learning**: ImageNet pre-trained (Inception-V3)
- **Focus**: Synthetic-to-real domain adaptation

## Video/Temporal Handling
- **Temporal Model**: None - frame-based classification only
- **3D Convolutions**: No
- **Sequence Modeling**: No (RNN/LSTM/GRU)
- **Frame-based**: All three models operate on single frames/point clouds
- **Time Series Steps**: Not applicable (static frame classification)

## Classes & WHO Steps
- **Total Classes**: 8 WHO-inspired gestures
  - Not identical to WHO steps (synthetic taxonomy)
- **Variations**: 4 characters, 4 environments

## Datasets Used
- **Synthetic Hand-Washing Gesture Dataset (Özakar & Gedikli, 2025)**
  - **Size**: 96,000 frames (64 minutes total)
  - **Modalities**: RGB, depth, depth-isolated, hand masks
  - **Format**: Images organized by gesture class
  - **Public Availability**: Yes (Google Drive, 5 download links)
  - **License**: CC BY (Attribution)

## Training Details

### Preprocessing Scripts
- **Inception-V3**: `inception/create_rgb_pickle.py`
- **PointNet**: `pointnet/create_pcd_pickle.py`
- **YOLOv8**: `yolo/ready_for_training.py`

### Training Scripts
- **Inception-V3**: `inception/train_inception.py`
- **YOLOv8 Classification**: `yolo/train.py`
- **YOLOv8 Segmentation**: `yolo/train-seg.py`
- **PointNet**: `pointnet/pointnet_train.py`

### Hyperparameters
- **YOLOv8**: Configured in YAML files (`yolo/yaml/training1-5.yaml`)
- **Inception-V3**: Standard TensorFlow/Keras defaults
- **PointNet**: PyTorch defaults for point cloud classification

### Pre-trained Models
- **Available**: Yes (included in repository or downloadable)
- **Models**: Inception-V3, YOLOv8n, PointNet weights provided

## Running Instructions

### 1. Download Dataset
```bash
# Download from Google Drive links in README (5 parts)
# Extract all parts to dataset folder
```

### 2. Prepare Data
```bash
# For Inception-V3 (RGB)
python inception/create_rgb_pickle.py

# For PointNet (Point Cloud)
python pointnet/create_pcd_pickle.py

# For YOLOv8 (Detection/Segmentation)
python yolo/ready_for_training.py
python yolo/ready_for_training-seg.py
```

### 3. Train Models
```bash
# Inception-V3
python inception/train_inception.py

# YOLOv8 Classification
python yolo/train.py

# YOLOv8 Segmentation
python yolo/train-seg.py

# PointNet
python pointnet/pointnet_train.py
```

## Key Features
- **Multi-modal synthetic data** (RGB, depth, masks)
- **Large-scale** (96k frames)
- **Domain adaptation** (synthetic → real transfer learning)
- **Pre-trained models** included
- **Multiple architectures** (CNN, YOLO, PointNet)
- **Public dataset** with CC BY license

## Limitations
- **Frame-based only** (no temporal modeling)
- **Synthetic-to-real gap** (domain shift challenges)
- **Not standard WHO steps** (8 synthetic gestures)
- **Blender rendering** required for extending dataset

## Research Paper
- **Title**: "Hand Washing Gesture Recognition Using Synthetic Dataset"
- **Authors**: Rüstem Özakar, Eyüp Gedikli
- **Journal**: Journal of Imaging, 2025
- **DOI**: 10.3390/jimaging11070208

## Availability
- **Code**: Open (CC BY license)
- **Dataset**: Yes (Google Drive, CC BY license)
- **Trained Weights**: Yes (included in repo)
- **External API**: No
