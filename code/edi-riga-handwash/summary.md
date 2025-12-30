# EDI-Riga Handwashing Movement Classifiers

## Overview
Hospital-focused WHO step classifiers with training scripts and pretrained checkpoints maintained by EDI-Riga (identical to EdgeWash codebase - same research group).

## Code Structure
- **Type**: Self-contained codebase (no external SDK)
- **Implementation**: Complete training pipeline with multiple architectures
- **Code Included**: Yes - preprocessing, training, evaluation scripts
- **Dependencies**: TensorFlow, Keras, OpenCV, NumPy
- **Note**: This is essentially the same codebase as EdgeWash, maintained by the same research group

## Model Architecture
- **Primary Models**:
  - **MobileNetV2** (frame-based, default)
  - **TimeDistributed + GRU** (video-based with temporal modeling)
  - **Two-stream network** (RGB + Optical Flow fusion)
- **Base Models**: MobileNetV2, ResNet, VGG (configurable)
- **Transfer Learning**: ImageNet pre-trained weights
- **Focus**: Hospital deployment and real-world validation

## Video/Temporal Handling
- **Frame-based**: Single image classification (no temporal modeling)
- **Video-based**: **GRU** for sequence modeling (not 3D conv)
- **Two-stream**: RGB + Optical Flow fusion
- **3D Convolutions**: No (uses GRU for temporal dependencies)
- **Sequence Length**: Configurable (default 5 frames)

## Classes & WHO Steps
- **Total Classes**: 7
  - 6 WHO movements
  - 1 "Other" class
- **WHO Coverage**: 6 core WHO movements + faucet control

## Datasets Used
- **PSKUS Hospital Dataset** (primary focus)
- **METC Lab Dataset**
- **Kaggle WHO6 Dataset**

## Training Details
Identical to EdgeWash (see EdgeWash summary for full details):
- Training scripts: `*-classify-frames.py`, `*-classify-videos.py`, `*-classify-merged-network.py`
- Hyperparameters: Environment variables (HANDWASH_NN, HANDWASH_NUM_EPOCHS, etc.)
- Preprocessing: Video → frames → optical flow

## Running Instructions
Same as EdgeWash:
```bash
# 1. Download datasets
./dataset-pskus/get-and-preprocess-dataset.sh
./dataset-metc/get-and-preprocess-dataset.sh
./dataset-kaggle/get-and-preprocess-dataset.sh

# 2. Train
export HANDWASH_NUM_EPOCHS=40
python pskus-classify-frames.py
```

## Key Features
- **Hospital-focused** validation and deployment
- Multiple architectures (frame/video/two-stream)
- GRU-based temporal modeling
- Optical flow integration
- Complete preprocessing pipeline

## Availability
- **Code**: Open (maintained by EDI-Riga)
- **Datasets**: Public (PSKUS, METC, Kaggle)
- **Trained Weights**: Not included (training required)
- **External API**: No
