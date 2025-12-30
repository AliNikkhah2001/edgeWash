# EdgeWash WHO Movement Classifier

## Overview
Lightweight edge-focused pipeline for WHO hand-washing movement recognition with exportable on-device models.

## Code Structure
- **Type**: Self-contained codebase (no external SDK)
- **Implementation**: Complete training pipeline with multiple architectures
- **Code Included**: Yes - preprocessing, training, evaluation, deployment scripts
- **Dependencies**: TensorFlow, Keras, OpenCV, NumPy

## Model Architecture
- **Primary Models**:
  - **MobileNetV2** (frame-based, default)
  - **TimeDistributed + GRU** (video-based with temporal modeling)
  - **Two-stream network** (RGB + Optical Flow fusion)
- **Base Models**: MobileNetV2, ResNet, VGG (configurable via `HANDWASH_NN`)
- **Transfer Learning**: ImageNet pre-trained weights
- **Edge Deployment**: TFLite conversion available (`convert-model-to-tflite.py`)

## Video/Temporal Handling
- **Frame-based**: Single image classification (no temporal modeling)
- **Video-based**: **GRU** for sequence modeling (not 3D conv)
- **Two-stream**: 
  - Spatial stream: RGB frames
  - Temporal stream: **Optical Flow** frames
  - Fusion: Late fusion of predictions
- **3D Convolutions**: No (uses GRU for temporal dependencies)
- **Sequence Length**: Configurable via `HANDWASH_NUM_FRAMES` (default 5)
- **Time Series Steps**: GRU processes frame sequences temporally

## Classes & WHO Steps
- **Total Classes**: 7
  - 6 WHO movements (palm-to-palm, palm-over-dorsum, interlaced, back-of-fingers, thumb-rub, fingertips)
  - 1 "Other" class (turning off faucet, non-washing actions)
- **WHO Coverage**: 6 core WHO movements + faucet control

## Datasets Used
1. **PSKUS Hospital Dataset**
   - 3,185 hospital videos (Latvia)
   - Public (Zenodo)
2. **METC Lab Dataset**
   - Lab-based recordings (Riga Stradins University)
   - Public (Zenodo)
3. **Kaggle WHO6 Dataset**
   - ~292 Kaggle videos resorted to WHO structure
   - Public (GitHub mirror)

## Training Details

### Training Scripts
- **Frame-based**: `*-classify-frames.py` (e.g., `pskus-classify-frames.py`)
- **Video-based**: `*-classify-videos.py` (TimeDistributed + GRU)
- **Two-stream**: `*-classify-merged-network.py` (RGB + Optical Flow)

### Hyperparameters (Environment Variables)
- **HANDWASH_NN**: Base model name (default: "MobileNetV2")
- **HANDWASH_NUM_LAYERS**: Trainable layers count (default: 0 = fine-tune top only)
- **HANDWASH_NUM_EPOCHS**: Max epochs (default: 20, early stopping enabled)
- **HANDWASH_NUM_FRAMES**: Frames per video sequence (default: 5)
- **HANDWASH_SUFFIX**: User-defined experiment suffix
- **HANDWASH_PRETRAINED_MODEL**: Path to pretrained model for transfer learning
- **HANDWASH_EXTRA_LAYERS**: Additional dense layers before output (default: 0)

### Preprocessing Pipeline
1. Download datasets: `dataset-*/get-and-preprocess-dataset.sh`
2. Extract frames: `dataset-*/separate-frames.py`
3. (Optional) Calculate optical flow: `calculate-optical-flow.py`
4. Train model: `*-classify-frames.py` / `*-classify-videos.py` / `*-classify-merged-network.py`

## Running Instructions

### 1. Download Datasets
```bash
# PSKUS Hospital Dataset
./dataset-pskus/get-and-preprocess-dataset.sh

# METC Lab Dataset
./dataset-metc/get-and-preprocess-dataset.sh

# Kaggle WHO6 Dataset
./dataset-kaggle/get-and-preprocess-dataset.sh
```

### 2. (Optional) Calculate Optical Flow
```bash
python calculate-optical-flow.py dataset-pskus
python calculate-optical-flow.py dataset-metc
python calculate-optical-flow.py dataset-kaggle
```

### 3. Train Models
```bash
# Frame-based (MobileNetV2, 40 epochs)
export HANDWASH_NUM_EPOCHS=40
export HANDWASH_NN=MobileNetV2
python pskus-classify-frames.py

# Video-based (TimeDistributed + GRU, 5 frames)
export HANDWASH_NUM_FRAMES=5
python pskus-classify-videos.py

# Two-stream (RGB + Optical Flow)
python pskus-classify-merged-network.py
```

### 4. Convert to TFLite (Edge Deployment)
```bash
python convert-model-to-tflite.py
```

### 5. Run Streamlit Demo (Optional)
```bash
streamlit run tools/edgewash_streamlit_app.py
```

## Key Features
- **Multiple architectures** for comparison (frame/video/two-stream)
- **GRU-based temporal modeling** (not 3D conv)
- **Optical flow integration** for motion-aware predictions
- **Edge-ready** (TFLite conversion for on-device inference)
- **Complete preprocessing** (video → frames → optical flow)
- **Multiple datasets** (PSKUS, METC, Kaggle)
- **Configurable hyperparameters** (via environment variables)

## Limitations
- **GRU-based temporal modeling** (not 3D conv, sequential processing slower than 3D conv)
- **No pre-trained weights** (must train from scratch)
- **Manual hyperparameter tuning** (via environment variables, no auto-tuning)

## Availability
- **Code**: Open (license not specified)
- **Datasets**: Public (PSKUS, METC, Kaggle via Zenodo/GitHub)
- **Trained Weights**: Not included (training required)
- **External API**: No
