# TensorFlow Handwash Monitoring Demo

## Overview
Lightweight TF/Keras prototype that detects hand presence and dispenser use from depth video near sinks using transfer learning.

## Code Structure
- **Type**: Self-contained codebase (no external SDK)
- **Implementation**: Single Python script (`dettolhandwash.py`)
- **Code Included**: Yes - inference script provided
- **Dependencies**: TensorFlow 1.x, OpenCV

## Model Architecture
- **Base Model**: Pre-trained CNN (likely MobileNet/Inception via TensorFlow 1.x retrain workflow)
- **Approach**: **Single-frame classification** (no video/temporal modeling)
- **Transfer Learning**: Uses TensorFlow's `retrain.py` workflow with ImageNet weights
- **Model Files**: 
  - `tf_files/retrained_graph.pb` (frozen graph)
  - `tf_files/retrained_labels.txt` (7 classes)

## Video/Temporal Handling
- **Temporal Model**: None - purely frame-by-frame classification
- **3D Convolutions**: No
- **Sequence Modeling**: No (RNN/LSTM/GRU)
- **Duration Tracking**: External logic in Python script (5 seconds per step)
- **Time Series Steps**: Sequential frame predictions tracked with counters

## Classes & WHO Steps
- **Total Classes**: 7
  - "no hands"
  - "step 2" through "step 7" (WHO steps)
- **WHO Coverage**: 6 core WHO movements

## Datasets Used
- **Training Data**: Custom dataset (not provided in repository)
- **Format**: Video frames labeled by WHO step
- **Public Availability**: No dataset bundled

## Training Details
- **Training Script**: Uses TensorFlow's `retrain.py` (transfer learning workflow)
- **Hyperparameters**: Standard TensorFlow defaults
  - Learning rate, batch size, epochs configured via TF retrain flags
- **Pre-trained Weights**: ImageNet (via TensorFlow Hub)
- **Training Process**: Not included in repo (only inference script provided)

## Running Instructions

### Prerequisites
```bash
pip install tensorflow==1.15 opencv-python
```

### Inference
```bash
python dettolhandwash.py
# Requires: 
#   - tf_files/retrained_graph.pb
#   - tf_files/retrained_labels.txt
#   - Connected webcam
```

### Training (External)
Must use TensorFlow's transfer learning tools:
```bash
# Example with TensorFlow for Poets workflow
python -m scripts.retrain \
  --image_dir=<your_dataset> \
  --output_graph=tf_files/retrained_graph.pb \
  --output_labels=tf_files/retrained_labels.txt
```

## Key Features
- Real-time webcam processing
- Step duration tracking (5 seconds per step)
- Sequential step progression logic
- Lightweight inference (suitable for edge devices)

## Limitations
- **No temporal modeling** - ignores motion patterns
- **Legacy TensorFlow 1.x** - outdated framework
- **No training code included** - only inference
- **No pre-trained weights** - must train from scratch
- **Frame-based only** - duration tracked externally

## Availability
- **Code**: Open (MIT license assumed)
- **Datasets**: Not bundled (must provide own)
- **Trained Weights**: Not provided
- **External API**: No
