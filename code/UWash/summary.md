# UWash Smartwatch Handwashing Assessment

## Overview
IMU-based smartwatch pipeline for detecting and scoring handwashing quality/compliance, with preprocessing and model training scripts.

## Code Structure
- **Type**: Self-contained codebase (no external SDK)
- **Implementation**: Complete training and evaluation pipeline
- **Code Included**: Yes - preprocessing, training, evaluation scripts
- **Dependencies**: PyTorch, NumPy, scikit-learn

## Model Architecture
- **Primary Model**: **1D ResNet** (ResNet1D-18/34/50)
- **Alternative Models**:
  - U-Net (1D segmentation)
  - Transformer (UTFormer for time-series)
  - MLP (multilayer perceptron)
  - MobileNet (1D adaptation)
- **Approach**: Wearable sensor-based classification
- **Input**: IMU time-series (accelerometer, gyroscope, magnetometer)

## Video/Temporal Handling
- **Modality**: **No video** - pure IMU sensor data
- **Temporal Model**: 1D CNNs for time-series sequences
- **3D Convolutions**: N/A (1D signals, not video)
- **Sequence Modeling**: Implicit in 1D CNN architecture
- **Segment Lengths**: 64 or 128 samples (configurable)
- **Time Series Steps**: Quality assessment over handwashing duration

## Classes & WHO Steps
- **Total Classes**: 10 quality assessment classes
- **Focus**: Handwashing **quality scoring**, not just step detection
- **WHO Alignment**: Follows WHO guidelines for quality assessment

## Datasets Used
- **UWash Smartwatch Dataset**
  - Source: Google Drive (link in README)
  - Format: Raw IMU CSV files → preprocessed numpy arrays
  - Sensors: 3-axis accelerometer, gyroscope, magnetometer
  - Sessions: Multiple per participant
  - Public Availability: Yes (via Google Drive)

## Training Details

### Preprocessing Pipeline
1. **decode_sensor_data.py** - Decode raw IMU data from smartwatch
2. **shift_data.py** - Temporal alignment and synchronization
3. **augment_data.py** - Data augmentation for IMU signals

### Training Scripts
- **normal_64.py** / **normal_128.py** - Standard training (segment length 64/128)
- **li_64.py** - Location-independent training
- **ui_64.py** - User-independent training

### Evaluation Scripts
- **normal_eval_64.py** / **normal_eval_128.py** - Standard evaluation
- **li_eval_64.py** - Location-independent evaluation
- **ui_eval_64.py** - User-independent evaluation

### Hyperparameters
- **Segment Length**: 64 or 128 samples
- **Training Strategies**: 
  - Normal (standard train/test split)
  - Location-independent (generalize across locations)
  - User-independent (generalize across users)
- **Model Architectures**: ResNet1D (default), UTFormer, MobileNet, U-Net, MLP
- **Batch Size, Learning Rate, Epochs**: Configurable in TrainConfig.py

## Running Instructions

### 1. Download Dataset
```bash
# Download from Google Drive link in README
# Link: https://drive.google.com/file/d/1ZRdRiwXp4xbFUWIIjIQ0OEK6gK0cwODN/view
```

### 2. Preprocess Data
```bash
# IMPORTANT: Modify base_path in each script first
python pre_validation/decode_sensor_data.py
python pre_validation/shift_data.py
python pre_validation/augment_data.py
```

### 3. Configure Paths
```bash
# Edit DatasetConfig.py to point to your preprocessed dataset
vim UWasher/data/DatasetConfig.py
```

### 4. Train Model
```bash
# Standard training (segment length 64)
python UWasher/train_eval/normal_64.py

# Standard training (segment length 128)
python UWasher/train_eval/normal_128.py

# Location-independent
python UWasher/train_eval/li_64.py

# User-independent
python UWasher/train_eval/ui_64.py
```

### 5. Evaluate Model
```bash
# Standard evaluation
python UWasher/train_eval/normal_eval_64.py

# Location-independent evaluation
python UWasher/train_eval/li_eval_64.py

# User-independent evaluation
python UWasher/train_eval/ui_eval_64.py
```

## Key Features
- **Wearable-based** (no video, privacy-preserving)
- **Quality assessment** (not just step detection)
- **Multiple training strategies** (location/user-independent)
- **Multiple architectures** (ResNet1D, Transformer, U-Net, MLP, MobileNet)
- **Complete preprocessing pipeline** (raw IMU → train/test splits)

## Limitations
- **Requires wearables** (smartwatch with IMU sensors)
- **IMU-only** (no visual information)
- **Quality scoring focus** (not fine-grained WHO step detection)
- **Trained weights not included** (must train from scratch)

## Availability
- **Code**: Open (MIT license)
- **Dataset**: Yes (Google Drive)
- **Trained Weights**: Not included (training required)
- **External API**: No
