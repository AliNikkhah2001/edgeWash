# HandWash Multimodal Surgical Rub Recognition

## Overview
Multimodal hand-rub recognition project combining RGB video and additional cues, plus a feedback UI for user guidance.

## Code Structure
- **Type**: Self-contained codebase (no external SDK)
- **Implementation**: Complete training, evaluation, and prediction pipeline
- **Code Included**: Yes - training, evaluation, prediction scripts + web UI
- **Dependencies**: PyTorch, NumPy, OpenCV, Node.js (for UI)

## Model Architecture
- **Primary Models**:
  - **AlexNet + LSTM** (best model, 128 frames)
  - **ResNet-50 + LSTM** (deeper architecture)
  - **Custom CNN + LSTM** (lightweight)
  - **ConvLSTM** (3D-like temporal modeling)
- **Transfer Learning**: ImageNet pre-trained weights (AlexNet, ResNet-50)
- **Focus**: Surgical handwashing (different from WHO steps)

## Video/Temporal Handling
- **Temporal Model**: **LSTM/ConvLSTM** for video sequences
- **3D Convolutions**: ConvLSTM provides 3D-like temporal modeling
- **Sequence Modeling**: LSTM processes frame features sequentially
- **Sequence Length**: Configurable via `--num_frames` (default 128)
- **Frame Sampling**: Uniform sampling from videos
- **Time Series Steps**: LSTM captures temporal dependencies across frames

## Classes & WHO Steps
- **Focus**: Surgical hand rub recognition (8 classes)
- **Dataset-specific**: Not standard WHO steps (custom surgical rub taxonomy)

## Datasets Used
- **Custom Surgical Handwashing Dataset**
  - Source: Google Drive download
  - Format: Pre-processed numpy arrays
  - Size: Not specified in README
  - Public Availability: Yes (via Google Drive link)
  - Download: `wget https://storage.googleapis.com/dl-big-project/dataset.zip`

## Training Details

### Training Script
- **Main Script**: `train.py`
- **Jupyter Notebooks**: 
  - `Model Experiments.ipynb` - Architecture comparison
  - `Model Experiments Testing.ipynb` - Validation/test evaluation

### Command-Line Arguments
- **--arch**: Architecture choice (`convlstm`, `alexnet`, `resnet50`, `custom`)
- **--epochs**: Number of training epochs
- **--batch**: Batch size
- **--num_frames**: Frames per video (default 128 for best model)
- **--lr**: Learning rate
- **--beta1**: Adam optimizer first momentum (default 0.9)
- **--beta2**: Adam optimizer second momentum (default 0.999)
- **--weight_decay**: L2 regularization weight
- **--gamma**: Learning rate scheduler gamma
- **--step_size**: Learning rate scheduler step size
- **--cuda**: Enable CUDA training (flag)
- **--checkpoint**: Path to checkpoint for resuming/fine-tuning
- **--save_dir**: Directory to save trained model
- **--data_aug**: Data augmentation type (`None`, `contrast`, `translate`)
- **--aug_prob**: Probability of applying augmentation

### Hyperparameter Tuning Experiments
- Tested 4 architectures (ConvLSTM, AlexNet+LSTM, ResNet-50+LSTM, Custom+LSTM)
- Varied batch size, learning rate, spatial dimensions
- Best model: **AlexNet + LSTM with 128 frames**
- Data augmentation: Contrast and translation tested

### Best Model Weights
- **Available**: Yes (Google Drive)
- **Download**: `wget https://storage.googleapis.com/dl-big-project/alexnet_128.pt`
- **Note**: Non-deterministic RNN behavior may cause slight variations

## Running Instructions

### 1. Download Dataset
```bash
wget https://storage.googleapis.com/dl-big-project/dataset.zip
unzip dataset.zip
```

### 2. Download Pre-trained Model (Optional)
```bash
wget https://storage.googleapis.com/dl-big-project/alexnet_128.pt
mkdir -p save_weights
mv alexnet_128.pt save_weights/
```

### 3. Train Model
```bash
# Default model (Custom CNN + LSTM)
python train.py

# AlexNet + LSTM (best model)
python train.py --arch alexnet --num_frames 128 --epochs 50 --cuda

# ResNet-50 + LSTM
python train.py --arch resnet50 --num_frames 128 --cuda

# ConvLSTM
python train.py --arch convlstm --cuda

# With data augmentation
python train.py --arch alexnet --num_frames 128 --data_aug contrast --aug_prob 0.5
```

### 4. Evaluate Model
```bash
# Evaluate on validation set
python evaluate.py --checkpoint ./alexnet_128.pt --arch alexnet --dataset validation

# Evaluate on test set
python evaluate.py --checkpoint ./alexnet_128.pt --arch alexnet --dataset test
```

### 5. Predict on New Video
```bash
python predict.py --checkpoint ./alexnet_128.pt --video_path /path/to/video.mp4 --arch alexnet
```

### 6. Run Web UI (Optional)
```bash
cd handwashUI
npm install
npm start
# UI available at http://localhost:3000
```

## Key Features
- **Video-based temporal modeling** (LSTM/ConvLSTM)
- **Multiple architectures** for comparison
- **Extensive hyperparameter tuning** documented
- **Pre-trained weights available** (best model: AlexNet+LSTM)
- **Data augmentation** (contrast, translation)
- **Web UI** for user feedback (Heroku + StreamLit)
- **Complete pipeline** (train, evaluate, predict)

## Web Demo
- **Hosted**: Heroku
- **Framework**: StreamLit
- **URL**: https://handwashdl.herokuapp.com/ (may not be active)

## Limitations
- **Non-deterministic CUDA RNN** (results may vary slightly)
- **Surgical focus** (not standard WHO steps)
- **Custom dataset** (specific to surgical handwashing)
- **Video pre-processing required** (uniform sampling)

## Availability
- **Code**: Open (license not specified)
- **Dataset**: Yes (Google Drive)
- **Trained Weights**: Yes (Google Drive - AlexNet+LSTM)
- **External API**: No (but has web UI)
