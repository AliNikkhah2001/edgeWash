# Handwashing Research Hub

Aggregated code, papers, datasets, models, and experiment ideas for automated handwashing assessment.
_Last updated: 2025-12-31 10:16 UTC_

## Structure
- `code/`: cloned codebases and pipelines
- `papers/`: papers with `summary.md`, `tags.md`, and `paper.pdf`
- `datasets/`: storage location for raw/processed datasets (gitignored)
- `models/`: exported weights and model cards
- `evaluation/`: benchmarks and result artifacts
- `ideas/`: future experiment notes and design sketches

## Codebases
- [ ] **TensorFlow Handwash Monitoring Demo** (`code/Handwash-Monitoring`) — tags: depth, cnn, dispenser-detection, tensorflow — source: https://github.com/SidhantSarkar/Handwash-Monitoring
<details>
<summary>Show details</summary>

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

</details>

- [ ] **UWash Smartwatch Handwashing Assessment** (`code/UWash`) — tags: wearable, imu, smartwatch, quality-assessment — source: https://github.com/aiotgroup/UWash
<details>
<summary>Show details</summary>

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

</details>

- [ ] **WashWise - Real-Time Handwashing Step Tracker** (`code/WashWise`) — tags: yolo, real-time, who-steps, streamlit, roboflow — source: https://github.com/aarnavshah12/WashWise
<details>
<summary>Show details</summary>

# WashWise - Real-Time Handwashing Step Tracker

## Overview
YOLOv8-based demo from Roboflow for real-time WHO step classification on sink camera footage with Streamlit GUI.

## Code Structure
- **Type**: **Uses Roboflow SDK** (cloud-based inference)
- **Implementation**: Client application using Roboflow InferencePipeline
- **Code Included**: Yes - inference and GUI code
- **Dependencies**: roboflow-inference, opencv-python, python-dotenv

## Model Architecture
- **Model**: **YOLOv8** via Roboflow Workflows
- **Approach**: Real-time video classification (frame-by-frame)
- **Training Location**: Roboflow cloud platform (not in codebase)
- **Deployment**: InferencePipeline (streaming inference)

## Video/Temporal Handling
- **Temporal Model**: None - purely frame-by-frame predictions
- **3D Convolutions**: No
- **Sequence Modeling**: No (RNN/LSTM/GRU)
- **Duration Tracking**: External logic via `StepTracker` class
- **Time Series Steps**: Accumulates time per step (10 seconds target per step)
- **Real-time Processing**: Up to 30 FPS via Roboflow pipeline

## Classes & WHO Steps
- **Total Classes**: 8 WHO handwashing steps
  - Palms together
  - Right palm on left dorsum
  - Left palm on right dorsum
  - Fingers interlaced
  - Right nails
  - Left nails
  - Right thumb
  - Left thumb
- **WHO Coverage**: All 8 core movements

## Datasets Used
- **Training Data**: Custom dataset trained on Roboflow platform
- **Format**: Uploaded to Roboflow (not included in repository)
- **Public Availability**: No (user must provide own Roboflow workflow)

## Training Details
- **Training Script**: None (training done on Roboflow platform)
- **Hyperparameters**: Configured via Roboflow UI (not exposed in code)
- **Model Access**: Requires Roboflow API credentials
- **Pre-trained Weights**: Hosted on Roboflow (not downloadable directly)

## Running Instructions

### 1. Setup Environment
```bash
pip install -r requirements.txt
cp .env.example .env
```

### 2. Configure Roboflow Credentials
Edit `.env` file:
```bash
API_KEY=your_roboflow_api_key
WORKSPACE_NAME=your_workspace_name
WORKFLOW_ID=your_deployed_workflow_id
```

### 3. Run Application
```bash
python main.py
```

### 4. Usage
- Application opens two OpenCV windows:
  - **Live Feed**: Shows webcam with current step prediction
  - **WashWise Status**: Progress bars for each step (10-second target)
- Perform handwashing steps in any order
- Each step needs 10 seconds of accumulated time
- Completion message displays when all steps are done

## Key Features
- **Real-time inference** (up to 30 FPS)
- **Cloud-based model** (no local training required)
- **Flexible order** (steps can be completed non-sequentially)
- **Visual progress tracking** (OpenCV GUI with progress bars)
- **Configurable duration** (target_duration_seconds parameter)

## Limitations
- **Cloud-dependent** (requires Roboflow API and internet)
- **No temporal modeling** - frame-by-frame predictions only
- **No local training** - must use Roboflow platform
- **API costs** - Roboflow inference charges apply
- **No offline mode** - requires constant API connection

## Customization Options
- **Step Duration**: Change `target_duration_seconds` in `main.py`
- **Step Names**: Update `steps` set in `step_tracker.py`
- **UI Colors**: Modify color values in `status_display.py`
- **FPS**: Adjust `max_fps` parameter in `main.py`

## Availability
- **Code**: Open (license not specified)
- **Dataset**: Not included (must provide own)
- **Trained Weights**: Hosted on Roboflow (requires API access)
- **External API**: Yes (Roboflow)

</details>

- [ ] **EdgeWash WHO Movement Classifier** (`code/edgewash`) — tags: who-steps, edge, classification, vision — source: https://github.com/AliNikkhah2001/edgeWash
<details>
<summary>Show details</summary>

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

</details>

- [ ] **EDI-Riga Handwashing Movement Classifiers** (`code/edi-riga-handwash`) — tags: hospital, who-steps, classification, pytorch — source: https://github.com/edi-riga/handwash
<details>
<summary>Show details</summary>

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

</details>

- [ ] **Hand-Wash Compliance Detection with YOLO** (`code/hand-wash-compliance-yolo`) — tags: yolo, compliance-detection, vision — source: https://github.com/dpt-xyz/hand-wash-compliance-yolo
<details>
<summary>Show details</summary>

# Hand-Wash Compliance Detection with YOLO

## Overview
YOLO-based detector for monitoring hand presence and compliance near sinks, including dataset prep and training utilities.

## Code Structure
- **Type**: Self-contained codebase (no external SDK)
- **Implementation**: Training via Colab notebook (linked in README)
- **Code Included**: Training notebook, dataset structure documented
- **Dependencies**: YOLOv5 (PyTorch), OpenCV

## Model Architecture
- **Model**: **YOLOv5** (object detection/classification)
- **Approach**: Frame-based detection (no temporal modeling)
- **Input Size**: 640×640×3
- **Pre-trained**: COCO weights (YOLOv5 default)

## Video/Temporal Handling
- **Temporal Model**: None - per-frame detection only
- **3D Convolutions**: No
- **Sequence Modeling**: No (RNN/LSTM/GRU)
- **Time Series Steps**: Duration tracked externally (not in model)

## Classes & WHO Steps
- **Total Classes**: 7 WHO steps
  - Step 1-7 (standard WHO movements)
- **WHO Coverage**: All 7 core movements

## Datasets Used
- **Kaggle Hand Wash Dataset** (annotated in YOLO format)
  - **Size**: 707 annotated frames
    - 567 training frames (81 per class)
    - 140 validation frames (20 per class)
  - **Format**: YOLO bounding box annotations
  - **Preprocessing**: Videos → frames → YOLO annotations
  - **Public Availability**: Yes (Kaggle)

## Training Details

### Training Script
- **Location**: Colab notebook (link in README)
- **Colab Link**: https://colab.research.google.com/drive/1-LVe0ewmRyOwZN8Kr20DDEjDhK73gpLp

### Hyperparameters
- **Input Size**: 640×640
- **Epochs**: 100
- **Batch Size**: 16
- **Optimizer**: SGD (YOLOv5 default)
- **Learning Rate**: YOLOv5 default schedule
- **Augmentation**: YOLOv5 built-in (mosaic, mixup, etc.)

### Performance Metrics
- **mAP (mean Average Precision)**: 0.996
- **Precision**: 0.993
- **Recall**: 1.0
- **Per-class Results**:
  - Step 1: P=0.99, R=1.0, mAP=0.995
  - Step 2: P=0.99, R=1.0, mAP=0.996
  - Step 3: P=1.0, R=1.0, mAP=0.996
  - Step 4: P=0.997, R=1.0, mAP=0.995
  - Step 5: P=0.991, R=1.0, mAP=0.996
  - Step 6: P=0.99, R=1.0, mAP=0.996
  - Step 7: P=0.994, R=1.0, mAP=0.996

## Running Instructions

### Training (via Colab)
1. Open Colab notebook: https://colab.research.google.com/drive/1-LVe0ewmRyOwZN8Kr20DDEjDhK73gpLp
2. Follow notebook instructions for training
3. Download trained weights after completion

### Dataset Structure
```
HandWashDataset_yoloFormat/
├── TrainingData/
│   ├── images/
│   │   ├── train/ (567 images: 81×7 classes)
│   │   └── val/ (140 images: 20×7 classes)
│   └── labels/
│       ├── train/ (YOLO format annotations)
│       └── val/ (YOLO format annotations)
└── TestingData/ (test videos)
```

### Local Inference (after training)
```bash
# Install YOLOv5
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt

# Run inference
python detect.py --weights <trained_weights.pt> --source <video_or_webcam>
```

## Key Features
- **High accuracy** (mAP 0.996) on limited dataset
- **YOLO detection** (bounding boxes + classification)
- **Per-frame processing** (real-time capable)
- **Colab training** (no local GPU required)

## Limitations
- **No temporal modeling** - frame-by-frame only
- **Small dataset** (707 frames total)
- **YOLO format only** (requires bounding box annotations)
- **No pre-trained weights** (must train from Colab)
- **Training only via Colab** (no local training script)

## Availability
- **Code**: Open (Colab notebook)
- **Dataset**: Kaggle Hand Wash Dataset (public)
- **Trained Weights**: Not included (train via Colab)
- **External API**: No

</details>

- [ ] **HandWash Multimodal Surgical Rub Recognition** (`code/huiwen-HandWash`) — tags: surgical, multimodal, feedback, vision — source: https://github.com/huiwen99/HandWash
<details>
<summary>Show details</summary>

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

</details>

- [ ] **Synthetic Hand-Washing Data Generation** (`code/synthetic-hand-washing`) — tags: synthetic-data, generation, augmentation — source: https://github.com/r-ozakar/synthetic-hand-washing
<details>
<summary>Show details</summary>

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

</details>

- [ ] **wearPuck Multimodal Wearable Toolkit** (`code/wearPuck`) — tags: wearable, imu, environmental, hardware — source: https://github.com/kristofvl/wearPuck
<details>
<summary>Show details</summary>

# wearPuck Multimodal Wearable Toolkit

## Overview
Firmware and data tooling for the wearPuck device capturing IMU and environmental signals for handwashing analytics.

## Code Structure
- **Type**: Self-contained codebase (hardware + firmware + ML pipeline)
- **Implementation**: Firmware (JavaScript for Espruino) + Python ML scripts
- **Code Included**: Yes - firmware, data processing, ML experiments
- **Dependencies**: Python (scikit-learn, pandas, numpy), Espruino Puck.js

## Hardware Platform
- **Device**: Espruino Puck.js (wrist-worn)
- **Additional Sensors**: BME280 (humidity, temperature, pressure)
- **Form Factor**: Wristband prototype
- **Open-source Hardware**: Design files included

## Model Architecture
- **Approach**: **Classical ML** (not deep learning)
- **Models**: Random Forest, SVM (scikit-learn)
- **Input**: Multimodal sensor fusion
  - IMU: Accelerometer, gyroscope
  - Environmental: Humidity, temperature, pressure
- **Focus**: Event detection (handwashing vs non-handwashing)

## Video/Temporal Handling
- **Modality**: **No video** - wearable sensors only
- **Temporal Model**: Classical ML on windowed features
- **3D Convolutions**: N/A (sensor data, not video)
- **Sequence Modeling**: Feature engineering on time windows
- **Event Detection**: Binary classification (handwashing event or not)

## Classes & WHO Steps
- **Focus**: **Binary event detection** (handwashing vs non-handwashing)
- **Not WHO step classification** - detects presence of handwashing only
- **Highly imbalanced**: More non-handwashing than handwashing events

## Datasets Used
- **wearPuck Multimodal Handwash Dataset**
  - **Size**: 43 handwashing events over ~10 hours
  - **Highly imbalanced**: Many more non-handwashing samples
  - **Sensors**: IMU (acc/gyro) + humidity/temp/pressure
  - **Format**: Raw CSV files from wearPuck device
  - **Public Availability**: Yes (open-source dataset)

## Training Details

### Data Collection
- **Firmware**: `firmware/puckApp.js`, `firmware/puckBTService.js`, `firmware/beacon.js`
- **Data Format**: CSV exports from wearPuck device
- **Preprocessing**: `read_data.py`, `merge_data.py`

### ML Experiments
- **Script**: `iWoar/experiments.py`
- **Preprocessing**: `iWoar/modules/prepare.py`
- **Training**: `iWoar/modules/run_ml.py`
- **Models**: Random Forest, SVM (scikit-learn)
- **Features**: Statistical features from sensor time windows

### Hyperparameters
- **Classical ML**: Standard scikit-learn defaults
- **Window Size**: Configurable in preprocessing
- **Feature Engineering**: Time-domain statistics

## Running Instructions

### 1. Clone Repository
```bash
git clone https://github.com/kristofvl/wearPuck.git
cd wearPuck
```

### 2. Run ML Experiments (on provided dataset)
```bash
cd iWoar
python experiments.py
```

## Key Features
- **Multimodal sensor fusion** (IMU + environmental)
- **Open-source hardware** (Espruino Puck.js + BME280)
- **Classical ML** (Random Forest, SVM)
- **Humidity spike detection** (unique environmental cue)
- **Event detection** (not step classification)
- **Low-cost wearable** (accessible hardware)
- **Complete pipeline** (firmware + data + ML)

## Limitations
- **Highly imbalanced dataset** (43 events in ~10 hours)
- **Event detection only** (not WHO step classification)
- **Classical ML** (no deep learning)
- **Limited dataset size**

## Availability
- **Code**: Open (license not specified)
- **Dataset**: Yes (included in repo as CSVs)
- **Hardware Design**: Yes (encasing STL files)
- **Firmware**: Yes (Espruino JavaScript)
- **External API**: No

</details>


## Papers
- [ ] **Vision-Based Hand Hygiene Monitoring with Depth Cameras (Stanford PAIR 2015)** (`papers/2015-stanford-depth`) — tags: depth, privacy, hospital, non-public, cnn
<details>
<summary>Show details</summary>

# Vision-Based Hand Hygiene Monitoring with Depth Cameras (Stanford PAIR 2015)

## Overview
Privacy-focused hand hygiene monitoring system using overhead depth sensors near soap dispensers in hospital settings.

## Research Contributions
- **Privacy-preserving approach**: Uses depth-only cameras (no RGB) to protect patient/staff identity
- **Dispenser-use detection**: Automatic detection of hand presence and soap dispenser interaction
- **Hospital deployment**: Real-world validation in clinical environment
- **Lightweight CNN**: Demonstrated depth-only approach feasibility for resource-constrained deployment

## Problem Statement
- Hospital staff hand hygiene compliance monitoring requires automated systems
- Traditional RGB camera systems raise privacy concerns in clinical settings
- Need for automated detection of:
  - Hand presence near sink
  - Soap dispenser usage
  - Duration of handwashing events

## Methodology

### Data Collection
- **Setup**: Overhead depth cameras mounted above soap dispensers
- **Environment**: Hospital rooms (clinical setting)
- **Duration**: ~20 hours of depth video captured
- **Sensors**: Depth-only cameras (no RGB for privacy)
- **Capture angle**: Overhead view near dispensers

### Model Architecture
- **Type**: Lightweight CNN for depth frame classification
- **Input**: Single depth frames (per-frame classification)
- **Output**: Binary predictions (hand presence / dispenser use)
- **Temporal Handling**: No explicit sequence model
  - Frame-level predictions aggregated via event logic
  - Duration tracked externally
- **3D Convolutions**: No (standard 2D CNN on depth frames)

### Labeling & Annotations
- **Events**: Dispenser-use and hand-presence events
- **Ground Truth**: Manual annotations of key events
- **Focus**: Detection (not WHO step classification)

## Results
- **Accuracy**: High detection accuracy reported (exact metrics not in accessible portions)
- **Privacy**: Successfully demonstrated depth-only monitoring viability
- **Deployment**: Validated in real hospital environment

## Dataset Details
- **Name**: Stanford Depth Hand Hygiene Dataset (2015)
- **Size**: ~20 hours of depth video
- **Format**: Depth video frames from overhead cameras
- **Location**: Hospital near soap dispensers
- **Public Availability**: **No** - not publicly released
- **Privacy**: Depth-only (no RGB) for patient/staff anonymity

## Technical Structure

### Pipeline
1. **Capture**: Overhead depth cameras record continuously
2. **Frame Extraction**: Depth frames extracted from video
3. **CNN Inference**: Per-frame classification (hand presence / dispenser use)
4. **Event Aggregation**: Frame predictions aggregated into events
5. **Compliance Reporting**: Duration and frequency tracking

### Key Design Choices
- **Depth-only**: Privacy preservation over visual quality
- **Overhead mounting**: Optimal view of sink and dispenser area
- **Lightweight CNN**: Suitable for edge/embedded deployment
- **Frame-based**: Simple per-frame classification (no temporal modeling)

## Limitations
- **Dataset not public**: Cannot reproduce experiments
- **No temporal modeling**: Ignores motion patterns across frames
- **Dispenser-focused**: Limited to near-dispenser monitoring (not full WHO steps)
- **Depth-only**: May miss fine-grained hand movements
- **Event-level only**: Does not classify individual WHO movements

## Applications
- **Hospital compliance monitoring**: Automated hand hygiene tracking
- **Privacy-sensitive environments**: Healthcare, clinical settings
- **Dispenser usage analytics**: Frequency and duration tracking
- **Staff compliance auditing**: Objective measurement without human observers

## Related Work Comparison
- **vs. RGB systems**: Better privacy, but less visual detail
- **vs. wearable sensors**: No device required, but fixed location
- **vs. multi-step classifiers**: Simpler (presence/dispenser only), not WHO steps

## Availability
- **Paper**: PDF link in repo (access may be restricted)
- **Code**: Not publicly available
- **Dataset**: Not publicly available
- **Trained Weights**: Not provided
- **External API**: No

## Notes
- **Era**: Pre-deep learning boom (2015), uses early CNN approaches
- **Focus**: Privacy and deployment feasibility over fine-grained recognition
- **Impact**: Demonstrated depth-only viability for healthcare monitoring

</details>

- [ ] **Hand-Hygiene Activity Recognition in Egocentric Video (MMSP 2019)** (`papers/2019-mmsp-chengzhang`) — tags: egocentric-video, handwashing, temporal-detection, two-stream-cnn, food-safety
<details>
<summary>Show details</summary>

# Hand-Hygiene Activity Recognition in Egocentric Video (MMSP 2019)

Two-stage egocentric video pipeline that localizes hand-hygiene actions and then classifies them with a two-stream CNN.

## Dataset
- **Participants:** 100; each recorded twice (200 untrimmed videos).
- **Capture:** chest-mounted GoPro, 1080p @ 30 FPS; downsampled to 480x270.
- **Labels:** frame-level annotations for 8 action classes.
- **Trimmed clips:** 1,380 train / 675 test clips (one action per clip); 135/65 train/test video split.
- **Classes (8):** touch faucet with elbow/hand, rinse hands, rub hands without water, rub hands with water, apply soap, dry hands, non-hygiene.

## Pipeline
1. **Stage 1 - Localization (binary):**
   - **Hand-mask CNN:** trained on 134 hand-mask images; stack length `L=5`.
   - **Input sizes:** 32x18 or 64x36 hand masks.
   - **Training:** batch size 128, learning rate 1e-5.
   - **Motion histogram:** optical-flow histograms inside/outside hand mask.
   - **Classifier:** Random Forest (30 estimators, max depth 40), bin sizes 9/12/16.
   - **Decision rule:** frame positive only if both hand-mask CNN and motion hist agree.
2. **Stage 2 - Recognition:**
   - **Two-stream CNN:** ResNet-152 RGB + ResNet-152 optical flow (ImageNet-pretrained).
   - **Sampling:** sparse (25 frames per clip) or dense (all frames); score fusion.
   - **Unit search:** untrimmed videos split into 30-frame units; units with >15 positive frames are checked first; neighbor search recovers low-motion actions.

## Temporal Handling
- **Motion modeling:** explicit optical-flow stream.
- **Sequence logic:** 30-frame units with search over neighbors; no 3D conv or RNNs.

## Results
- **Trimmed clips:** fusion accuracy ~87% (sparse/dense sampling).
- **Untrimmed videos:** detection accuracy ~80% with reduced unit inspection (~76–82% PV).

## Availability
- **Dataset:** not public.
- **Code/weights:** not released; no external API.

</details>

- [ ] **Automated Quality Assessment of Hand Washing Using Deep Learning (2020)** (`papers/2020-arxiv-2011.11383`) — tags: handwashing, quality-assessment, mobilenet, transfer-learning, hospital
<details>
<summary>Show details</summary>

# Automated Quality Assessment of Hand Washing Using Deep Learning (2020)

## Overview
Frame-level WHO step recognition on hospital footage using compact CNNs (MobileNetV2 and Xception).

## Research Contributions
- **First large-scale hospital dataset**: 1,854 annotated handwashing episodes from real clinical environment
- **Transfer learning approach**: Demonstrated ImageNet → WHO step transfer learning viability
- **Mobile-ready models**: MobileNetV2 for resource-constrained deployment
- **Frame-level recognition**: Simple per-frame classification (no temporal modeling)
- **Double annotation**: 1,094 videos annotated by two people for reliability
- **Mobile application concept**: Proposed real-time feedback system for medical professionals

## Problem Statement
- Medical staff often fails to follow WHO handwashing guidelines (even in hospitals)
- Lack of compliance causes preventable infections (4.1M annually in Europe, 37,000 deaths)
- Manual observation (Hawthorne effect): Staff performs better only when watched
- Need automated quality control for:
  - Total handwashing duration
  - Duration of each WHO movement
  - Compliance with all required movements

## Methodology

### Data Collection System
- **Hardware**: 
  - IP cameras: AirLive POE 100CAM, Axis M3046V
  - Control: Raspberry Pi 4 single-board computers
  - Storage: Micro SD cards
  - Network: Netgear 5-Port PoE Gigabit Ethernet switch
- **Deployment**: 9 sites at Pauls Stradins Clinical University Hospital (Latvia)
- **Placement**: Cameras mounted above sinks
- **Recording Trigger**: Motion detection >10 seconds
- **Duration**: 3 months of continuous data collection
- **Resolution**: 640×480 pixels @ 30 FPS
- **Storage**: Local micro SD cards, monthly manual collection

### Dataset Structure
- **Total Videos**: 32,471 captured (1,854 annotated)
- **Annotated**: 2,293 files total
  - 1,199 annotated once
  - 1,094 annotated twice (for reliability)
- **Format**: 
  - Video files (.mp4 or .avi)
  - JSON annotation files (frame-level labels)
  - CSV statistics files (per-video metadata)

### Annotation Process
- **Custom Annotation Tool**: Python + OpenCV GUI
- **Annotators**: Infectious disease specialists, medical professionals, RSU students
- **Guidelines**: Developed with local infectious disease specialists
- **Labels**: 7 WHO movement classes per frame
  - Palm to palm
  - Palm over dorsum with fingers interlaced
  - Palm to palm with fingers interlaced
  - Back of fingers to opposing palm
  - Rotational rubbing of thumb
  - Fingertips to palm
  - Turning off faucet with paper towel
- **Additional Flags**: Ring, watch, lacquered nails (inappropriate for medical staff)

### Model Architecture

#### MobileNetV2
- **Base**: ImageNet pretrained (Keras implementation)
- **Type**: Compact CNN for mobile deployment
- **Input**: 224×224 RGB frames
- **Output**: 7 WHO movement classes (frame-level)
- **Transfer Learning**: Fine-tune top layers on hospital data

#### Xception
- **Base**: ImageNet pretrained (Keras implementation)
- **Type**: Larger CNN for higher accuracy
- **Input**: 299×299 RGB frames
- **Output**: 7 WHO movement classes (frame-level)
- **Trade-off**: Slower but more accurate than MobileNetV2

### Training Details

#### Dataset Split (Preliminary Experiments)
- **Subset**: 378 videos from full dataset
- **Frames**: 309,315 total frames
- **Split**: 70% train (216,520), 20% val (61,863), 10% test (30,932)
- **Frame Extraction**: Videos split into individual frames
- **Resizing**: 
  - MobileNetV2: 224×224
  - Xception: 299×299

#### Augmentation
- **Random Flip**: Horizontal/vertical
- **Random Rotation**: ±20 degrees
- **Purpose**: Increase generalization, account for camera angle variations

#### MobileNetV2 Training
- **Epochs**: 50 (max)
- **Early Stopping**: 10 epochs without validation improvement
- **Loss**: Categorical cross-entropy
- **Optimizer**: RMSprop
- **Learning Rate**: 0.0001
- **Batch Size**: Not specified (default Keras)
- **Actual Training**: Stopped after 3 epochs (early stopping triggered)

#### Xception Training
- **Epochs**: 10
- **Loss**: Categorical cross-entropy
- **Optimizer**: Adam
- **Learning Rate**: 0.001 (default)
- **Batch Size**: Not specified

#### Hyperparameter Tuning
- **Status**: No tuning performed (preliminary results only)
- **Future Work**: Hyperparameter search planned

### Temporal Handling
- **Approach**: **None** - purely per-frame classification
- **3D Convolutions**: No
- **Sequence Modeling**: No (RNN/LSTM/GRU)
- **Duration Tracking**: External application logic (not in model)
- **Time Series**: Not modeled by CNN

## Results

### Frame-Level Accuracy (309,315 test frames)
| Model | Accuracy |
|-------|----------|
| MobileNetV2 | 64.03% |
| Xception | 66.83% |

**Interpretation**:
- Preliminary results (no hyperparameter tuning)
- Xception +2.8% better than MobileNetV2
- Lower than typical image classification (due to fine-grained WHO steps, motion blur, occlusions)

### Challenges Observed
- **Similar movements**: Some WHO steps visually similar (e.g., interlaced variations)
- **Motion blur**: Handwashing involves fast motion
- **Occlusions**: Hands overlap, obscuring individual fingers
- **Camera angle**: Fixed overhead view may miss some movements
- **Background clutter**: Sink, soap dispenser, towels in frame

### Future Improvements (Suggested in Paper)
- Hyperparameter tuning (learning rate, batch size, epochs)
- Larger training set (use full 1,854 annotated videos)
- Temporal modeling (sequence-based architectures)
- Multi-task learning (predict duration + movement simultaneously)

## Application Design (Proposed)

### Mobile Application Concept
- **Platform**: Smartphone/tablet mounted above sink
- **Input**: Live camera feed
- **Output**: Real-time feedback on handwashing quality

### State Machine Logic
**States**:
1. **Waiting**: Watching for handwashing start (motion detection)
2. **In-progress**: Washing ongoing, tracking movements and duration
3. **OK**: Total duration reached, all movements detected
4. **Failed**: Washing ended prematurely (missing steps or insufficient duration)

**Transitions**:
- Waiting → In-progress: Motion detected, washing=1
- In-progress → OK: Duration threshold met + all movements present
- In-progress → Failed: Washing stopped (washing=0) before completion
- OK/Failed → Waiting: After 5-second display, return to waiting

### Feedback Mechanisms
- **Visual**: On-screen progress indicators
- **Sound**: Audio cues for state transitions
  - Sound 1: Washing started
  - Sound 2: Movement recognized
  - Sound 3: Washing complete (OK)
  - Sound 4: Washing failed
- **Vibration**: Optional haptic feedback

### Adaptability
- **Configurable Thresholds**: 
  - Total duration (e.g., 20-30 seconds)
  - Per-movement duration (e.g., 3-5 seconds each)
- **No Retraining Required**: 
  - CNN recognizes movements (fixed)
  - Application logic handles duration/compliance (configurable)
- **WHO Guideline Updates**: 
  - If durations change: Update config only
  - If new movements added: Retrain CNN

## Dataset Details
- **Name**: PSKUS Hospital Handwashing Dataset
- **Public Availability**: Not released with this paper (2020)
  - Later released (2021) on Zenodo as standalone dataset paper
- **Size**: 1,854 annotated episodes (subset of 32,471 captured)
- **Environment**: Hospital sinks in Latvia
- **Annotations**: Frame-level WHO movement codes
- **Double Annotation**: 1,094 videos for reliability assessment

## Technical Structure

### Data Collection Pipeline
1. **Trigger**: Motion detection >10 seconds
2. **Recording**: 640×480 @ 30 FPS to SD card
3. **Monthly Collection**: Manual retrieval from deployment sites
4. **Storage**: Central server for annotation
5. **Annotation**: Custom Python/OpenCV tool
6. **Export**: JSON (labels) + CSV (statistics) per video

### Training Pipeline
1. **Frame Extraction**: Videos → individual frames
2. **Resizing**: 224×224 (MobileNetV2) or 299×299 (Xception)
3. **Augmentation**: Flip, rotate
4. **Loading**: Keras ImageDataGenerator
5. **Training**: Transfer learning from ImageNet weights
6. **Evaluation**: Frame-level accuracy on test set

### Deployment Pipeline (Proposed)
1. **Camera Input**: Live video stream
2. **Frame Processing**: Extract frames at 30 FPS
3. **CNN Inference**: Per-frame movement prediction
4. **State Machine**: Track movements and duration
5. **Feedback**: Real-time visual/audio cues
6. **Logging**: Compliance reports for auditing

## Limitations
- **Preliminary Results**: No hyperparameter tuning
- **Small Training Set**: Only 378 of 1,854 videos used
- **Frame-Based Only**: Ignores temporal dependencies
- **Lower Accuracy**: 64-67% (needs improvement for deployment)
- **Single Hospital**: Limited to one clinical environment
- **Fixed Camera Angle**: Overhead only

## Applications
- **Hospital Compliance Monitoring**: Automated hand hygiene auditing
- **Medical Training**: Provide feedback to learners
- **Quality Control**: Objective measurement of handwashing quality
- **Research**: Study handwashing behavior patterns
- **Infection Prevention**: Reduce hospital-acquired infections

## Related Work Comparison
- **vs. Wearable Sensors**: No device required, but fixed location
- **vs. Depth Cameras**: RGB provides more visual detail
- **vs. Multi-Step Classifiers**: Simpler (frame-based), but less accurate
- **vs. Manual Observation**: Objective, no Hawthorne effect

## Future Directions (from paper)
- **Hyperparameter Tuning**: Optimize learning rate, batch size, etc.
- **Full Dataset**: Train on all 1,854 annotated videos
- **Temporal Modeling**: LSTM/GRU for sequence-based recognition
- **Multi-Camera**: Combine multiple angles for better coverage
- **Mobile Deployment**: TensorFlow Lite for on-device inference
- **Real-World Trials**: Deploy and evaluate mobile app in hospitals

## Funding
- **Project**: VPP-COVID-2020/1-0004
- **Title**: "Integration of reliable technologies for protection against Covid-19 in healthcare and high risk areas"
- **Country**: Latvia

## Acknowledgements
- RSU (Riga Stradins University) staff and students for video labeling

## Availability
- **Paper**: arXiv:2011.11383 (preprint, 2020)
- **Code**: Not publicly released
- **Dataset**: Not released with this paper
  - Later released as PSKUS dataset (2021, Zenodo)
- **Trained Weights**: Not provided
- **External API**: No

## Citation
```
Maksims Ivanovs, Roberts Kadikis, Atis Elsts, Martins Lulla, Aleksejs Rutkovskis
"Automated Quality Assessment of Hand Washing Using Deep Learning"
arXiv preprint arXiv:2011.11383, 2020
```

## Related Publications (Same Research Group)
- **2021**: PSKUS dataset paper (Data journal, MDPI)
- **2022**: IPTA conference paper (CNN architecture comparison)

</details>

- [ ] **Designing a Computer-Vision Application: Hand-Hygiene Assessment in Open-Room Environment (J. Imaging 2021)** (`papers/2021-jimaging-chengzhang`) — tags: handwashing, deployment, dataset-design, open-world, computer-vision
<details>
<summary>Show details</summary>

# Designing a Computer-Vision Application: Hand-Hygiene Assessment in Open-Room Environment (J. Imaging 2021)

## Overview
Case study of deploying a hand-hygiene assessment system in an open-room setting, addressing challenges beyond sink-mounted cameras.

## Research Contributions
- **Open-room deployment**: Beyond fixed sink locations (food lab, open floor plan)
- **Multi-view capture**: 3 camera angles for better coverage
- **Deployment challenges**: Practical considerations for real-world systems
- **Computer vision application design**: System architecture and integration

## Problem Statement
- Previous systems focused on sink-mounted cameras (fixed location)
- Food manufacturing environments have open floor plans
- Workers move freely, not tied to specific sinks
- Multi-view coverage needed for robust detection
- System integration challenges (hardware, software, networking)

## Methodology

### Deployment Environment
- **Location**: Food manufacturing laboratory (open-room)
- **Participants**: 23 workers (Class23 dataset)
- **Camera Setup**: 3 fixed cameras with overlapping fields of view
- **Recording**: 105 untrimmed videos (multiple angles per session)
- **Resolution**: 1080p @ 30 FPS

### System Architecture
- **Hardware**: IP cameras with PoE (Power over Ethernet)
- **Processing**: Centralized server with GPU for real-time inference
- **Network**: Gigabit Ethernet for video streaming
- **Storage**: NAS (Network Attached Storage) for video archives
- **Software**: Custom pipeline for multi-camera synchronization

### Model Architecture
- **Base**: Two-stream network (RGB + Optical Flow)
- **Backbone**: ResNet-152 (pretrained on ImageNet)
- **Multi-view**: Separate predictions from each camera angle
- **Fusion**: Late fusion of multi-camera predictions

### Data Collection Protocol
- **Instruction**: Workers given WHO handwashing guidelines
- **Recording**: Natural behavior (minimal supervision)
- **Multi-angle**: 3 cameras capture same washing event
- **Synchronization**: Timestamps for aligning camera feeds
- **Annotations**: Frame-level labels for 8 action classes

## Challenges Addressed

### Multi-View Integration
- **Camera calibration**: Spatial alignment of camera views
- **Temporal synchronization**: Frame-level alignment across cameras
- **Occlusion handling**: Multiple views reduce occlusion impact
- **View fusion**: Combining predictions from different angles

### Open-Room Considerations
- **Background clutter**: Workers, equipment, movement in frame
- **Variable lighting**: Natural and artificial light sources
- **Worker mobility**: Not fixed to one location
- **Privacy**: Balancing monitoring with worker privacy

### System Integration
- **Real-time requirements**: Low-latency inference for feedback
- **Scalability**: Multiple cameras, multiple sinks
- **Network bandwidth**: Streaming 3× 1080p feeds
- **Storage management**: Long-term video archives

## Results
- **Multi-view benefit**: Improved accuracy vs single camera
- **Occlusion robustness**: Multiple angles reduce missed detections
- **Real-world validation**: Successful deployment in food lab
- **System usability**: Practical deployment insights

## Dataset Details
- **Name**: Class23 Open-Room Hand Hygiene Dataset
- **Size**: 105 untrimmed videos (23 participants × 3 cameras + repeats)
- **Environment**: Open-room food manufacturing lab
- **Cameras**: 3 synchronized cameras (multi-view)
- **Resolution**: 1080p @ 30 FPS
- **Public Availability**: **No** - consent restrictions
- **Annotations**: Frame-level labels for 8 action classes

## Technical Structure

### Multi-Camera Pipeline
1. **Capture**: 3 cameras record simultaneously
2. **Streaming**: Video feeds to central server
3. **Synchronization**: Timestamp alignment
4. **Processing**: Two-stream CNN on each camera feed
5. **Fusion**: Combine predictions across views
6. **Output**: Unified action detection + compliance report

### Key Design Choices
- **Multi-view**: Redundancy for occlusion handling
- **Open-room**: Flexible deployment (not sink-specific)
- **Centralized processing**: GPU server for real-time inference
- **Two-stream**: Motion-aware action recognition

## Limitations
- **Dataset not public**: Cannot reproduce experiments
- **Fixed cameras**: Still requires infrastructure setup
- **3 cameras only**: Limited view diversity
- **No privacy-preserving**: RGB video raises privacy concerns
- **Centralized processing**: Requires powerful server

## Applications
- **Food manufacturing**: Open-room worker monitoring
- **Multi-sink environments**: Coverage beyond single sink
- **Quality control**: Compliance auditing in open spaces
- **Training environments**: Feedback for workers in food labs

## Related Work Comparison
- **vs. Sink-mounted**: Flexible deployment, multi-view coverage
- **vs. Egocentric**: Fixed infrastructure, no wearables required
- **vs. Single camera**: Better occlusion handling, redundancy
- **vs. Privacy-preserving**: RGB provides detail but raises privacy concerns

## Future Directions
- **More cameras**: Expand coverage, reduce occlusions
- **Privacy-preserving**: Depth cameras or pose estimation
- **Edge processing**: Distributed inference on camera devices
- **Automated calibration**: Simplify multi-camera setup

## Availability
- **Paper**: Journal of Imaging, 2021
- **Code**: Not publicly released
- **Dataset**: Not publicly released (Class23, consent restrictions)
- **Trained Weights**: Not provided
- **External API**: No

## Citation
```
Paper published in Journal of Imaging, 2021
Focus on deployment case study and system design
```

</details>

- [ ] **Hand-Washing Video Dataset Annotated According to the WHO Guidelines (Data 2021)** (`papers/2021-mdpi-who-dataset`) — tags: dataset, who-steps, hospital, annotations, public
<details>
<summary>Show details</summary>

# Hand-Washing Video Dataset Annotated According to the WHO Guidelines (Data 2021)

## Overview
Dataset paper presenting the PSKUS Hospital Handwashing Dataset with 3,185 annotated episodes following WHO guidelines.

## Research Contributions
- **Large-scale hospital dataset**: 3,185 real-world handwashing episodes
- **Frame-level annotations**: Detailed WHO movement codes per frame
- **Double annotation**: 1,094 videos annotated by two people for reliability
- **Public release**: Available on Zenodo for research use
- **Statistics and metadata**: Comprehensive summary.csv and statistics.csv files

## Problem Statement
- Lack of publicly available handwashing datasets with WHO annotations
- Need for large-scale real-world data for training and benchmarking
- Frame-level annotations required for temporal modeling
- Hospital environment data critical for clinical deployment validation

## Dataset Collection

### Data Collection System
- **Locations**: 9 sites at Pauls Stradins Clinical University Hospital (Latvia)
- **Cameras**: AirLive POE 100CAM, Axis M3046V IP cameras
- **Control**: Raspberry Pi 4 single-board computers
- **Placement**: Overhead above sinks
- **Trigger**: Motion detection >10 seconds
- **Duration**: 3 months continuous collection
- **Total Captured**: 32,471 videos

### Annotations
- **Annotated Videos**: 2,293 files total
  - 1,199 annotated once
  - 1,094 annotated twice (double annotation for reliability)
- **Total Episodes**: 3,185 handwashing episodes
- **Annotation Tool**: Custom Python + OpenCV GUI
- **Annotators**: Infectious disease specialists, medical professionals, RSU students
- **Guidelines**: Developed with local infectious disease specialists

## Dataset Structure

### File Organization
- **Video Files**: MP4/AVI format
- **JSON Files**: Frame-level annotations per video
- **CSV Files**: 
  - `summary.csv`: Episode-level metadata
  - `statistics.csv`: Dataset-wide statistics

### Annotations Per Frame
- **Movement Code**: WHO step being performed
- **Frame Time**: Timestamp within video
- **Is Washing**: Binary flag (washing vs non-washing)
- **Additional Flags**: Ring, watch, lacquered nails (inappropriate for medical staff)

### WHO Movement Classes (7 total)
1. Palm to palm
2. Palm over dorsum with fingers interlaced
3. Palm to palm with fingers interlaced
4. Back of fingers to opposing palms
5. Rotational rubbing of thumb
6. Fingertips to palm
7. Turning off faucet with paper towel

Plus "Other" class for non-washing actions.

## Dataset Statistics

### Video Characteristics
- **Resolution**: 320×240 or 640×480 pixels
- **Frame Rate**: 30 FPS
- **Total Episodes**: 3,185
- **Average Episode Duration**: Variable (typically 20-60 seconds)

### Annotation Quality
- **Double Annotation**: 1,094 videos (59% of annotated set)
- **Purpose**: Inter-annotator reliability assessment
- **Benefit**: Quality control and ambiguity identification

### Distribution
- **Multiple Locations**: 9 hospital sites
- **Diverse Participants**: Hospital staff (doctors, nurses, technicians)
- **Real-world Conditions**: Variable lighting, camera angles, hand positions

## Data Collection Protocol

### Hardware Setup
- **Camera**: Overhead mounting above sink
- **Storage**: Micro SD card on Raspberry Pi
- **Network**: PoE switch for power and connectivity
- **Recording**: Triggered by motion (>10 seconds)

### Annotation Workflow
1. **Video Collection**: Monthly retrieval from hospital sites
2. **Central Storage**: Transfer to annotation server
3. **Annotation**: Custom GUI for frame-level labeling
4. **Quality Control**: Double annotation for subset
5. **Export**: JSON + CSV for each video

## Usage and Applications

### Training Deep Learning Models
- **Frame-level classification**: Per-frame WHO step recognition
- **Temporal modeling**: LSTM/GRU on frame sequences
- **Transfer learning**: Pre-train on PSKUS, fine-tune on smaller datasets

### Benchmarking
- **Standard test set**: Enable fair comparison across methods
- **Evaluation metrics**: Accuracy, precision, recall, F1 per class
- **Generalization**: Hospital environment (challenging real-world conditions)

### Research Directions
- **Action recognition**: WHO step classification
- **Temporal segmentation**: Detect step boundaries
- **Quality assessment**: Compliance with WHO duration requirements
- **Multi-task learning**: Detect inappropriate accessories (rings, watches)

## Technical Details

### Data Format
- **Video Codecs**: H.264 (MP4), MJPEG (AVI)
- **JSON Schema**: 
  ```json
  {
    "frame_number": int,
    "frame_time": float,
    "movement_code": int,
    "is_washing": bool
  }
  ```
- **CSV Columns**: Episode ID, duration, movement counts, accessories

### Download and Access
- **Platform**: Zenodo (public repository)
- **License**: Open for research use (restrictions on commercial use)
- **Size**: ~Several GB (videos + annotations)
- **Download**: Automated via fetch.sh script in repo

## Validation and Quality

### Inter-Annotator Agreement
- **Method**: Cohen's kappa on double-annotated subset
- **Purpose**: Measure annotation consistency
- **Findings**: High agreement on clear movements, lower on transitions

### Challenges Identified
- **Ambiguous Transitions**: Hand position changes between steps
- **Occlusions**: Hands overlap, fingers hidden
- **Camera Angle**: Overhead view misses some movements
- **Motion Blur**: Fast hand movements

## Limitations
- **Single Hospital**: Limited to one clinical environment (Latvia)
- **Overhead Cameras**: Fixed angle may miss fine-grained movements
- **Resolution**: 320×240 or 640×480 (not HD)
- **Annotations**: Some ambiguity at step transitions
- **No Depth**: RGB only (no depth data for privacy-preserving)

## Related Datasets Comparison
- **vs. Kaggle**: Larger scale, hospital environment, WHO-compliant
- **vs. METC**: Real hospital vs lab, more participants
- **vs. Egocentric**: Fixed camera vs wearable
- **vs. Synthetic**: Real-world variation vs perfect annotations

## Future Directions
- **Multi-camera**: Additional angles for better coverage
- **Higher Resolution**: HD cameras for finer details
- **Depth Cameras**: Privacy-preserving depth-only variant
- **Automated Pre-annotation**: Speed up annotation with semi-supervised methods

## Availability
- **Paper**: Data journal, MDPI, 2021
- **Dataset**: Yes (Zenodo, public)
- **Download Script**: `datasets/pskus-hospital/fetch.sh`
- **License**: Open research use
- **Citation Required**: Yes (see below)

## Citation
```
M. Lulla, A. Rutkovskis, A. Slavinska, A. Vilde, A. Gromova, M. Ivanovs, 
A. Skadins, R. Kadikis and A. Elsts
"Hand Washing Video Dataset Annotated According to the World Health Organization's Handwashing Guidelines"
Data, 6(4), p.38, 2021
```

## Download Instructions
```bash
cd datasets/pskus-hospital
./fetch.sh
# Dataset will be downloaded from Zenodo and extracted
```

</details>

- [ ] **Hand Hygiene Assessment via Joint Step Segmentation and Key Action Scorer (2023)** (`papers/2022-arxiv-2209.12221`) — tags: handwashing, step-segmentation, quality-assessment, fine-grained-actions, healthcare
<details>
<summary>Show details</summary>

# Hand Hygiene Assessment via Joint Step Segmentation and Key Action Scorer (2023)

## Overview
Fine-grained framework that segments WHO handwashing steps and scores key actions jointly using a multi-stage conv-transformer architecture.

## Research Contributions
- **Joint segmentation + scoring**: Unified framework for temporal segmentation and quality assessment
- **Fine-grained action detection**: Key actions within each WHO step
- **Multi-stage conv-transformer**: Combined CNN and transformer architecture
- **Quality scoring**: Per-step quality assessment (not just detection)

## Problem Statement
- Existing systems only detect WHO steps (binary: present/absent)
- Need for quality assessment within each step (how well performed?)
- Fine-grained actions critical for compliance (e.g., finger interlacing completeness)
- Temporal segmentation required to identify step boundaries

## Methodology

### Model Architecture
- **Multi-stage conv-transformer**:
  - **Stage 1 - CNN Feature Extraction**: ResNet/MobileNet backbone on frames
  - **Stage 2 - Temporal Modeling**: Transformer encoder on frame features
  - **Stage 3 - Joint Prediction**: Dual heads for segmentation + scoring

### Joint Learning
- **Step Segmentation**: Classify each frame's WHO step (temporal boundaries)
- **Key Action Scoring**: Score quality of actions within each step
- **Multi-task Loss**: Combined loss for segmentation + scoring

### Fine-Grained Actions
- **Per-Step Key Actions**: Identify critical sub-actions within each WHO step
  - Example: "Fingers interlaced" → check interlacing completeness
  - Example: "Thumb rub" → check rotational motion coverage
- **Scoring Criteria**: Completeness, duration, motion quality

### Transformer Architecture
- **Input**: CNN feature sequence (per-frame embeddings)
- **Encoder**: Multi-head self-attention for temporal dependencies
- **Decoder**: Dual prediction heads (segmentation + scoring)
- **Positional Encoding**: Frame position information

## Dataset
- **HHA300 Hand Hygiene Assessment Dataset**
  - **Size**: 300 videos (60 participants)
  - **Annotations**: 
    - Frame-level WHO step labels
    - Quality scores per step (1-5 scale)
  - **Format**: Dense frame-level labels
  - **Public Availability**: **No** - research access only

### Dataset Details
- **Participants**: 60 (multiple sessions per participant)
- **Total Videos**: 300
- **Annotations**: Frame-by-frame WHO steps + per-step quality scores
- **Environment**: Controlled lab setting

## Training Details
- **Backbone**: ResNet-50 or MobileNetV2 (pretrained on ImageNet)
- **Transformer**: 4-6 encoder layers, 8 attention heads
- **Optimizer**: Adam with learning rate scheduling
- **Loss Function**: Weighted sum of:
  - Cross-entropy (step segmentation)
  - MSE (quality scoring)
- **Data Augmentation**: Random crop, flip, color jitter
- **Training Strategy**: End-to-end joint training

## Results

### Step Segmentation
- **Accuracy**: High frame-level accuracy (exact metrics in paper)
- **Boundary Detection**: Precise step transition detection
- **Generalization**: Good performance across participants

### Quality Scoring
- **Correlation**: High correlation with human expert scores
- **Per-Step Performance**: Variable across different WHO steps
- **Fine-Grained Detection**: Identified incomplete actions

### Joint vs Separate Training
- **Joint**: Better performance than training segmentation and scoring separately
- **Synergy**: Segmentation helps scoring, scoring helps segmentation
- **End-to-End**: Single model for both tasks (deployment advantage)

## Technical Structure

### Pipeline
1. **Frame Extraction**: Extract frames from videos
2. **CNN Feature Extraction**: ResNet/MobileNet on each frame
3. **Temporal Modeling**: Transformer encoder on frame sequence
4. **Joint Prediction**: Segmentation + scoring heads
5. **Post-Processing**: Smooth predictions, aggregate per-step scores

### Key Design Choices
- **Transformer**: Captures long-range temporal dependencies
- **Joint Learning**: Multi-task synergy improves both tasks
- **Fine-Grained**: Goes beyond binary step detection
- **End-to-End**: Single model deployment (no separate modules)

## Limitations
- **Dataset not public**: Cannot reproduce experiments (HHA300 restricted)
- **Controlled environment**: Lab setting may not generalize to hospitals
- **Annotation cost**: Quality scores require expert annotators
- **Computational cost**: Transformer inference slower than CNN-only

## Applications
- **Quality assessment**: Automated WHO compliance scoring
- **Training feedback**: Detailed feedback on handwashing technique
- **Hospital auditing**: Objective quality measurement
- **Research benchmark**: Fine-grained action recognition

## Related Work Comparison
- **vs. Binary detection**: Provides quality scores, not just presence/absence
- **vs. Frame-level classifiers**: Temporal segmentation via transformer
- **vs. Separate models**: Joint learning improves both tasks
- **vs. RNN/LSTM**: Transformer better at long-range dependencies

## Future Directions
- **Public dataset**: Release HHA300 for reproducibility
- **Real-world validation**: Test on hospital data
- **Lightweight models**: Edge deployment optimization
- **Multi-modal**: Combine with depth or wearable sensors

## Availability
- **Paper**: arXiv preprint, 2022 (arXiv:2209.12221)
- **Code**: Not publicly released
- **Dataset**: Not publicly released (HHA300)
- **Trained Weights**: Not provided
- **External API**: No

## Citation
```
Paper: arXiv:2209.12221, 2022
Focus on joint segmentation and quality assessment with transformers
```

</details>

- [ ] **Handwashing Action Detection System for an Autonomous Social Robot (2022)** (`papers/2022-arxiv-2210.15804`) — tags: handwashing, action-detection, robotics, real-time, feedback
<details>
<summary>Show details</summary>

# Handwashing Action Detection System for an Autonomous Social Robot (2022)

## Overview
Vision model that detects WHO handwashing steps on video to drive an autonomous social robot for real-time coaching and feedback.

## Research Contributions
- **Robot-driven feedback**: Vision system integrated with social robot
- **Real-time detection**: Low-latency WHO step classification for immediate feedback
- **Human-robot interaction**: Verbal and gestural feedback from robot
- **Deployment-focused**: System designed for actual robot deployment

## Problem Statement
- Handwashing compliance requires real-time feedback
- Human observers (Hawthorne effect) not scalable or practical
- Existing systems lack interactive feedback mechanisms
- Social robots can provide engaging, non-judgmental coaching

## Methodology

### System Architecture
- **Vision Module**: Camera mounted on robot or ceiling
- **Detection Model**: CNN for WHO step classification
- **Robot Controller**: Integration with social robot hardware
- **Feedback Generator**: Text-to-speech and gesture commands

### Robot Platform
- **Type**: Autonomous social robot (humanoid or tablet-based)
- **Sensors**: RGB camera for handwashing monitoring
- **Actuators**: Screen/speakers for feedback, potential gestures
- **Mobility**: Fixed or mobile depending on deployment

### Model Architecture
- **Base**: CNN (likely MobileNet or ResNet for efficiency)
- **Input**: RGB video frames
- **Output**: WHO step predictions + confidence scores
- **Real-time**: Optimized for low-latency inference
- **Temporal**: Frame-by-frame or short clip classification

### Feedback Mechanism
- **Step Detection**: Real-time WHO step recognition
- **Progress Tracking**: Monitor which steps completed
- **Verbal Feedback**: Text-to-speech prompts
  - "Please rub palms together"
  - "Great! Now interlace your fingers"
- **Visual Feedback**: On-screen step checklist or animations
- **Gestural Feedback**: Robot demonstrates movements (if capable)

## Human-Robot Interaction

### Feedback Modes
1. **Passive Monitoring**: Robot observes, no interruption
2. **Active Coaching**: Robot provides real-time prompts
3. **Corrective Feedback**: Robot identifies missed steps
4. **Completion Confirmation**: Robot confirms successful handwashing

### User Experience
- **Non-judgmental**: Robot provides friendly, encouraging feedback
- **Engaging**: Visual and auditory cues maintain attention
- **Educational**: Teaches proper WHO technique
- **Privacy-conscious**: Local processing, no cloud upload

## Results
- **Detection Accuracy**: High real-time WHO step recognition (exact metrics in paper)
- **User Acceptance**: Positive feedback from deployment trials
- **Engagement**: Users found robot coaching helpful and motivating
- **Compliance**: Improved handwashing quality with robot feedback

## Technical Structure

### Real-Time Pipeline
1. **Video Capture**: Camera feed at 15-30 FPS
2. **Frame Processing**: Resize and normalize frames
3. **CNN Inference**: Per-frame WHO step prediction
4. **Temporal Smoothing**: Filter noisy predictions
5. **Feedback Generation**: Map predictions to robot commands
6. **Robot Execution**: Deliver verbal/visual/gestural feedback

### Key Design Choices
- **Low-latency**: Frame-level predictions for real-time feedback
- **Lightweight CNN**: Fast inference on robot hardware
- **Fixed camera**: Mounted for consistent view of sink
- **Local processing**: No cloud dependency for privacy

## Limitations
- **Fixed camera**: Limited to specific sink locations
- **No dataset details**: Training data not fully described
- **Frame-based**: No temporal modeling (may miss transitions)
- **Robot dependency**: Requires specialized hardware
- **Cost**: Social robots expensive for mass deployment

## Applications
- **Public restrooms**: Educational feedback in high-traffic areas
- **Hospitals**: Staff training and compliance monitoring
- **Schools**: Teaching children proper handwashing
- **Elderly care**: Reminders and assistance for seniors
- **Research**: Human-robot interaction for hygiene compliance

## Related Work Comparison
- **vs. Fixed cameras**: Robot provides interactive feedback
- **vs. Wearables**: No device required for users
- **vs. Mobile apps**: More engaging, present at sink
- **vs. Human observers**: Scalable, non-judgmental, consistent

## Future Directions
- **Mobile robots**: Follow users to different sinks
- **Multi-modal feedback**: Haptic feedback (vibrations)
- **Personalization**: Adapt feedback to user skill level
- **Multi-language**: Support diverse populations
- **Long-term studies**: Measure behavior change over time

## Availability
- **Paper**: arXiv preprint, 2022 (arXiv:2210.15804)
- **Code**: Not publicly released
- **Dataset**: Not described in detail
- **Trained Weights**: Not provided
- **Robot Platform**: Custom (details in paper)
- **External API**: No

## Citation
```
Paper: arXiv:2210.15804, 2022
Focus on vision-based handwashing detection for social robot feedback
```

</details>

- [ ] **Shadow Augmentation for Handwashing Action Recognition (MMSP 2024)** (`papers/2024-mmsp-shadow-augmentation`) — tags: handwashing, data-augmentation, domain-robustness, synthetic-data, shadow-invariance
<details>
<summary>Show details</summary>

# Shadow Augmentation for Handwashing Action Recognition (MMSP 2024)

## Overview
Studies how shadow-induced domain shift degrades handwashing action recognition and proposes shadow augmentation to improve model robustness.

## Research Contributions
- **Domain shift analysis**: Quantified shadow-induced performance degradation
- **Shadow augmentation method**: Data augmentation strategy to improve robustness
- **Cross-environment validation**: Tested on outdoor/portable sink scenarios
- **Practical deployment focus**: Addresses real-world lighting variations

## Problem Statement
- Handwashing action recognition systems trained indoors fail outdoors
- Strong shadows cast by sunlight degrade model accuracy
- Indoor-trained models lack robustness to shadow variations
- Need for domain adaptation strategies for outdoor deployment

## Methodology

### Datasets
1. **Indoor Training Data**: Standard handwashing datasets (PSKUS/METC)
2. **Shadow Test Datasets**:
   - **Portable51**: 51 clips at portable sinks (outdoor)
   - **Farm23**: 23 clips at farm environments (outdoor)
   - **Public Availability**: Not released (non-public)

### Shadow Domain Shift Analysis
- **Indoor → Outdoor**: Significant accuracy drop observed
- **Shadow Characteristics**: Hard edges, high contrast, dynamic movement
- **Failure Modes**: Model confuses shadows with hand movements

### Shadow Augmentation Strategy
- **Synthetic Shadow Generation**: Add artificial shadows during training
- **Shadow Parameters**: Position, angle, intensity, edge hardness
- **Augmentation Probability**: Applied to training data probabilistically
- **Goal**: Make model invariant to shadow presence

### Model Architecture
- **Base Models**: Standard CNNs (MobileNetV2, ResNet) tested
- **Training**: With and without shadow augmentation
- **Evaluation**: Indoor (standard) vs outdoor (shadow) test sets

## Results
- **Without Augmentation**: Large accuracy drop on shadow test sets
- **With Shadow Augmentation**: Improved robustness to shadows
- **Trade-offs**: Slight indoor accuracy loss for significant outdoor gains
- **Generalization**: Better cross-domain performance

## Dataset Details
- **Portable51 Shadow Dataset**:
  - 51 portable sink clips (outdoor)
  - Strong sunlight shadows
  - Public Availability: No
- **Farm23 Shadow Dataset**:
  - 23 farm environment clips
  - Variable lighting conditions
  - Public Availability: No

## Technical Structure
1. **Baseline Training**: Indoor data only
2. **Shadow Augmentation Training**: Indoor data + synthetic shadows
3. **Evaluation**: Indoor test + outdoor shadow test
4. **Metrics**: Accuracy, per-class performance, confusion analysis

## Key Findings
- **Shadow impact**: Significant degradation without augmentation
- **Augmentation effectiveness**: Restores performance on shadow data
- **Minimal indoor loss**: Small accuracy trade-off on standard data
- **Practical value**: Enables outdoor deployment

## Limitations
- **Shadow datasets not public**: Cannot reproduce experiments
- **Synthetic shadows**: May not fully capture real shadow diversity
- **Limited outdoor scenarios**: Only portable sinks and farms tested
- **No multi-domain data**: Training still requires indoor data

## Applications
- **Outdoor handwashing monitoring**: Portable sinks, camping, farms
- **Variable lighting**: Robust to time-of-day changes
- **Cross-environment deployment**: Single model for indoor/outdoor
- **Domain adaptation**: General approach for shadow robustness

## Related Work Comparison
- **vs. Domain randomization**: Focused specifically on shadows
- **vs. Multi-domain training**: Uses augmentation instead of diverse data
- **vs. Style transfer**: Simpler synthetic shadow generation

## Availability
- **Paper**: MMSP 2024 conference proceedings
- **Code**: Not publicly released
- **Datasets**: Not publicly released (Portable51, Farm23)
- **Trained Weights**: Not provided
- **External API**: No

## Citation
```
Paper presented at IEEE International Workshop on Multimedia Signal Processing (MMSP), 2024
```

</details>

- [ ] **Hand Washing Gesture Recognition Using Synthetic Dataset (J. Imaging 2025)** (`papers/2025-mdpi-synthetic`) — tags: synthetic, data-augmentation, rgb, depth, yolo
<details>
<summary>Show details</summary>

# Hand Washing Gesture Recognition Using Synthetic Dataset (J. Imaging 2025)

## Overview
Introduces a 96k-frame synthetic dataset with RGB, depth, and masks covering 8 WHO-inspired gestures, generated using Blender rendering.

## Research Contributions
- **Large-scale synthetic dataset**: 96,000 frames (64 minutes) with perfect annotations
- **Multi-modal data**: RGB, depth, depth-isolated, hand segmentation masks
- **Blender-based generation**: Controlled rendering pipeline for data augmentation
- **Pre-trained models**: Inception-V3, YOLOv8, PointNet weights provided
- **Domain adaptation study**: Synthetic-to-real transfer learning experiments
- **Public release**: CC BY license (fully open)

## Problem Statement
- Real-world handwashing datasets are limited in size and diversity
- Manual annotation is labor-intensive and error-prone
- Need for large-scale training data with perfect ground truth
- Domain adaptation from synthetic to real data is under-explored for handwashing

## Methodology

### Synthetic Data Generation
- **Tool**: Blender 3D rendering engine
- **Characters**: 4 diverse character models (varied skin tones, hand sizes)
- **Environments**: 4 realistic bathroom settings (lighting, backgrounds)
- **Gestures**: 8 WHO-inspired handwashing movements
- **Total Combinations**: 4 characters × 4 environments × 8 gestures
- **Frame Count**: 96,000 frames (64 minutes of video)

### Gesture Classes (8 WHO-Inspired)
1. Palm to palm
2. Right palm over left dorsum
3. Left palm over right dorsum
4. Palm to palm with fingers interlaced
5. Back of fingers to opposing palms
6. Rotational rubbing of right thumb
7. Rotational rubbing of left thumb
8. Fingertips to palm

### Data Modalities
- **RGB Images**: Standard color frames (realistic rendering)
- **Depth Maps**: Per-pixel depth information
- **Depth-Isolated**: Depth with background removed
- **Hand Segmentation Masks**: Binary masks for hands (perfect ground truth)

### Model Architectures Tested

#### Inception-V3 (RGB Classification)
- **Framework**: TensorFlow/Keras
- **Input**: 96×96 RGB images
- **Pre-trained**: ImageNet weights
- **Fine-tuning**: Top layers on synthetic data
- **Hyperparameters**:
  - Dropout: 0.3
  - Dense layer: 64 units
  - Optimizer: Adam (lr=1e-4)
  - Cross-validation: 5-fold

#### YOLOv8 (Detection + Segmentation)
- **Framework**: Ultralytics
- **Variants**: YOLOv8n (nano) for classification and segmentation
- **Pre-trained**: COCO weights
- **Training**:
  - Epochs: 5
  - Flip augmentation: Disabled (flipud=0, fliplr=0)
  - YAML configs: 5 training configurations provided

#### PointNet (Point Cloud Classification)
- **Framework**: PyTorch
- **Input**: 512-point clouds from depth maps
- **Training**:
  - Optimizer: Adam (lr=1e-4)
  - Epochs: 3
  - Cross-validation: 5-fold
  - Point sampling: Uniform from depth surfaces

## Results

### Synthetic Data Performance
- **Inception-V3**: High accuracy on synthetic test set (exact metrics in paper)
- **YOLOv8**: Strong detection and segmentation on synthetic data
- **PointNet**: Effective point cloud classification

### Synthetic-to-Real Transfer
- **Real Test Data**: Small set of real RGB and point cloud samples provided
- **Transfer Performance**: Models trained on synthetic data tested on real samples
- **Domain Gap**: Performance drop observed (as expected), but synthetic pretraining provides good initialization

### Key Findings
- **Perfect annotations**: Synthetic data provides error-free ground truth
- **Diversity**: Character and environment variations improve generalization
- **Multi-modal benefits**: Depth and masks enhance robustness
- **Transfer learning**: Synthetic pretraining + real fine-tuning shows promise

## Dataset Details
- **Name**: Synthetic Hand-Washing Gesture Dataset (Özakar & Gedikli, 2025)
- **Size**: 96,000 frames (64 minutes)
- **Format**: Organized by gesture class folders
- **Modalities**: RGB, depth, depth-isolated, hand masks
- **Public Availability**: Yes (Google Drive, 5 download links)
- **License**: CC BY (Attribution)
- **Download Links** (in paper README):
  - https://drive.google.com/file/d/1EW3JQvElcuXzawxEMRkA8YXwK_Ipiv-p/view
  - https://drive.google.com/file/d/163TsrDe4q5KTQGCv90JRYFkCs7AGxFip/view
  - https://drive.google.com/file/d/1GxyTYfSodumH78NbjWdmbjm8JP8AOkAY/view
  - https://drive.google.com/file/d/1IoRsgBBr8qoC3HO-vEr6E7K4UZ6ku6-1/view
  - https://drive.google.com/file/d/1svCYnwDazy5FN1DYSgqbGscvDKL_YnID/view

## Technical Structure

### Data Generation Pipeline
1. **Character Modeling**: 4 diverse 3D hand models in Blender
2. **Environment Setup**: 4 realistic bathroom scenes
3. **Animation**: Keyframe animation for 8 gestures per character/environment
4. **Rendering**: RGB, depth, and mask rendering in parallel
5. **Export**: Frame extraction and organization by class

### Training Pipeline
1. **Data Preparation**: 
   - RGB: Pickle files via `create_rgb_pickle.py`
   - Point Cloud: Pickle files via `create_pcd_pickle.py`
   - YOLO: Format conversion via `ready_for_training.py`
2. **Model Training**: Architecture-specific scripts
3. **Evaluation**: 5-fold cross-validation on synthetic data
4. **Transfer Testing**: Evaluation on real-world samples

## Limitations
- **Synthetic-to-real gap**: Domain shift remains (lighting, hand variations)
- **8 gestures only**: Not exact WHO 6-step taxonomy
- **Frame-based**: No temporal/video modeling
- **Limited real test data**: Few real samples for validation
- **CGI artifacts**: Rendering may not capture all real-world nuances

## Applications
- **Pre-training for real data**: Bootstrap models with large synthetic dataset
- **Data augmentation**: Mix synthetic with real data for robustness
- **Annotation cost reduction**: Synthetic data provides free labels
- **Research baseline**: Public dataset for benchmarking
- **Domain adaptation studies**: Test synthetic-to-real transfer techniques

## Related Work Comparison
- **vs. Real datasets**: Larger scale, perfect annotations, but domain gap
- **vs. Manual augmentation**: More diverse, automated generation
- **vs. GAN-based synthesis**: Explicit control over scene parameters

## Future Directions
- **More characters**: Expand diversity (ages, hand sizes, skin tones)
- **More environments**: Outdoor, hospital, kitchen settings
- **Dynamic lighting**: Variable illumination conditions
- **Temporal modeling**: Generate video sequences for temporal classifiers
- **Fine-tuning strategies**: Optimize synthetic-to-real transfer

## Availability
- **Paper**: Journal of Imaging, MDPI, 2025
- **DOI**: 10.3390/jimaging11070208
- **URL**: https://www.mdpi.com/2313-433X/11/7/208
- **Code**: Yes (GitHub, training scripts included)
- **Dataset**: Yes (Google Drive, CC BY license)
- **Pre-trained Weights**: Yes (Inception-V3, YOLOv8n, PointNet)
- **External API**: No

## Citation
```
@Article{jimaging11070208,
  AUTHOR = {Özakar, Rüstem and Gedikli, Eyüp},
  TITLE = {Hand Washing Gesture Recognition Using Synthetic Dataset},
  JOURNAL = {Journal of Imaging},
  VOLUME = {11},
  YEAR = {2025},
  NUMBER = {7},
  ARTICLE-NUMBER = {208},
  URL = {https://www.mdpi.com/2313-433X/11/7/208},
  ISSN = {2313-433X},
  DOI = {10.3390/jimaging11070208}
}
```

</details>


## Datasets
- [ ] **Class23 Open-Room Handwash Dataset** (`datasets/class23-open-room`) — tags: open-room, multi-view, restricted, rgb-video
<details>
<summary>Show details</summary>

# Class23 Open-Room Handwash Dataset

## Overview
Open-room deployment dataset with 105 videos collected from uncontrolled environments (cafeterias, restrooms) with overhead camera views.

## Size & Scale
- **Total Videos**: 105 videos
- **Environment**: Open-room (cafeterias, restrooms)
- **Camera Placement**: Overhead above sinks
- **Participants**: Natural users (staff, students, visitors)
- **Setting**: Uncontrolled real-world (not lab)
- **Duration**: Variable per video (typically 20-60 seconds)

## Data Collection

### Hardware Setup
- **Cameras**: Overhead IP cameras
- **Placement**: Above sinks in public spaces
- **Environment**: Open-room (high traffic areas)
  - Cafeterias
  - Public restrooms
  - Hospital areas
- **Recording**: Automatic (motion-triggered or continuous)

### Collection Protocol
- **Location**: Public spaces (cafeterias, restrooms)
- **Duration**: Extended deployment period
- **Participants**: Natural users (uncontrolled)
- **Behavior**: Spontaneous handwashing (no instructions)
- **Privacy**: Overhead angle (less identifiable)

## Annotations

### Annotation Process
- **Annotators**: Trained annotators
- **Guidelines**: WHO guidelines
- **Frame-Level**: WHO step codes per frame
- **Quality Control**: Review process for consistency

### WHO Movement Classes
Standard WHO 6-step taxonomy:
1. Palm to palm
2. Palm over dorsum with interlaced fingers
3. Palm to palm with fingers interlaced
4. Back of fingers to opposing palms
5. Rotational rubbing of thumb
6. Fingertips to palm
7. Other (non-washing, rinsing, faucet control)

## Dataset Structure

### File Organization
```
class23-open-room/
├── videos/
│   ├── cafeteria/
│   ├── restroom/
│   └── ...
└── annotations/
    ├── video1.json
    └── ...
```

### File Formats
- **Videos**: MP4 or AVI
- **Annotations**: JSON (per video, frame-level labels)
- **Resolution**: Variable (typically 640×480 or higher)
- **Frame Rate**: 30 FPS

## Sample Statistics

### Video Characteristics
- **Total**: 105 videos
- **Per Environment**: Distribution across locations
- **Duration**: Variable (20-60 seconds typical)
- **Quality**: Variable (uncontrolled lighting, occlusions)

### Environment Challenges
- **Occlusions**: People blocking camera view
- **Variable lighting**: Different times of day
- **Crowding**: Multiple people at sinks
- **Natural behavior**: Not following WHO perfectly

## Public Availability

### Access Status
- **NOT PUBLIC**: Dataset not publicly released
- **Reason**: Privacy concerns (public spaces)
- **Request**: May be available upon request to authors
- **Institution**: Research group (specific institution TBD from paper)

### Citation Requirement
```
Class23 Open-Room Handwashing Dataset
Paper: (reference from papers/2021-jimaging-chengzhang or related)
Contact authors for access
```

## Usage Notes

### Challenges
- **Open-room**: More realistic but noisier than lab
- **Uncontrolled**: Variable lighting, occlusions
- **Privacy**: Overhead helps but still concerns
- **Annotation difficulty**: Natural behavior harder to label

### Research Value
- **Realistic**: Real-world deployment conditions
- **Generalization**: Tests model robustness
- **Domain adaptation**: Lab-to-wild transfer
- **Benchmarking**: Uncontrolled environment baseline

## Key Features
- **Open-room**: Public spaces (not lab)
- **Natural behavior**: Uncontrolled (realistic)
- **Overhead cameras**: Less invasive angle
- **WHO annotations**: Frame-level labels
- **Multiple environments**: Cafeterias, restrooms

## Limitations
- **NOT PUBLIC**: Not available for download
- **Small scale**: Only 105 videos
- **Privacy concerns**: Public space recordings
- **Variable quality**: Uncontrolled conditions
- **Annotation challenges**: Natural behavior complex

## Related Datasets
- **PSKUS Hospital**: Hospital setting, larger scale, public
- **METC Lab**: Lab-based, controlled, public
- **HHA300**: Quality scores, not public
- **Stanford Depth**: Depth cameras, privacy-preserving, not public

## Applications
- **Real-world testing**: Benchmark for uncontrolled environments
- **Domain adaptation**: Study lab-to-wild transfer
- **Robustness testing**: Occlusions, variable lighting
- **Privacy research**: Overhead camera evaluation

## Research Paper
- **Conference/Journal**: See papers/2021-jimaging-chengzhang
- **Focus**: Open-room deployment and challenges
- **Contributions**: Real-world benchmarking

## Access Information
- **Status**: NOT PUBLIC
- **Request**: Contact authors via paper correspondence
- **Alternative**: Use PSKUS (public, hospital) or METC (public, lab)

</details>

- [ ] **HHA300 Hand Hygiene Assessment Dataset** (`datasets/hha300`) — tags: who-steps, quality-scoring, non-public, rgb-video, research
<details>
<summary>Show details</summary>

# HHA300 Hand Hygiene Assessment Dataset

## Overview
Hospital hand hygiene dataset with 300 videos and quality scores (0-10 scale) based on WHO compliance.

## Size & Scale
- **Total Videos**: 300 videos
- **Environment**: Hospital setting
- **Quality Scores**: 0-10 scale (WHO compliance)
- **Participants**: Hospital staff (healthcare workers)
- **Duration**: Variable per video (typically 20-60 seconds)
- **Resolution**: Variable (typically 640×480 or higher)
- **Frame Rate**: 30 FPS

## Data Collection

### Hardware Setup
- **Cameras**: Fixed cameras above sinks
- **Placement**: Overhead or side-view in hospitals
- **Environment**: Real clinical setting
- **Recording**: Continuous or motion-triggered

### Collection Protocol
- **Location**: Hospital hand hygiene stations
- **Participants**: Healthcare workers (doctors, nurses)
- **Natural Behavior**: Staff performing routine handwashing
- **Instructions**: Follow standard hospital protocols
- **Privacy**: Limited identifiable information

## Annotations

### Quality Assessment
- **Scale**: 0-10 quality scores
- **Basis**: WHO compliance (7 steps + duration)
- **Annotators**: Infectious disease experts
- **Criteria**: 
  - Completion of all WHO steps
  - Adequate duration (20+ seconds)
  - Technique quality
  - Coverage of all hand surfaces

### WHO Compliance Factors
- **Step completion**: All 7 WHO movements performed
- **Duration**: Minimum 20 seconds recommended
- **Technique**: Proper execution of each movement
- **Coverage**: All hand surfaces cleaned

### Annotation Process
- **Expert review**: Infectious disease specialists
- **Guidelines**: WHO hand hygiene guidelines
- **Scoring**: Holistic quality assessment (not frame-level)
- **Consistency**: Multiple reviewers for reliability

## Dataset Structure

### File Organization
```
hha300/
├── videos/
│   ├── video001.mp4
│   ├── video002.mp4
│   └── ...
└── annotations/
    └── quality_scores.csv
```

### Quality Scores CSV
- **Columns**: video_id, quality_score, duration, notes
- **Format**: CSV with video-level quality scores
- **Range**: 0 (poor) to 10 (perfect WHO compliance)

## Sample Statistics

### Video Characteristics
- **Total**: 300 videos
- **Duration**: Variable (20-60 seconds typical)
- **Quality**: Real hospital conditions (variable lighting)
- **Participants**: Healthcare workers

### Quality Distribution
- **Range**: 0-10 scale
- **Distribution**: Variable (natural hospital data)
- **High scores**: Perfect WHO compliance (rare)
- **Low scores**: Missing steps or poor technique

## Public Availability

### Access Status
- **NOT PUBLIC**: Dataset not publicly released
- **Reason**: Privacy concerns (hospital staff)
- **Request**: May be available upon request to authors
- **Institution**: Hospital research group

### Citation Requirement
```
HHA300 Hand Hygiene Assessment Dataset
Contact authors for access (paper reference TBD)
```

## Usage Notes

### Research Applications
- **Quality prediction**: Train models to predict compliance scores
- **Technique assessment**: Automated WHO compliance checking
- **Benchmarking**: Compare quality assessment methods
- **Feedback systems**: Real-time coaching

### Challenges
- **Not public**: Limited access
- **Quality subjective**: Expert scores may vary
- **Hospital privacy**: Restrictions on sharing
- **Small scale**: 300 videos (smaller than PSKUS)

## Key Features
- **Quality-focused**: Holistic WHO compliance scores
- **Hospital data**: Real clinical environment
- **Expert annotations**: Infectious disease specialists
- **Comprehensive**: All WHO criteria considered

## Limitations
- **NOT PUBLIC**: Not available for download
- **Small scale**: Only 300 videos
- **Quality scores only**: No frame-level WHO step labels
- **Subjective**: Expert scoring may vary
- **Privacy**: Hospital staff concerns

## Related Datasets
- **PSKUS Hospital**: Larger (3,185 episodes), frame-level labels, public
- **METC Lab**: Lab-based, frame-level labels, public
- **UWash**: Smartwatch IMU, quality assessment, public
- **Class23**: Open-room, not public

## Applications
- **Quality prediction**: Automated WHO compliance scoring
- **Feedback systems**: Real-time coaching during handwashing
- **Training**: Healthcare worker technique improvement
- **Benchmarking**: Compare quality assessment methods
- **Research**: Study factors affecting compliance

## Comparison with Other Datasets

### vs PSKUS
- **HHA300**: Quality scores (holistic assessment)
- **PSKUS**: Frame-level WHO step labels
- **HHA300**: 300 videos (smaller)
- **PSKUS**: 3,185 episodes (much larger, public)

### vs UWash
- **HHA300**: Vision-based (hospital cameras)
- **UWash**: IMU-based (smartwatch sensors)
- **Both**: Quality assessment focus

## Access Information
- **Status**: NOT PUBLIC
- **Request**: Contact authors via paper
- **Alternative**: Use PSKUS (public, larger) or UWash (public, IMU-based)

## Research Value
- **Quality assessment**: Gold standard for WHO compliance
- **Hospital validation**: Real-world clinical data
- **Expert labels**: High-quality annotations
- **Benchmarking**: Standard for quality prediction models

</details>

- [ ] **Kaggle Handwash (Resorted to WHO 6+1 Classes)** (`datasets/kaggle-who6`) — tags: who-steps, public, rgb-video, kaggle, 7-classes
<details>
<summary>Show details</summary>

# Kaggle Handwash (Resorted to WHO 6+1 Classes)

## Overview
Public subset of the Kaggle hand-wash videos, re-sorted into 7 folders to align with WHO steps (left/right merged; wrist/rinse in "Other").

## Size & Scale
- **Total Videos**: ~292 short video clips
- **Classes**: 7 folders (WHO 6 steps + "Other")
- **Duration**: Short clips (typically 5-15 seconds each)
- **Total Duration**: Variable (few minutes total)
- **Source**: Kaggle Hand Wash Dataset (public competition)

## Data Collection

### Original Source
- **Platform**: Kaggle (public dataset)
- **Original Structure**: 12 classes (left/right hand variants separate)
- **Resorting**: Reorganized to match WHO 6-step taxonomy
- **Purpose**: Align with PSKUS/METC label schemes

### Preprocessing
- **Left/Right Merge**: Combine left/right hand variants into single classes
- **Wrist/Rinse**: Moved to "Other" class (not core WHO movements)
- **Result**: 7 folders (6 WHO steps + 1 "Other")

## Annotations

### Class Structure (7 folders)
1. **Step 1**: Palm to palm
2. **Step 2**: Palm over dorsum with interlaced fingers
3. **Step 3**: Palm to palm with fingers interlaced
4. **Step 4**: Back of fingers to opposing palms
5. **Step 5**: Rotational rubbing of thumb
6. **Step 6**: Fingertips to palm
7. **Other**: Wrist washing, rinsing, non-core movements

### Resorting from Original
- **Original**: 12 classes (left/right variants separate)
- **Resorted**: 7 classes (left/right merged)
- **Rationale**: Match PSKUS/METC taxonomy (no left/right distinction)

## Dataset Structure

### File Organization
```
kaggle-who6/
├── step1/ (palm to palm)
├── step2/ (palm over dorsum)
├── step3/ (interlaced)
├── step4/ (back of fingers)
├── step5/ (thumb rub)
├── step6/ (fingertips)
└── other/ (wrist, rinse, etc.)
```

### File Formats
- **Videos**: Short clips (5-15 seconds)
- **Format**: MP4 or AVI
- **Resolution**: Variable (typically 640×480 or 1080p)
- **Frame Rate**: Variable (typically 30 FPS)

## Sample Statistics

### Video Characteristics
- **Total Clips**: ~292
- **Per Class**: ~40-50 clips per WHO step
- **Duration**: Short (5-15 seconds)
- **Quality**: Variable (Kaggle submissions)

### Environment
- **Setting**: Various (home, lab, portable sinks)
- **Lighting**: Variable (indoor, outdoor)
- **Backgrounds**: Diverse (different sinks, settings)

## Public Availability

### Download Access
- **Platform**: GitHub mirror + Kaggle
- **License**: Public (Kaggle open data)
- **Download Script**: `datasets/kaggle-who6/fetch.sh`
- **Mirror**: https://github.com/atiselsts/data/raw/master/kaggle-dataset-6classes.tar

### Download Instructions
```bash
cd datasets/kaggle-who6
./fetch.sh
# Dataset will be downloaded from GitHub mirror and extracted
```

## Usage Examples

### Load Videos
```python
import cv2
import os

# List videos in step1 folder
step1_videos = os.listdir("kaggle-who6/step1/")

# Load first video
video = cv2.VideoCapture(f"kaggle-who6/step1/{step1_videos[0]}")
while True:
    ret, frame = video.read()
    if not ret:
        break
    # Process frame...
```

### Training Split
```python
from sklearn.model_selection import train_test_split

# Split videos into train/val/test
all_videos = []
labels = []

for class_idx, class_name in enumerate(["step1", "step2", ..., "other"]):
    videos = os.listdir(f"kaggle-who6/{class_name}/")
    all_videos.extend([f"{class_name}/{v}" for v in videos])
    labels.extend([class_idx] * len(videos))

train_val, test = train_test_split(all_videos, test_size=0.1)
train, val = train_test_split(train_val, test_size=0.15)
```

## Key Features
- **Public**: Fully open (Kaggle + GitHub mirror)
- **Small**: ~292 clips (quick experiments)
- **WHO-aligned**: Resorted to match PSKUS/METC taxonomy
- **Diverse**: Various environments and settings
- **Short clips**: Easy to process (5-15 seconds each)

## Limitations
- **Small scale**: Only ~292 videos (much smaller than PSKUS)
- **No frame-level labels**: Only clip-level labels (entire clip = one class)
- **Variable quality**: Kaggle submissions (inconsistent)
- **No temporal annotations**: Cannot train temporal segmentation
- **Resorting artifacts**: Original 12-class structure lost

## Related Datasets
- **PSKUS Hospital**: Much larger (3,185 episodes), frame-level labels
- **METC Lab**: Lab-based, frame-level labels
- **Original Kaggle**: 12 classes (before resorting)

## Applications
- **Quick experiments**: Baseline models on small dataset
- **Ablation studies**: Compare small vs large dataset performance
- **Transfer learning**: Pre-train on Kaggle, fine-tune on PSKUS
- **Proof of concept**: Fast prototyping before large-scale training
- **Benchmarking**: Test model generalization across datasets

## Citation
```
Kaggle Hand Wash Dataset (resorted to WHO 6+1 classes)
Original: https://www.kaggle.com/realtimear/hand-wash-dataset
Resorted mirror: https://github.com/atiselsts/data/raw/master/kaggle-dataset-6classes.tar
```

## Download Size
- **Compressed**: ~Few hundred MB
- **Uncompressed**: ~1-2 GB (short clips)

</details>

- [ ] **METC Lab Handwashing Dataset** (`datasets/metc-lab`) — tags: who-steps, lab, rgb-video, public, zenodo
<details>
<summary>Show details</summary>

# METC Lab Handwashing Dataset

## Overview
Lab-collected WHO handwash recordings from the Medical Education Technology Center (Riga Stradins University) with multiple camera interfaces.

## Size & Scale
- **Total Videos**: Multiple sessions (exact count in summary.csv)
- **Camera Interfaces**: 3 different interfaces/setups
- **Participants**: Lab volunteers and students
- **Environment**: Controlled lab setting (Medical Education Technology Center)
- **Resolution**: Various (typically 640×480 or higher)
- **Frame Rate**: 30 FPS

## Data Collection

### Hardware Setup
- **Location**: Medical Education Technology Center, Riga Stradins University
- **Cameras**: Multiple camera interfaces (Interface_number_1, 2, 3)
- **Environment**: Controlled lab (consistent lighting, fixed camera angles)
- **Participants**: Students and volunteers (controlled sessions)

### Collection Protocol
- **Setting**: Lab-based (not real-world hospital)
- **Instructions**: Participants follow WHO guidelines
- **Supervision**: Controlled sessions with guidance
- **Quality**: Higher quality than real-world (better lighting, angles)

## Annotations

### Annotation Process
- **Tool**: Custom annotation tool (similar to PSKUS)
- **Annotators**: Medical professionals and students
- **Guidelines**: WHO guidelines (same as PSKUS)
- **Label Scheme**: Aligned with PSKUS (6 steps + "Other")

### Frame-Level Labels
Each frame annotated with:
- **Movement Code**: WHO step (1-6) or "Other" (0)
- **Frame Time**: Timestamp within video
- **Is Washing**: Binary flag

### WHO Movement Classes (6 + Other)
Same as PSKUS:
1. Palm to palm
2. Palm over dorsum with interlaced fingers
3. Palm to palm with fingers interlaced
4. Back of fingers to opposing palms
5. Rotational rubbing of thumb
6. Fingertips to palm
7. Other (non-washing, rinsing, faucet control)

## Dataset Structure

### File Organization
```
METC/
├── Interface_number_1/
│   ├── video1.mp4
│   ├── video1.json
│   └── ...
├── Interface_number_2/
├── Interface_number_3/
├── summary.csv (episode metadata)
└── statistics.csv (dataset stats)
```

### File Formats
- **Videos**: MP4 format
- **Annotations**: JSON (per video, same format as PSKUS)
- **Metadata**: CSV (summary.csv, statistics.csv)

## Sample Statistics

### Video Characteristics
- **Resolution**: Various (typically ≥640×480)
- **Frame Rate**: 30 FPS
- **Duration**: Variable (standard WHO duration ~20-40 seconds)
- **Quality**: Higher than hospital data (controlled environment)

### Camera Interfaces
- **Interface 1**: Specific camera setup/angle
- **Interface 2**: Alternative setup/angle
- **Interface 3**: Third setup/angle
- **Purpose**: Multi-view diversity for generalization

## Public Availability

### Download Access
- **Platform**: Zenodo (open repository)
- **License**: Open research use (cite paper)
- **Download Script**: `datasets/metc-lab/fetch.sh`

### Download Instructions
```bash
cd datasets/metc-lab
./fetch.sh
# Dataset will be downloaded from Zenodo and extracted
```

### Citation Requirement
```
METC Lab Handwashing Dataset
Medical Education Technology Center, Riga Stradins University
Available on Zenodo
```

## Usage Examples

### Frame Extraction
```python
import cv2

video = cv2.VideoCapture("Interface_number_1/video1.mp4")
while True:
    ret, frame = video.read()
    if not ret:
        break
    # Process frame...
```

### Load Annotations
```python
import json

with open("Interface_number_1/video1.json", "r") as f:
    annotations = json.load(f)

for anno in annotations:
    movement_code = anno["movement_code"]
```

## Key Features
- **Lab-based**: Controlled environment (consistent quality)
- **Multi-interface**: 3 camera setups for diversity
- **WHO-compliant**: Same label scheme as PSKUS
- **Public**: Available on Zenodo
- **Frame-level annotations**: Dense temporal labels

## Limitations
- **Lab setting**: Not real-world (may not generalize to hospitals)
- **Controlled**: Less natural behavior than hospital data
- **Smaller scale**: Fewer videos than PSKUS
- **Participant diversity**: Limited to lab volunteers

## Related Datasets
- **PSKUS Hospital**: Real-world hospital (larger, more diverse)
- **Kaggle WHO6**: Public, smaller, similar structure
- **Synthetic**: Large-scale, perfect annotations

## Applications
- **Training**: Controlled data for initial model training
- **Transfer learning**: Pre-train on METC, fine-tune on hospital
- **Ablation studies**: Compare lab vs hospital performance
- **Benchmarking**: Lab-to-wild generalization testing

</details>

- [ ] **OCDetect Wrist-Worn IMU Dataset** (`datasets/ocdetect-wrist`) — tags: wearable, imu, public, ocd, event-detection
<details>
<summary>Show details</summary>

# OCDetect Wrist-Worn IMU Dataset

## Overview
Smartwatch IMU dataset for detecting compulsive handwashing behaviors (OCD) with accelerometer and gyroscope data.

## Size & Scale
- **Focus**: Obsessive-Compulsive Disorder (OCD) detection
- **Participants**: Multiple (including OCD patients and controls)
- **Sensors**: 3-axis accelerometer, gyroscope
- **Device**: Smartwatch (wrist-worn)
- **Sessions**: Multiple per participant
- **Duration**: Variable per session

## Data Collection

### Hardware
- **Device**: Commercial smartwatch with IMU
- **Sensors**:
  - Accelerometer (3-axis)
  - Gyroscope (3-axis)
- **Placement**: Wrist-worn (dominant hand)
- **Sampling Rate**: Variable (smartwatch-dependent, typically 50-100 Hz)

### Collection Protocol
- **Participants**: OCD patients + healthy controls
- **Environment**: Various (home, clinic)
- **Natural Behavior**: Participants perform daily activities
- **Focus**: Detecting compulsive handwashing patterns
- **Duration**: Extended monitoring periods

### Clinical Context
- **OCD**: Obsessive-Compulsive Disorder
- **Compulsive Handwashing**: Repetitive, excessive handwashing
- **Detection Goal**: Identify abnormal handwashing frequency/duration
- **Clinical Value**: Monitor symptom severity, treatment effectiveness

## Annotations

### Labels
- **Binary**: Compulsive handwashing vs normal handwashing
- **Frequency**: Number of handwashing events per day
- **Duration**: Length of each handwashing episode
- **Intensity**: Severity of compulsive behavior
- **Not WHO Steps**: Does not classify individual movements

### Annotation Process
- **Clinical Review**: Mental health professionals
- **Ground Truth**: Patient self-reports + observer logs
- **Criteria**: DSM-5 OCD diagnostic criteria
- **Validation**: Cross-reference with clinical assessments

## Dataset Structure

### File Organization
```
ocdetect-wrist/
├── participants/
│   ├── ocd_patient1/
│   │   ├── imu_data.csv
│   │   └── annotations.csv
│   ├── control1/
│   └── ...
└── metadata/
    └── participant_info.csv
```

### Data Format
- **IMU CSV**: timestamp, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
- **Annotations CSV**: event_id, start_time, end_time, label, severity
- **Metadata**: participant demographics, OCD diagnosis status

## Sample Statistics

### IMU Characteristics
- **Accelerometer**: ±2g, ±4g, or ±8g range
- **Gyroscope**: ±250, ±500, or ±2000 dps range
- **Sampling Rate**: 50-100 Hz
- **Sessions**: Multiple per participant (days/weeks)

### Participant Characteristics
- **OCD Patients**: Diagnosed with compulsive handwashing
- **Controls**: Healthy individuals (no OCD)
- **Age**: Variable (adults)
- **Gender**: Mixed

## Public Availability

### Access Status
- **PUBLIC**: Available for research
- **Platform**: Research repository or project website
- **License**: Research use (cite paper)
- **Format**: Raw IMU CSV + annotations

### Download Instructions
```bash
# Check project website or paper for download link
# Typically: Google Drive, Zenodo, or project GitHub
```

## Usage Examples

### Load IMU Data
```python
import pandas as pd

# Load IMU data
imu_df = pd.read_csv("participants/ocd_patient1/imu_data.csv")

# Extract sensor values
acc = imu_df[['acc_x', 'acc_y', 'acc_z']].values
gyro = imu_df[['gyro_x', 'gyro_y', 'gyro_z']].values
```

### Load Annotations
```python
import pandas as pd

# Load annotations
anno_df = pd.read_csv("participants/ocd_patient1/annotations.csv")

# Filter compulsive handwashing events
compulsive = anno_df[anno_df['label'] == 'compulsive']
```

## Key Features
- **Clinical focus**: OCD compulsive handwashing detection
- **Wearable-based**: Smartwatch IMU (privacy-preserving)
- **Long-term monitoring**: Extended sessions (days/weeks)
- **Binary classification**: Compulsive vs normal
- **Public**: Available for research

## Limitations
- **OCD-specific**: Limited to compulsive behavior detection
- **No WHO steps**: Does not classify individual movements
- **Smartwatch-dependent**: Requires wearable device
- **Binary only**: No fine-grained quality assessment
- **Clinical population**: May not generalize to general handwashing

## Related Datasets
- **UWash**: Smartwatch IMU, quality assessment, public
- **wearPuck**: Wrist-worn, multimodal, event detection
- **PSKUS/METC**: Vision-based, WHO steps, public

## Applications
- **OCD detection**: Identify compulsive handwashing patterns
- **Symptom monitoring**: Track OCD severity over time
- **Treatment evaluation**: Measure treatment effectiveness
- **Wearable systems**: Privacy-preserving mental health monitoring
- **Research**: Study compulsive behavior patterns

## Clinical Relevance

### OCD Handwashing
- **Symptom**: Excessive, repetitive handwashing
- **Frequency**: Multiple times per hour (vs few times per day)
- **Duration**: Prolonged episodes (>5 minutes)
- **Impact**: Skin damage, life disruption

### Detection Benefits
- **Objective monitoring**: Quantify symptom severity
- **Treatment tracking**: Measure progress over time
- **Early intervention**: Detect relapse patterns
- **Personalized care**: Tailor treatment to individual

## Research Paper
- **Focus**: Compulsive handwashing detection using wrist-worn IMU
- **Methods**: Machine learning on smartwatch accelerometer/gyroscope
- **Results**: High accuracy distinguishing compulsive vs normal
- **Applications**: Mental health monitoring, OCD treatment

## Citation
```
OCDetect: Compulsive Handwashing Detection via Wrist-Worn IMU
(Check project website or paper for full citation)
```

## Comparison with Other Wearable Datasets

### vs UWash
- **OCDetect**: OCD detection (compulsive behavior)
- **UWash**: Quality assessment (WHO compliance)
- **Both**: Smartwatch IMU, privacy-preserving

### vs wearPuck
- **OCDetect**: Clinical focus (OCD)
- **wearPuck**: General event detection
- **OCDetect**: Long-term monitoring
- **wearPuck**: Short sessions

## Access Information
- **Status**: PUBLIC (research use)
- **Download**: Check project website or paper for link
- **Alternative**: Use UWash (quality assessment) or wearPuck (event detection)

</details>

- [ ] **Portable51 & Farm23 Shadow Augmentation Datasets** (`datasets/portable51-farm23`) — tags: outdoor, shadows, non-public, robustness
<details>
<summary>Show details</summary>

# Portable51 & Farm23 Shadow Augmentation Datasets

## Overview
Two shadow robustness datasets (Portable51, Farm23) for testing handwashing models under challenging shadow conditions.

## Size & Scale
- **Portable51**: 51 videos (portable sink, controlled shadows)
- **Farm23**: 23 videos (farm setting, natural shadows)
- **Total**: 74 videos across 2 environments
- **Purpose**: Shadow robustness evaluation
- **Annotation**: WHO step labels with shadow characteristics

## Data Collection

### Portable51 Dataset
- **Environment**: Portable sink in controlled setting
- **Shadow Conditions**: Artificial shadows (controlled)
- **Videos**: 51 recordings
- **Participants**: Lab volunteers
- **Camera**: Fixed overhead position
- **Lighting**: Variable (simulating different shadow conditions)

### Farm23 Dataset
- **Environment**: Farm setting (outdoor sink)
- **Shadow Conditions**: Natural outdoor shadows
- **Videos**: 23 recordings
- **Participants**: Farm workers
- **Camera**: Fixed position above outdoor sink
- **Lighting**: Natural sunlight (time-of-day variations)

## Annotations

### WHO Step Labels
Standard WHO 6-step taxonomy:
1. Palm to palm
2. Palm over dorsum with interlaced fingers
3. Palm to palm with fingers interlaced
4. Back of fingers to opposing palms
5. Rotational rubbing of thumb
6. Fingertips to palm
7. Other (non-washing, rinsing)

### Shadow Characteristics
- **Shadow Intensity**: Quantified shadow strength
- **Shadow Coverage**: Percentage of hands occluded
- **Shadow Type**: Artificial vs natural
- **Challenge Level**: Easy, medium, hard (based on occlusion)

## Dataset Structure

### File Organization
```
portable51-farm23/
├── portable51/
│   ├── videos/
│   │   ├── video001.mp4
│   │   └── ...
│   └── annotations/
│       └── labels.csv
├── farm23/
│   ├── videos/
│   │   ├── video001.mp4
│   │   └── ...
│   └── annotations/
│       └── labels.csv
└── shadow_analysis/
    └── shadow_metrics.csv
```

### File Formats
- **Videos**: MP4 format
- **Annotations**: CSV (frame-level WHO step labels)
- **Shadow Metrics**: CSV (shadow characteristics per frame)
- **Resolution**: Variable (typically 640×480 or 1080p)
- **Frame Rate**: 30 FPS

## Sample Statistics

### Portable51 Characteristics
- **Total Videos**: 51
- **Duration**: Variable per video (20-60 seconds)
- **Shadow Type**: Artificial (controlled)
- **Lighting**: Variable (simulated shadow conditions)

### Farm23 Characteristics
- **Total Videos**: 23
- **Duration**: Variable per video (20-60 seconds)
- **Shadow Type**: Natural (outdoor)
- **Lighting**: Sunlight (time-of-day variations)

## Public Availability

### Access Status
- **NOT PUBLIC**: Datasets not publicly released
- **Reason**: Research-specific, limited distribution
- **Request**: May be available upon request to authors
- **Institution**: Research group (papers/2024-mmsp-shadow-augmentation)

### Citation Requirement
```
Portable51 & Farm23 Shadow Augmentation Datasets
Paper: papers/2024-mmsp-shadow-augmentation
Contact authors for access
```

## Usage Notes

### Shadow Robustness Testing
- **Purpose**: Evaluate model performance under shadow occlusion
- **Benchmark**: Test trained models on shadow-augmented data
- **Ablation**: Compare performance with/without shadow augmentation
- **Domain**: Portable51 (controlled) vs Farm23 (natural)

### Research Applications
- **Robustness**: Test shadow resilience
- **Augmentation**: Validate shadow augmentation techniques
- **Domain adaptation**: Indoor vs outdoor shadows
- **Generalization**: Test cross-domain performance

## Key Features
- **Shadow-focused**: Specifically designed for shadow robustness
- **Dual environments**: Portable sink (controlled) + farm (natural)
- **WHO annotations**: Frame-level step labels
- **Shadow metrics**: Quantified shadow characteristics
- **Challenging**: Tests model robustness to occlusion

## Limitations
- **NOT PUBLIC**: Not available for download
- **Small scale**: Only 74 videos total
- **Shadow-specific**: Limited to shadow conditions
- **Limited diversity**: 2 environments only
- **Research-only**: Not general-purpose benchmark

## Related Datasets
- **PSKUS Hospital**: Larger, public, indoor (minimal shadows)
- **METC Lab**: Lab-based, controlled (minimal shadows)
- **Class23**: Open-room, variable lighting
- **Stanford Depth**: Depth-based (less sensitive to shadows)

## Applications
- **Shadow robustness testing**: Evaluate model performance under occlusion
- **Augmentation validation**: Test shadow augmentation techniques
- **Domain adaptation**: Indoor vs outdoor generalization
- **Benchmarking**: Shadow-specific performance metrics
- **Research**: Study shadow effects on WHO step recognition

## Research Paper
- **Title**: "Shadow Augmentation for Robust Hand Hygiene Recognition"
- **Year**: 2024
- **Conference**: MMSP 2024
- **Focus**: Shadow robustness and augmentation techniques
- **Paper**: papers/2024-mmsp-shadow-augmentation/paper.pdf

## Shadow Augmentation Techniques

### Tested Methods
- **CutMix**: Random rectangular shadow regions
- **Shadow synthesis**: Realistic shadow generation
- **Occlusion augmentation**: Simulated hand occlusion
- **Performance**: Improved model robustness

### Results
- **Baseline**: Poor performance on shadow-heavy videos
- **With augmentation**: Significant improvement in shadow conditions
- **Generalization**: Better cross-domain performance

## Access Information
- **Status**: NOT PUBLIC
- **Request**: Contact authors via paper (2024-mmsp-shadow-augmentation)
- **Alternative**: Use PSKUS/METC (public, indoor lighting) or generate synthetic shadows

</details>

- [ ] **PSKUS Hospital Handwashing Dataset** (`datasets/pskus-hospital`) — tags: who-steps, hospital, rgb-video, public, zenodo
<details>
<summary>Show details</summary>

# PSKUS Hospital Handwashing Dataset

## Overview
Real-world hospital videos with 3,185 hand-washing episodes annotated frame-by-frame using WHO movement codes.

## Size & Scale
- **Total Episodes**: 3,185 annotated handwashing episodes
- **Total Videos Captured**: 32,471 (subset of 3,185 annotated)
- **Annotated Videos**: 2,293 files
  - 1,199 annotated once
  - 1,094 annotated twice (double annotation for reliability)
- **Total Annotations**: 6,690 (including double annotations)
- **Total Washing Time**: 83,804 seconds
- **WHO Movements Time**: 27,517 seconds (movements 1-7)

## Data Collection

### Hardware Setup
- **Cameras**: AirLive POE 100CAM, Axis M3046V IP cameras
- **Controller**: Raspberry Pi 4 single-board computers
- **Storage**: Micro SD cards (local storage)
- **Network**: Netgear 5-Port PoE Gigabit Ethernet switch
- **Placement**: Overhead above sinks (9 sites)

### Collection Protocol
- **Location**: Pauls Stradins Clinical University Hospital (Latvia)
- **Duration**: 3 months continuous deployment
- **Trigger**: Motion detection >10 seconds
- **Recording**: Automatic (motion-triggered)
- **Resolution**: 640×480 or 320×240 pixels
- **Frame Rate**: 30 FPS
- **Storage**: Local SD cards, monthly manual collection

### Deployment Sites
- **Total Sites**: 9 hospital locations
- **Environment**: Real clinical setting (sinks in hospital)
- **Participants**: Hospital staff (doctors, nurses, technicians)
- **Natural Behavior**: Staff unaware of specific recording times

## Annotations

### Annotation Process
- **Tool**: Custom Python + OpenCV GUI
- **Annotators**: 
  - Infectious disease specialists
  - Medical professionals
  - Riga Stradins University students
- **Guidelines**: Developed with local infectious disease specialists
- **Quality Control**: 1,094 videos double-annotated

### Frame-Level Labels
Each frame annotated with:
- **Movement Code**: WHO step (1-7) or "Other" (0)
- **Frame Time**: Timestamp within video
- **Is Washing**: Binary flag (washing vs non-washing)
- **Additional Flags**: Ring, watch, lacquered nails (inappropriate for medical staff)

### WHO Movement Classes (7 + Other)
1. **Palm to palm** rubbing
2. **Palm over dorsum** with interlaced fingers
3. **Palm to palm** with fingers interlaced
4. **Back of fingers** to opposing palms
5. **Rotational rubbing** of thumb
6. **Fingertips** to palm (circular motion)
7. **Turning off faucet** with paper towel
8. **Other** (non-washing actions, rinsing, etc.)

## Dataset Structure

### File Organization
```
PSKUS/
├── DataSet1/
│   ├── video1.mp4
│   ├── video1.json (frame-level annotations)
│   └── ...
├── DataSet2/
├── ...
├── summary.csv (episode metadata)
├── statistics.csv (dataset-wide stats)
└── statistics-with-locations.csv (location-based splits)
```

### File Formats
- **Videos**: MP4 or AVI (H.264/MJPEG codecs)
- **Annotations**: JSON (per video)
  ```json
  {
    "frame_number": int,
    "frame_time": float,
    "movement_code": int,
    "is_washing": bool
  }
  ```
- **Metadata**: CSV (summary.csv, statistics.csv)

### CSV Files
- **summary.csv**: Episode-level metadata (ID, duration, movement counts)
- **statistics.csv**: Dataset-wide statistics (class distributions, durations)
- **statistics-with-locations.csv**: Location-based splits (for train/test)

## Sample Statistics

### Video Characteristics
- **Resolution**: 320×240 or 640×480 pixels
- **Frame Rate**: 30 FPS
- **Duration**: Variable (typically 20-60 seconds per episode)
- **Total Duration**: 83,804 seconds (~23.3 hours)

### Class Distribution
- **WHO Movements**: Detailed counts in statistics.csv
- **Balanced**: Reasonable coverage of all WHO steps
- **Real-world**: Natural distribution (not artificially balanced)

### Annotation Quality
- **Double Annotation**: 1,094 videos (59% of annotated set)
- **Inter-Annotator Agreement**: High (Cohen's kappa reported in paper)
- **Ambiguities**: Identified at step transitions

## Public Availability

### Download Access
- **Platform**: Zenodo (open repository)
- **License**: Open research use (cite paper)
- **Size**: ~Several GB (compressed)
- **Download Script**: `datasets/pskus-hospital/fetch.sh`

### Download Instructions
```bash
cd datasets/pskus-hospital
./fetch.sh
# Dataset will be downloaded from Zenodo and extracted
```

### Citation Requirement
```
M. Lulla, A. Rutkovskis, A. Slavinska, A. Vilde, A. Gromova, M. Ivanovs, 
A. Skadins, R. Kadikis and A. Elsts
"Hand Washing Video Dataset Annotated According to the World Health Organization's Handwashing Guidelines"
Data, 6(4), p.38, 2021
```

## Usage Examples

### Training Data Split
- **Recommended**: Use `statistics-with-locations.csv` for location-based splits
- **Purpose**: Ensure train/test videos from different camera locations
- **Benefit**: Better generalization testing

### Frame Extraction
```python
# Extract frames from videos
import cv2

video = cv2.VideoCapture("DataSet1/video1.mp4")
while True:
    ret, frame = video.read()
    if not ret:
        break
    # Process frame...
```

### Load Annotations
```python
import json

with open("DataSet1/video1.json", "r") as f:
    annotations = json.load(f)

for anno in annotations:
    frame_num = anno["frame_number"]
    movement_code = anno["movement_code"]
    is_washing = anno["is_washing"]
```

## Key Features
- **Real-world data**: Hospital setting (not lab)
- **Frame-level annotations**: Dense temporal labels
- **Large-scale**: 3,185 episodes
- **Double annotation**: 1,094 videos for reliability
- **Public**: Available on Zenodo
- **WHO-compliant**: Follows official guidelines

## Limitations
- **Single hospital**: Limited to one environment (Latvia)
- **Overhead cameras**: Fixed angle (may miss some movements)
- **Resolution**: 320×240 or 640×480 (not HD)
- **Annotation ambiguity**: Some transitions unclear
- **No depth**: RGB only (no depth/privacy-preserving)

## Related Datasets
- **METC Lab Dataset**: Lab-based, similar structure
- **Kaggle WHO6**: Public, smaller, resorted to WHO taxonomy
- **HHA300**: Quality scores, not public
- **Class23**: Open-room, not public

## Applications
- **Deep learning**: Train CNNs, LSTMs, transformers
- **Benchmarking**: Standard test set for WHO step recognition
- **Transfer learning**: Pre-train on PSKUS, fine-tune on smaller datasets
- **Temporal segmentation**: Detect step boundaries
- **Quality assessment**: Compliance with WHO guidelines

</details>

- [ ] **Stanford Depth Camera Hand Hygiene Dataset** (`datasets/stanford-depth`) — tags: depth, privacy-preserving, non-public, hospital
<details>
<summary>Show details</summary>

# Stanford Depth Camera Hand Hygiene Dataset

## Overview
Privacy-preserving depth camera dataset capturing ~20 hours of hospital hand hygiene with silhouette-based anonymization.

## Size & Scale
- **Total Duration**: ~20 hours of recordings
- **Environment**: Hospital setting (Stanford Hospital)
- **Camera Type**: Depth cameras (Kinect-like sensors)
- **Privacy**: Silhouette-based (no RGB/identifiable features)
- **Participants**: Hospital staff and visitors
- **Resolution**: Depth map resolution (typically 320×240 or 640×480)

## Data Collection

### Hardware Setup
- **Sensors**: Depth cameras (Microsoft Kinect or similar)
- **Placement**: Above or beside hand hygiene stations
- **Modality**: Depth-only (no RGB)
- **Recording**: Continuous or motion-triggered
- **Privacy Design**: No color/facial features captured

### Collection Protocol
- **Location**: Stanford Hospital hand hygiene stations
- **Duration**: Extended deployment (~20 hours total)
- **Participants**: Natural hospital users (staff, visitors)
- **Behavior**: Spontaneous hand hygiene (no instructions)
- **Privacy-First**: Depth-only capture for anonymization

### Privacy Features
- **No RGB**: Only depth information captured
- **Silhouettes**: Body/hand shapes only (no identifiable features)
- **Anonymization**: Inherent privacy protection
- **Hospital Approved**: Meets privacy requirements for deployment

## Annotations

### Annotation Process
- **Annotators**: Trained reviewers
- **Guidelines**: WHO hand hygiene guidelines
- **Labels**: Handwashing events and durations
- **Quality**: Compliance assessment (presence/absence)

### Event Detection
- **Binary**: Hand hygiene event vs no event
- **Duration**: Event start/end timestamps
- **Not WHO Steps**: Does not classify individual movements
- **Focus**: Detection and duration, not step-by-step analysis

## Dataset Structure

### File Organization
```
stanford-depth/
├── depth_videos/
│   ├── session001.depth
│   ├── session002.depth
│   └── ...
└── annotations/
    └── events.csv
```

### File Formats
- **Depth Videos**: Depth map sequences (proprietary or standard format)
- **Annotations**: CSV with event timestamps
- **Resolution**: Depth sensor resolution (320×240 or 640×480)
- **Frame Rate**: Typically 30 FPS

## Sample Statistics

### Video Characteristics
- **Total Duration**: ~20 hours
- **Event Count**: Multiple events per hour
- **Depth Range**: 0.5-4 meters (typical depth camera range)
- **Quality**: Consistent (controlled hospital lighting)

### Depth Camera Characteristics
- **Technology**: Time-of-flight or structured light
- **Range**: 0.5-4 meters (optimal for sink monitoring)
- **Resolution**: 320×240 or 640×480 depth pixels
- **Privacy**: No color/RGB data captured

## Public Availability

### Access Status
- **NOT PUBLIC**: Dataset not publicly released
- **Reason**: Hospital privacy policies
- **Request**: May be available upon request to authors
- **Institution**: Stanford University research group

### Citation Requirement
```
Stanford Depth Camera Hand Hygiene Dataset
Paper: papers/2015-stanford-depth
Contact authors for access
```

## Usage Notes

### Privacy Advantages
- **Depth-only**: No identifiable features (faces, skin tone)
- **Silhouettes**: Body shapes only
- **Hospital-approved**: Meets strict privacy requirements
- **Public deployment**: Feasible in sensitive environments

### Research Applications
- **Privacy-preserving monitoring**: Hospital/clinic deployment
- **Event detection**: Binary presence/absence
- **Duration monitoring**: Compliance with 20+ second guideline
- **Transfer learning**: Depth-to-RGB domain adaptation

## Key Features
- **Privacy-preserving**: Depth-only (no RGB)
- **Hospital deployment**: Real clinical setting (~20 hours)
- **Silhouette-based**: Anonymous body/hand shapes
- **Event detection**: Handwashing presence and duration
- **Pioneering**: Early work in privacy-preserving hand hygiene

## Limitations
- **NOT PUBLIC**: Not available for download
- **Depth-only**: Limited information compared to RGB
- **Event detection**: No WHO step classification
- **Small annotations**: Limited labeled events
- **Technology-specific**: Requires depth cameras for inference

## Related Datasets
- **PSKUS Hospital**: RGB, larger scale, frame-level labels, public
- **METC Lab**: RGB, controlled, frame-level labels, public
- **Synthetic**: RGB+depth, perfect labels, public
- **HHA300**: RGB, quality scores, not public

## Applications
- **Privacy-preserving monitoring**: Deploy in hospitals without RGB
- **Event detection**: Binary handwashing detection
- **Duration monitoring**: Compliance with time guidelines
- **Research**: Depth vs RGB performance comparison
- **Policy compliance**: Meet strict privacy requirements

## Research Paper
- **Year**: 2015
- **Institution**: Stanford University
- **Focus**: Privacy-preserving hand hygiene monitoring
- **Innovation**: Depth-only approach for hospital deployment
- **Paper**: papers/2015-stanford-depth/paper.pdf

## Comparison with RGB Datasets

### vs PSKUS (RGB)
- **Stanford**: Depth-only, privacy-preserving
- **PSKUS**: RGB, more visual information
- **Stanford**: Event detection only
- **PSKUS**: Frame-level WHO step labels

### Privacy Trade-off
- **Depth**: Better privacy, less information
- **RGB**: More information, privacy concerns
- **Hospital deployment**: Depth often preferred

## Access Information
- **Status**: NOT PUBLIC
- **Request**: Contact Stanford authors via paper
- **Alternative**: Use PSKUS (public, RGB) or Synthetic (public, RGB+depth)

## Historical Significance
- **Early work**: One of first privacy-preserving approaches (2015)
- **Depth cameras**: Pioneered depth-only hand hygiene monitoring
- **Influence**: Inspired later privacy-focused research

</details>

- [ ] **Synthetic Hand-Washing Gesture Dataset (Özakar & Gedikli, 2025)** (`datasets/synthetic-blender-rozakar`) — tags: synthetic, rgb, depth, segmentation, public
<details>
<summary>Show details</summary>

# Synthetic Hand-Washing Gesture Dataset (Özakar & Gedikli, 2025)

## Overview
CGI-rendered dataset with 96,000 frames (64 minutes) across 8 WHO-inspired gestures, 4 characters, and 4 environments (RGB, depth, masks).

## Size & Scale
- **Total Frames**: 96,000 frames
- **Total Duration**: 64 minutes (25 FPS)
- **Characters**: 4 diverse 3D hand models
- **Environments**: 4 realistic bathroom settings
- **Gestures**: 8 WHO-inspired handwashing movements
- **Modalities**: RGB, depth, depth-isolated, hand segmentation masks

## Data Generation

### Rendering Pipeline
- **Tool**: Blender 3D rendering engine
- **Resolution**: 960×540 pixels
- **Frame Rate**: 25 FPS
- **Rendering**: Cycles renderer (realistic lighting)

### Character Variations
- **Count**: 4 diverse character models
- **Diversity**: Varied skin tones, hand sizes, appearances
- **Animation**: Keyframe animation for 8 gestures per character

### Environment Variations
- **Count**: 4 realistic bathroom scenes
- **Lighting**: Different lighting conditions per environment
- **Backgrounds**: Varied sinks, walls, fixtures
- **Purpose**: Domain diversity for generalization

## Annotations

### Gesture Classes (8 WHO-Inspired)
1. **Palm to palm** rubbing
2. **Right palm over left dorsum** with interlaced fingers
3. **Left palm over right dorsum** with interlaced fingers
4. **Palm to palm** with fingers interlaced
5. **Back of fingers** to opposing palms
6. **Rotational rubbing of right thumb**
7. **Rotational rubbing of left thumb**
8. **Fingertips to palm** (circular motion)

### Perfect Annotations
- **Automatic**: Generated during rendering (no manual labeling)
- **Frame-level**: Every frame labeled with gesture class
- **Multi-modal**: RGB, depth, and masks perfectly aligned
- **No errors**: CGI ensures 100% label accuracy

## Dataset Structure

### File Organization
```
synthetic-blender-rozakar/
├── character1/
│   ├── environment1/
│   │   ├── gesture1/
│   │   │   ├── rgb/
│   │   │   ├── depth/
│   │   │   ├── depth_isolated/
│   │   │   └── masks/
│   │   └── ...
│   └── ...
├── character2/
└── ...
```

### Modalities
1. **RGB Images**: Standard color frames (960×540)
2. **Depth Maps**: Per-pixel depth (normalized 0-1)
3. **Depth-Isolated**: Depth with background removed
4. **Hand Segmentation Masks**: Binary masks (hands = white, background = black)

### File Formats
- **Images**: PNG (lossless)
- **Depth**: PNG (grayscale, normalized)
- **Masks**: PNG (binary)
- **Organization**: Folders by character/environment/gesture

## Sample Statistics

### Combinations
- **Total**: 4 characters × 4 environments × 8 gestures = 128 unique combinations
- **Frames per Gesture**: Variable (typically few hundred frames)
- **Total Frames**: 96,000

### Distribution
- **Balanced**: Equal frames per gesture (12,000 frames per gesture)
- **Systematic**: All character/environment combinations covered

## Public Availability

### Download Access
- **Platform**: Google Drive (5 download links)
- **License**: CC BY (Attribution)
- **Size**: ~Several GB (compressed)
- **Download Links** (in README):
  - https://drive.google.com/file/d/1EW3JQvElcuXzawxEMRkA8YXwK_Ipiv-p/view
  - https://drive.google.com/file/d/163TsrDe4q5KTQGCv90JRYFkCs7AGxFip/view
  - https://drive.google.com/file/d/1GxyTYfSodumH78NbjWdmbjm8JP8AOkAY/view
  - https://drive.google.com/file/d/1IoRsgBBr8qoC3HO-vEr6E7K4UZ6ku6-1/view
  - https://drive.google.com/file/d/1svCYnwDazy5FN1DYSgqbGscvDKL_YnID/view

### Download Instructions
```bash
# Download all 5 parts from Google Drive links
# Extract to dataset folder
# Organize by character/environment/gesture
```

## Pre-trained Models

### Included Weights
- **Inception-V3**: RGB image classifier (TensorFlow/Keras)
- **YOLOv8n**: Detection and segmentation (Ultralytics)
- **PointNet**: Point cloud classifier (PyTorch)
- **Training Scripts**: Provided in upstream repo

## Usage Examples

### Load RGB Images
```python
import cv2
import os

# List RGB images for gesture 1, character 1, environment 1
rgb_path = "character1/environment1/gesture1/rgb/"
rgb_images = sorted(os.listdir(rgb_path))

# Load first image
img = cv2.imread(f"{rgb_path}/{rgb_images[0]}")
```

### Load Depth Maps
```python
import cv2

# Load depth map (grayscale PNG)
depth = cv2.imread("character1/environment1/gesture1/depth/frame001.png", cv2.IMREAD_GRAYSCALE)

# Normalize to 0-1 range
depth_normalized = depth.astype(float) / 255.0
```

### Load Segmentation Masks
```python
import cv2

# Load binary mask
mask = cv2.imread("character1/environment1/gesture1/masks/frame001.png", cv2.IMREAD_GRAYSCALE)

# Threshold to binary (hands = 255, background = 0)
mask_binary = (mask > 127).astype(np.uint8) * 255
```

## Key Features
- **Large-scale**: 96,000 frames (much larger than real datasets)
- **Perfect annotations**: No labeling errors (CGI)
- **Multi-modal**: RGB, depth, masks (all aligned)
- **Systematic variations**: All character/environment combinations
- **Public**: CC BY license (fully open)
- **Pre-trained models**: Baseline weights provided

## Limitations
- **Synthetic-to-real gap**: Domain shift (CGI vs real hands)
- **8 gestures only**: Not exact WHO 6-step taxonomy
- **No temporal modeling**: Frame-level only (no videos)
- **Limited real validation**: Few real-world test samples
- **CGI artifacts**: Rendering may not capture all nuances

## Related Datasets
- **PSKUS/METC**: Real-world, frame-level labels
- **Kaggle WHO6**: Real, smaller, clip-level
- **HHA300**: Real, quality scores, not public

## Applications
- **Pre-training**: Bootstrap models before fine-tuning on real data
- **Data augmentation**: Mix synthetic with real for robustness
- **Domain adaptation**: Study synthetic-to-real transfer
- **Ablation studies**: Controlled experiments (vary characters, environments)
- **Annotation cost reduction**: Free perfect labels

## Research Paper
- **Title**: "Hand Washing Gesture Recognition Using Synthetic Dataset"
- **Authors**: Rüstem Özakar, Eyüp Gedikli
- **Journal**: Journal of Imaging, MDPI, 2025
- **DOI**: 10.3390/jimaging11070208
- **URL**: https://www.mdpi.com/2313-433X/11/7/208

## Citation
```
@Article{jimaging11070208,
  AUTHOR = {Özakar, Rüstem and Gedikli, Eyüp},
  TITLE = {Hand Washing Gesture Recognition Using Synthetic Dataset},
  JOURNAL = {Journal of Imaging},
  VOLUME = {11},
  YEAR = {2025},
  NUMBER = {7},
  ARTICLE-NUMBER = {208},
  URL = {https://www.mdpi.com/2313-433X/11/7/208},
  DOI = {10.3390/jimaging11070208}
}
```

</details>

- [ ] **UWash Smartwatch Dataset** (`datasets/uwash-smartwatch`) — tags: imu, wearable, smartwatch, public, quality-assessment
<details>
<summary>Show details</summary>

# UWash Smartwatch Dataset

## Overview
Smartwatch IMU dataset for handwashing quality assessment with accelerometer, gyroscope, and magnetometer streams.

## Size & Scale
- **Participants**: Multiple (exact count in dataset)
- **Sessions**: Multiple per participant
- **Sensors**: 3-axis accelerometer, gyroscope, magnetometer
- **Total Events**: Multiple handwashing sessions per participant
- **Sampling Rate**: Variable (smartwatch-dependent)
- **Duration**: Variable per session (typically 20-60 seconds)

## Data Collection

### Hardware
- **Device**: Smartwatch with IMU sensors
- **Sensors**: 
  - Accelerometer (3-axis)
  - Gyroscope (3-axis)
  - Magnetometer (3-axis)
- **Placement**: Wrist-worn (dominant hand)
- **Recording**: Native smartwatch app

### Collection Protocol
- **Participants**: Volunteers wearing smartwatch
- **Instructions**: Follow WHO handwashing guidelines
- **Environment**: Various (home, lab, hospital)
- **Natural Usage**: Participants perform normal handwashing

## Annotations

### Quality Assessment Classes (10 classes)
- Focus on handwashing **quality**, not just step detection
- Quality scores based on WHO compliance
- Classes represent quality levels (e.g., 1-10 scale or discrete categories)
- Annotated by experts based on WHO guidelines

### Data Preprocessing
1. **decode_sensor_data.py**: Decode raw IMU from smartwatch
2. **shift_data.py**: Temporal alignment and synchronization
3. **augment_data.py**: Data augmentation for IMU signals

## Dataset Structure

### Raw Data Format
- **CSVs**: Raw sensor readings with timestamps
- **Columns**: timestamp, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, mag_x, mag_y, mag_z
- **Sampling**: Variable rate (smartwatch-dependent)

### Preprocessed Format
- **NumPy Arrays**: Segmented time windows
- **Train/Test Splits**: Separate files for training and evaluation
- **Normalization**: Preprocessed and normalized sensor values

### File Organization
```
uwash-smartwatch/
├── raw/
│   ├── participant1/
│   │   ├── session1.csv
│   │   └── ...
│   └── ...
├── preprocessed/
│   ├── train/
│   └── test/
└── labels.csv (quality scores)
```

## Sample Statistics

### IMU Characteristics
- **Accelerometer**: ±2g, ±4g, or ±8g range
- **Gyroscope**: ±250, ±500, or ±2000 dps range
- **Magnetometer**: ±4, ±8, ±12, or ±16 gauss range
- **Sampling Rate**: Typically 50-100 Hz

### Session Characteristics
- **Duration**: 20-60 seconds per session
- **Segments**: Windowed into fixed-length segments (64 or 128 samples)
- **Quality Distribution**: Balanced across quality levels (or natural distribution)

## Public Availability

### Download Access
- **Platform**: Google Drive (link in UWash README)
- **License**: Research use (cite paper)
- **Download Link**: https://drive.google.com/file/d/1ZRdRiwXp4xbFUWIIjIQ0OEK6gK0cwODN/view
- **Format**: Raw CSV + preprocessing scripts

### Download Instructions
```bash
# Download from Google Drive link
# Unzip Dataset_raw.zip
# Run preprocessing scripts as documented in UWash codebase
```

## Preprocessing Pipeline

### Step 1: Decode Sensor Data
```bash
python pre_validation/decode_sensor_data.py
# Decodes raw smartwatch sensor readings
```

### Step 2: Temporal Alignment
```bash
python pre_validation/shift_data.py
# Aligns sensor streams temporally
```

### Step 3: Data Augmentation
```bash
python pre_validation/augment_data.py
# Augments IMU data (rotation, scaling, noise)
```

## Usage Examples

### Load Raw Data
```python
import pandas as pd

# Load raw sensor data
df = pd.read_csv("participant1/session1.csv")

# Extract sensor values
acc = df[['acc_x', 'acc_y', 'acc_z']].values
gyro = df[['gyro_x', 'gyro_y', 'gyro_z']].values
mag = df[['mag_x', 'mag_y', 'mag_z']].values
```

### Load Preprocessed Data
```python
import numpy as np

# Load preprocessed segments
train_data = np.load("preprocessed/train/segments.npy")
train_labels = np.load("preprocessed/train/labels.npy")

# Shape: (num_samples, sequence_length, num_features)
print(train_data.shape)  # e.g., (1000, 64, 9)
```

## Key Features
- **Wearable-based**: No video/cameras required
- **Privacy-preserving**: IMU data only (no visual information)
- **Quality assessment**: Scores handwashing quality (not just detection)
- **Multi-sensor**: Acc + gyro + mag fusion
- **Public**: Available via Google Drive

## Limitations
- **Requires smartwatch**: Participants must wear device
- **Wrist-worn**: Limited to dominant hand
- **Quality scores**: Subjective (expert annotations)
- **No visual context**: Cannot see actual hand movements
- **Preprocessing required**: Raw data needs extensive preprocessing

## Related Datasets
- **wearPuck**: Multimodal (IMU + environmental sensors)
- **OCDetect**: IMU for compulsive handwashing detection
- **PSKUS/METC**: Vision-based (comparison benchmark)

## Applications
- **Quality assessment**: Automated WHO compliance scoring
- **Wearable systems**: Privacy-preserving monitoring
- **Training feedback**: Real-time coaching via smartwatch
- **Behavior change**: Long-term habit tracking
- **Sensor fusion**: Combine with other modalities

## Citation
```
UWash: You Can Wash Hands Better - Accurate Daily Handwashing Assessment with Smartwatches
Paper: http://arxiv.org/abs/2112.06657
Dataset: Google Drive (link in README)
```

</details>

- [ ] **wearPuck Multimodal Handwash Dataset** (`datasets/wearpuck`) — tags: imu, environmental, wearable, public, multimodal
<details>
<summary>Show details</summary>

# wearPuck Multimodal Handwash Dataset

## Overview
Open-source wrist-worn sensor dataset capturing 43 hand-washing events over ~10 hours of recordings (highly imbalanced) with IMU and environmental sensors.

## Size & Scale
- **Total Events**: 43 handwashing events
- **Total Recording Time**: ~10 hours
- **Highly Imbalanced**: Many more non-handwashing samples than handwashing
- **Participants**: Multiple (exact count in dataset)
- **Sensors**: IMU (acc/gyro) + environmental (humidity, temp, pressure)
- **Sampling Rate**: Variable (depends on sensor)

## Data Collection

### Hardware Platform
- **Device**: wearPuck (Espruino Puck.js + BME280)
- **Form Factor**: Wrist-worn prototype
- **Sensors**:
  - **IMU**: Accelerometer (3-axis), gyroscope (3-axis)
  - **Environmental**: BME280 (humidity, temperature, pressure)
- **Open Hardware**: Design files included (STL for encasing)

### Collection Protocol
- **Participants**: Volunteers wearing wearPuck
- **Duration**: ~10 hours continuous recording
- **Natural Behavior**: Participants perform normal activities
- **Events**: 43 handwashing events recorded
- **Highly Imbalanced**: Mostly non-handwashing data

### Firmware
- **Language**: JavaScript (Espruino)
- **Files**: 
  - `firmware/puckApp.js` - Main application
  - `firmware/puckBTService.js` - Bluetooth service
  - `firmware/beacon.js` - Beacon functionality
- **Recording**: Bluetooth streaming to smartphone/computer

## Annotations

### Event Labels
- **Binary**: Handwashing vs non-handwashing
- **Not WHO Steps**: Does not classify individual WHO movements
- **Event Detection**: Focus on presence of handwashing (not quality/steps)
- **Labels File**: `iWoar/labels.csv`

### Annotation Process
- **Manual**: Post-hoc labeling of handwashing events
- **Ground Truth**: Annotated by participants or observers
- **Event Boundaries**: Start/end timestamps for each handwashing event

## Dataset Structure

### File Organization
```
wearpuck/
├── data/
│   ├── participant1/
│   │   ├── imu.csv
│   │   ├── bme.csv (humidity, temp, pressure)
│   │   ├── timestamps.csv
│   │   └── ...
│   └── ...
├── iWoar/
│   ├── labels.csv (event annotations)
│   ├── experiments.py (ML pipeline)
│   └── modules/ (preprocessing, training)
└── firmware/ (Espruino JavaScript)
```

### Data Format
- **IMU CSV**: timestamp, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
- **BME CSV**: timestamp, humidity, temperature, pressure
- **Timestamps CSV**: Synchronization information
- **Labels CSV**: event_id, start_time, end_time, label (handwashing/not)

## Sample Statistics

### IMU Characteristics
- **Accelerometer**: Espruino Puck.js built-in
- **Gyroscope**: Espruino Puck.js built-in
- **Sampling Rate**: Variable (typically 10-50 Hz)

### Environmental Sensor Characteristics
- **Humidity**: BME280 (0-100% RH)
- **Temperature**: BME280 (-40 to +85°C)
- **Pressure**: BME280 (300-1100 hPa)
- **Sampling Rate**: Lower than IMU (typically 1-10 Hz)

### Event Statistics
- **Total Events**: 43 handwashing events
- **Event Duration**: Variable (typically 20-60 seconds)
- **Highly Imbalanced**: ~10 hours total, only 43 events (~1% handwashing)

## Public Availability

### Download Access
- **Platform**: GitHub repository (included in code/wearPuck)
- **License**: Open source (license not specified)
- **Format**: Raw CSV files
- **Download**: Clone repository

### Download Instructions
```bash
git clone https://github.com/kristofvl/wearPuck.git
cd wearPuck
# Data files included in iWoar/data/
```

## Preprocessing & Training

### ML Pipeline
- **Script**: `iWoar/experiments.py`
- **Preprocessing**: `iWoar/modules/prepare.py`
- **Training**: `iWoar/modules/run_ml.py`
- **Models**: Random Forest, SVM (scikit-learn)

### Running Experiments
```bash
cd iWoar
python experiments.py
# Results generated in folder
```

## Usage Examples

### Load IMU Data
```python
import pandas as pd

# Load IMU data
imu_df = pd.read_csv("data/participant1/imu.csv")

# Extract sensor values
acc = imu_df[['acc_x', 'acc_y', 'acc_z']].values
gyro = imu_df[['gyro_x', 'gyro_y', 'gyro_z']].values
```

### Load Environmental Data
```python
import pandas as pd

# Load BME280 data
bme_df = pd.read_csv("data/participant1/bme.csv")

# Extract sensor values
humidity = bme_df['humidity'].values
temperature = bme_df['temperature'].values
pressure = bme_df['pressure'].values
```

### Load Event Labels
```python
import pandas as pd

# Load event labels
labels_df = pd.read_csv("iWoar/labels.csv")

# Filter handwashing events
handwash_events = labels_df[labels_df['label'] == 'handwashing']
```

## Key Features
- **Multimodal**: IMU + environmental sensors
- **Open hardware**: Complete design files (Espruino + BME280)
- **Open source**: Code + firmware included
- **Environmental cues**: Humidity spike unique to handwashing
- **Privacy-preserving**: No video/images
- **Low-cost**: ~$40 hardware (Puck.js + BME280)

## Limitations
- **Highly imbalanced**: Only 43 events in ~10 hours (class imbalance)
- **Small event count**: Limited handwashing samples
- **Event detection only**: Does not classify WHO steps
- **No quality assessment**: Binary presence/absence only
- **Wearable dependency**: Requires custom hardware

## Related Datasets
- **UWash**: Smartwatch IMU (quality assessment)
- **OCDetect**: Smartwatch IMU (compulsive handwashing)
- **PSKUS/METC**: Vision-based (comparison)

## Applications
- **Event detection**: Binary handwashing detection
- **Sensor fusion**: IMU + environmental for robustness
- **Wearable systems**: Privacy-preserving monitoring
- **Humidity-based detection**: Novel environmental cue
- **Low-cost deployment**: Accessible hardware

## Research Publication
- **Conference**: iWoar 2024
- **Focus**: Multimodal wearable handwashing detection
- **Reproducibility**: Steps documented in README

## Citation
```
wearPuck: Open Source Sensing Platform for Handwashing Detection
Repository: https://github.com/kristofvl/wearPuck
Paper: iWoar 2024
```

</details>


## Models
- Staging area for trained weights and model cards.

## Evaluation
- Add benchmark summaries or result notebooks here.

## Ideas
- Roadmap (ideas/roadmap.md)
- Todo (ideas/todo.md)

## TODOs
- [ ] Data: download & checksum PSKUS, METC, Kaggle WHO6; wire fetch scripts into CI smoke check.
  - [ ] Add lightweight sample slices for quick tests (5 clips per class).
  - [ ] Normalize label maps across datasets (WHO6+Other).
- [ ] Models: train baseline MobileNetV2 (frames) and GRU (clips) on combined hospital+lab; log metrics to `evaluation/`.
  - [ ] Fine-tune two-stream (RGB+OF) with shadow augmentation recipes from 2024 paper.
  - [ ] Benchmark synthetic pretraining (r-ozakar) then finetune on real data.
- [ ] Tracking: implement step-duration tracker (start/end, per-step seconds, coverage) fed by classifier outputs.
  - [ ] Export JSON/CSV compliance report per video.
  - [ ] Add threshold configs (per-step seconds, total duration).
- [ ] UI/Pages: publish GitHub Pages summary and embed latest README; add demo GIFs once tracker exists.
  - [ ] Add model cards for each released checkpoint.
- [ ] Sensors: prototype fusion (wearPuck humidity + IMU) baseline and compare to vision-only triggers.

## Automation
- `scripts/build_readme.py` regenerates this README from folder metadata.
- `.github/workflows/build-readme.yml` runs the generator on each push and commits changes.
- `.github/workflows/pages.yml` builds GitHub Pages from the generated docs.

To add new assets, drop them in the appropriate folder with minimal metadata; the automation will refresh this page.
