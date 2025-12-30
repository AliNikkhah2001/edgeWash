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
