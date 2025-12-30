# Hand-Wash Compliance Detection with YOLO

YOLOv5-based detector for handwashing step detection and compliance tracking from video frames.

## Scope
- **Code ownership:** minimal repo + Colab notebook; relies on Ultralytics YOLOv5 (third-party).
- **Modalities:** RGB video frames (images).
- **Task:** object detection of 7 WHO steps, plus compliance/timing logic.

## Models
- **Backbone:** YOLOv5 (PyTorch).
- **Input:** 640x640 images with YOLO-format labels.
- **Temporal handling:** none in model; time/compliance computed by aggregating frame detections.

## Data
- **Dataset:** Kaggle Hand Wash Dataset; annotated in YOLO format.
- **Counts:** 707 labeled frames (567 train, 140 val) across 7 classes.

## Training
- **Scripts:** Colab notebook (external).
- **Hyperparameters:** epochs 100, batch size 16, input 640x640; reported mAP ~0.996.

## Running
1. Install YOLOv5 dependencies from Ultralytics.
2. Prepare YOLO-formatted dataset as in README.
3. Run training in the Colab notebook and export weights.

## Notes
- **Weights:** not included.
