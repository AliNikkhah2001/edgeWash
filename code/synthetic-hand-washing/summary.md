# Synthetic Hand-Washing Data Generation

Synthetic dataset + training scripts for frame-level handwashing gesture recognition with InceptionV3, YOLOv8, and PointNet.

## Scope
- **Code ownership:** full training scripts; no hosted SDK.
- **Modalities:** synthetic RGB, depth, masks, and point clouds.
- **Task:** 8-class gesture classification (frame-level).

## Models
- **InceptionV3:** ImageNet pretrained; 96x96 input; Dropout 0.3; Dense 64; Adam lr 1e-4; 5-fold CV.
- **YOLOv8n / YOLOv8n-seg:** Ultralytics pretrained; `epochs=5`, `flipud=0`, `fliplr=0`.
- **PointNet:** 512-point clouds; Adam lr 1e-4; 5-fold CV; epochs=3 in script.

## Data
- **Dataset:** synthetic Blender-rendered frames (see `datasets/synthetic-blender-rozakar`).
- **Structure:** scripts expect pickled RGB/point cloud data in `rgb pickle/` and `pcd pickle/`.

## Training
- **Scripts:** `inception/train_inception.py`, `yolo/train.py`, `yolo/train-seg.py`, `pointnet/pointnet_train.py`.
- **Hyperparameters:** set in scripts; adjust epochs and batch sizes as needed.

## Running
1. Download and extract the dataset to repo root.
2. Create pickles via `create_rgb_pickle.py` / `create_pcd_pickle.py`.
3. Run the training scripts for each model.

## Notes
- **Temporal handling:** none; all models are frame-level.
