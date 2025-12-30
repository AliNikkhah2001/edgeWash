# UWash Smartwatch Handwashing Assessment

PyTorch pipeline for handwashing quality assessment from smartwatch IMU time-series with multiple train/eval protocols.

## Scope
- **Code ownership:** full preprocessing + training code; no hosted SDK.
- **Modalities:** accelerometer + gyroscope (magnetometer present in raw data).
- **Task:** 10-class sequence classification for handwashing quality/steps.

## Models
- **Backbones:** 1D ResNet (`resnet1d.py`), MobileNetV2/V3 (`mobilenet.py`), UWasher SPP-UNet, UTFormer.
- **Input:** fixed-length IMU segments (`seq_len` 64 or 128) per sensor stream.
- **Temporal handling:** sequence modeling via 1D CNNs/transformer; no video.

## Data
- **Dataset:** `Dataset_raw.zip` (Google Drive) + preprocessed splits.
- **Preprocessing:** `decode_sensor_data.py` → `shift_data.py` → `augment_data.py`.
- **Protocols:** normal, location-independent, and user-independent splits.

## Training
- **Scripts:** `UWasher/train_eval/normal_64.py`, `normal_128.py`, `li_64.py`, `ui_64.py`, plus eval scripts.
- **Key hyperparameters:** epochs 300, batch size 4096, init_lr 1e-3, momentum 0.99, milestones [200,300,400,500,600,700,800], lr_down_ratio 0.1, loss `CEL` or `focal`.
- **Model config:** 10 classes, 2 sensors, 3 axes (`ModelConfig.py`).

## Running
1. Download `Dataset_raw.zip` and update `base_path` in `pre_validation/*.py`.
2. Run preprocessing scripts in order.
3. Update dataset path in `UWasher/data/DatasetConfig.py` (missing in this clone; see upstream repo).
4. Run a training script, e.g., `python UWasher/train_eval/normal_64.py`.

## Notes
- **Weights:** not bundled.
- **Missing file:** `UWasher/data/DatasetConfig.py` is referenced but absent here.
