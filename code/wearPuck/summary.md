# wearPuck Multimodal Wearable Toolkit

Hardware + data pipeline for a wrist-worn sensor device; includes classical ML experiments for handwashing event detection.

## Scope
- **Code ownership:** firmware + data collection + experiments; no hosted SDK.
- **Modalities:** IMU (acc/gyro), humidity, temperature, pressure, beacon/button signals.
- **Task:** handwash vs non-handwash event classification.

## Models
- **Algorithms:** RandomForest and SVM baselines (`iWoar/modules/run_ml.py`).
- **Temporal handling:** fixed windows (125/250 samples) with summary statistics; no deep sequence model.

## Data
- **Dataset:** not bundled; expected CSV recordings in `iWoar/data/`.
- **Structure:** collection script writes `imu.csv`, `bme.csv`, `timestamps.csv`, `button.csv`, `beacon.csv`, `capacitive.csv`; `labels.csv` defines segments.

## Training
- **Script:** `iWoar/experiments.py` → `modules/run_ml.py`.
- **Hyperparameters:** `n_estimators=250`, `window_size` 125 or 250, 5 repetitions, RandomOverSampler; leave-one-out CV.

## Running
1. Collect data with `read_data.py` or place CSVs in `iWoar/data/`.
2. Run `cd iWoar && python experiments.py`.

## Notes
- **Outputs:** `results_loso.csv` and `results_personalized.csv` are saved.
