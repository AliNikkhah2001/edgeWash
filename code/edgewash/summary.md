# EdgeWash WHO Movement Classifier

Keras/TensorFlow pipeline for WHO step classification on handwashing videos with single-frame, temporal, and two-stream models.

## Scope
- **Code ownership:** full preprocessing + training scripts; no hosted SDK.
- **Modalities:** RGB frames; optional optical flow.
- **Task:** 7-class WHO step classification (6 steps + Other).

## Models
- **Backbones:** MobileNetV2 by default (via `HANDWASH_NN`); optional extra dense layers.
- **Temporal handling:** TimeDistributed CNN + GRU (`*-classify-videos.py`) with `HANDWASH_NUM_FRAMES` (default 5).
- **Two-stream:** RGB + optical flow (`*-classify-merged-network.py`).

## Data
- **Datasets:** PSKUS, METC, Kaggle WHO6 subset.
- **Structure:** dataset-specific `dataset-*/separate-frames.py` scripts extract frames and split train/test; `calculate-optical-flow.py` builds flow stacks.

## Training
- **Scripts:** `kaggle-classify-frames.py`, `pskus-classify-videos.py`, `rsu-metc-classify-merged-network.py`, etc.
- **Hyperparameters (env vars):** `HANDWASH_NN`, `HANDWASH_NUM_LAYERS`, `HANDWASH_NUM_EPOCHS` (default 20), `HANDWASH_NUM_FRAMES` (default 5), `HANDWASH_EXTRA_LAYERS`, `HANDWASH_PRETRAINED_MODEL`, `HANDWASH_SUFFIX`.

## Running
1. Install deps + `ffmpeg` (`pip install -r requirements.txt`).
2. Run dataset `get-and-preprocess-dataset.sh` scripts.
3. (Optional) run `calculate-optical-flow.py`.
4. Train via the dataset-specific classify script.

## Notes
- **Outputs:** saved Keras models + metrics; `convert-model-to-tflite.py` supports edge export.
- **Availability:** datasets not bundled; weights not included.
