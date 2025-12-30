# HandWash Multimodal Surgical Rub Recognition

PyTorch project for classifying hand-rub steps from video clips with CNN-LSTM/ConvLSTM backbones plus a demo UI.

## Scope
- **Code ownership:** full training/eval/predict scripts; UI included; no hosted SDK.
- **Modalities:** RGB video clips.
- **Task:** 12-class step recognition (left/right variants of WHO steps).

## Models
- **Backbones:** ConvLSTM, CNN-LSTM with AlexNet, ResNet50, or custom CNN.
- **Temporal handling:** sequences of frames (`num_frames`) with LSTM/ConvLSTM; per-clip classification.

## Data
- **Dataset:** Kaggle Hand Wash Dataset; preprocessed to NumPy (`dataset.zip`).
- **Structure:** `HandWashDataset/train|val|test/Step_*` videos converted to `.npy` clips.

## Training
- **Scripts:** `train.py`, `evaluate.py`, `predict.py`.
- **Hyperparameters (defaults):** epochs 50, batch 32, num_frames 10, lr 1e-3, betas 0.9/0.9, weight_decay 1e-5, step_size 1, gamma 0.9; `--arch` selects `convlstm|alexnet|resnet50|custom`.
- **Weights:** pretrained `alexnet_128.pt` available via Google Cloud.

## Running
1. Download and unzip `dataset.zip` to repo root.
2. (Optional) download pretrained weights into `save_weights/`.
3. Train with `python train.py --arch alexnet`.
4. Evaluate with `python evaluate.py --checkpoint <path> --arch <arch>`.

## Notes
- **UI:** `handwashUI/` includes a Streamlit/React demo; requires Node + Python.
