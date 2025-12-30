# WashWise Real-Time Step Tracker

Webcam demo that uses a hosted Roboflow YOLOv8 workflow to classify handwashing steps and track time per step.

## Scope
- **Code ownership:** local UI/logic only; relies on Roboflow Inference API (third-party SDK).
- **Modalities:** live RGB webcam frames.
- **Task:** per-frame step classification + time accumulation.

## Models
- **Backbone:** Roboflow-hosted YOLOv8 workflow (weights are not local).
- **Input:** webcam frames via `roboflow-inference` client.
- **Temporal handling:** no sequence model; step timer accumulates per-frame predictions.

## Data
- **Dataset:** managed in Roboflow; not included in this repo.
- **Classes:** 8 handwashing steps (as defined in the deployed workflow).

## Training
- **Scripts:** none local; training is done on Roboflow platform.
- **Hyperparameters:** not in repo.

## Running
1. Install deps: `pip install -r requirements.txt`.
2. Copy `.env.example` to `.env` and set `API_KEY`, `WORKSPACE_NAME`, `WORKFLOW_ID`.
3. Run `python main.py` (requires internet + Roboflow account).

## Notes
- **Output:** OpenCV windows with step labels and progress bars.
- **Limitations:** no offline inference without exporting the Roboflow model.
