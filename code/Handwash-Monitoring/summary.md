# TensorFlow Handwash Monitoring Demo

Small TF1 + OpenCV webcam demo that classifies handwashing steps from single frames using a retrained graph.

## Scope
- **Code ownership:** minimal demo script; relies on an external TensorFlow retrained graph (not included).
- **Third-party:** TensorFlow 1.x graph format and OpenCV; no hosted SDK.
- **Task:** per-frame step classification with a simple step-progress counter.

## Models
- **Backbone:** TensorFlow retrain example (typically Inception-based ImageNet weights) exported to `.pb`.
- **Input:** single RGB frame captured from webcam (`photo.jpg`).
- **Temporal handling:** none; step completion uses a repeated-label counter (5 consecutive matches).

## Data
- **Dataset:** not bundled; labels expected for `No Hands` and `Step 2` ... `Step 7`.
- **Structure:** `tf_files/retrained_graph.pb` + `tf_files/retrained_labels.txt`.

## Training
- **Scripts:** not included; use TensorFlow `retrain.py` or equivalent to generate the graph.
- **Hyperparameters:** not specified in this repo.

## Running
1. Install dependencies (TensorFlow 1.x compatible build + OpenCV).
2. Place `retrained_graph.pb` and `retrained_labels.txt` in `tf_files/`.
3. Run `python dettolhandwash.py`.

## Notes
- **Outputs:** console prompts for step progression; no saved metrics.
- **Limitations:** relies on loop counts, not wall-clock timing.
