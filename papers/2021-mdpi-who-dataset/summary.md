# Hand-Washing Video Dataset Annotated According to the WHO Guidelines (Data 2021)

Introduces the PSKUS hospital dataset with frame-level WHO step annotations and baseline CNN results.

## Data
- **Episodes:** 3,185 videos; 6,690 annotations.
- **Capture:** 30 FPS; 320x240 and 640x480 resolutions.
- **Duration:** 83,804 s total washing time; 27,517 s for movements 1-7.
- **Structure:** `DataSet*` folders + per-video CSV/JSON labels + `summary.csv` and `statistics.csv`.

## Method
- **Baseline:** MobileNetV2 pretrained on ImageNet, trained on extracted frames.
- **Temporal handling:** none; per-frame classification.

## Training
- **Hyperparameters:** 10 epochs, Adam lr 0.008, categorical loss; 10% test / 90% train+val split.

## Results
- **Accuracy:** 0.7511 frame accuracy on the test split.

## Availability
- **Data:** public on Zenodo (PSKUS dataset).
- **Code/weights:** scripts mentioned, weights not provided; no API.
