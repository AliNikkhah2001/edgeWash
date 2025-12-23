# PSKUS Hospital Handwashing Dataset

Real-world hospital videos (Latvia) with 3,185 hand-washing episodes annotated frame-by-frame using WHO movement codes. Includes `summary.csv`, `statistics.csv`, and per-frame labels (`is_washing`, `movement_code`, `frame_time`). Resolution 320x240 or 640x480 at 30 FPS.

## Download
- Zenodo: `https://zenodo.org/record/4537209/files/Handwashing_dataset.zip?download=1` (large; requires Zenodo login sometimes).
- Script: `./fetch.sh` (downloads zip to `datasets/pskus-hospital/raw/`).

## Notes
- Merge left/right variants into 6 WHO steps + "Other".
- Copy `statistics-with-locations.csv` from `code/edgewash/dataset-pskus/` into the extracted dataset for location-aware splits.
