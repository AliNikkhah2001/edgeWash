# PSKUS Hospital Handwashing Dataset

Large hospital dataset with 3,185 annotated handwashing episodes (30 FPS; 320x240 or 640x480).

## Overview
- **Size:** 13 zip files totaling ~18.4 GB.
- **Episodes:** 3,185 videos; 6,690 annotations; 83,804 s total washing time.
- **Modalities:** RGB hospital videos.
- **Labels:** frame-level `is_washing` + `movement_code` with timestamps.

## Structure
- **Top-level:** `DataSet1`..`DataSet11`, `summary.csv`, `statistics.csv`, per-video CSV/JSON labels.
- **Splits:** not provided; use preprocessing scripts to create train/val/test.

## Availability
- **Access:** public on Zenodo.
- **Download:** `./fetch.sh` (downloads to `datasets/pskus-hospital/raw/`).

## Notes
- **Extra:** copy `statistics-with-locations.csv` from `code/edgewash/dataset-pskus/` for location-aware splits.
