# Kaggle Handwash (Resorted to WHO 6+1 Classes)

Public subset of the Kaggle hand-wash videos re-sorted into 7 WHO-style classes.

## Overview
- **Size:** ~1.21 GB tarball; ~292 short videos.
- **Classes:** 7 (6 WHO steps + Other; left/right merged).
- **Modalities:** RGB video; clip-level labels only.

## Structure
- **Layout:** `kaggle-dataset-6classes/<class_name>/` with videos.
- **Splits:** none provided; use preprocessing scripts to build train/test splits.

## Availability
- **Access:** public (Kaggle derivative hosted on GitHub).
- **Download:** `./fetch.sh` (downloads to `datasets/kaggle-who6/raw/`).

## Notes
- **Used by:** `code/edgewash` and YOLO demos.
