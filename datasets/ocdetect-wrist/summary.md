# OCDetect Wrist Motion Dataset (2023)

Real-world smartwatch dataset for handwashing event detection using wrist motion data.

## Overview
- **Size:** `OCDetect_dataset.zip` ~31.6 GB (Zenodo).
- **Participants:** 22 participants over 28 days; ~3,000 handwashing events.
- **Modalities:** wrist IMU time-series (accelerometer + gyroscope).
- **Labels:** event-level handwashing annotations (no WHO step labels).

## Structure
- **Packaging:** single ZIP from Zenodo; internal folder structure defined by release.
- **Splits:** not provided; users create train/test windows.

## Availability
- **Access:** public on Zenodo.
- **Download:** `./fetch.sh` (downloads to `datasets/ocdetect-wrist/raw/`).

## Notes
- **Use cases:** event detection, hygiene frequency modeling.
