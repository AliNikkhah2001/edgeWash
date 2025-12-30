# METC Lab Handwashing Dataset

Lab-collected WHO handwashing videos from RSU METC with multi-camera interfaces and frame-level labels.

## Overview
- **Size:** 3 zip files totaling ~2.12 GB.
- **Cameras:** Interface_number_1/2/3.
- **Modalities:** RGB video in lab environment.
- **Labels:** per-frame WHO step codes (6 steps + Other).

## Structure
- **Top-level:** `Interface_number_*/` directories plus `summary.csv` and `statistics.csv`.
- **Annotations:** per-frame CSV/JSON files aligned to PSKUS label scheme.

## Availability
- **Access:** public on Zenodo.
- **Download:** `./fetch.sh` (downloads to `datasets/metc-lab/raw/`).

## Notes
- **Use with:** `code/edgewash/dataset-metc` preprocessing scripts.
