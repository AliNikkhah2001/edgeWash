# wearPuck Multimodal Handwash Dataset (2024)

Wearable sensor dataset for handwashing event detection using IMU + environmental signals.

## Overview
- **Size:** ~10 hours of recordings; 43 handwashing events (reported by authors).
- **Modalities:** accelerometer/gyroscope + humidity, temperature, pressure.
- **Labels:** event-level annotations (handwash vs non-handwash).

## Structure
- **Recordings:** CSV streams such as `imu.csv`, `bme.csv`, `timestamps.csv`, `button.csv`, `beacon.csv`, `capacitive.csv`.
- **Labels:** `labels.csv` with start/end intervals; experiments expect `iWoar/data/*.csv`.

## Availability
- **Access:** raw dataset not bundled; collection scripts and labels are public in upstream repo.

## Notes
- **Use cases:** sensor fusion and event detection; no vision data.
