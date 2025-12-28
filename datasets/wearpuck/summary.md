# wearPuck Multimodal Handwash Dataset (2024)

Open-source wrist-worn sensor dataset capturing 43 hand-washing events over ~10 hours of recordings (highly imbalanced). Sensors: IMU (acc/gyro) + humidity/temperature/pressure. Collected with the wearPuck device; provides raw CSVs and preprocessing scripts.

- Data type: IMU + environmental sensor time-series.
- Labels: event-level hand-washing annotations across ~43 events.
- Availability: data and preprocessing scripts public in the upstream repo; no trained models/weights provided.

## Download
- Repo: `https://github.com/kristofvl/wearPuck` (data folder + preprocessing).

## Notes
- Good for sensor fusion experiments; humidity spikes are strong hand-wash cues. No vision data included.
