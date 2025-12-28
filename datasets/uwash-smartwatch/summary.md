# UWash Smartwatch Dataset (2023)

Smartwatch IMU dataset for handwashing quality assessment. Multiple sessions per participant with accelerometer/gyroscope/magnetometer streams; processed via provided scripts into train/test splits for 10 classes.

- Data type: wearable IMU time-series (acc/gyro/mag).
- Labels: handwashing quality/step-related classes (10-class splits in scripts).
- Availability: data public via Google Drive; preprocessing/training code in `code/UWash`; weights not included.

## Download
- Repo: `https://github.com/aiotgroup/UWash` (scripts + Google Drive data link in README).

## Notes
- Models target on-device efficiency; default configs use 300 epochs, batch size 4096, and learning rate 0.001 with milestone schedule.
