# wearPuck Multimodal Handwash Dataset

## Overview
Open-source wrist-worn sensor dataset capturing 43 hand-washing events over ~10 hours of recordings (highly imbalanced) with IMU and environmental sensors.

## Size & Scale
- **Total Events**: 43 handwashing events
- **Total Recording Time**: ~10 hours
- **Highly Imbalanced**: Many more non-handwashing samples than handwashing
- **Participants**: Multiple (exact count in dataset)
- **Sensors**: IMU (acc/gyro) + environmental (humidity, temp, pressure)
- **Sampling Rate**: Variable (depends on sensor)

## Data Collection

### Hardware Platform
- **Device**: wearPuck (Espruino Puck.js + BME280)
- **Form Factor**: Wrist-worn prototype
- **Sensors**:
  - **IMU**: Accelerometer (3-axis), gyroscope (3-axis)
  - **Environmental**: BME280 (humidity, temperature, pressure)
- **Open Hardware**: Design files included (STL for encasing)

### Collection Protocol
- **Participants**: Volunteers wearing wearPuck
- **Duration**: ~10 hours continuous recording
- **Natural Behavior**: Participants perform normal activities
- **Events**: 43 handwashing events recorded
- **Highly Imbalanced**: Mostly non-handwashing data

### Firmware
- **Language**: JavaScript (Espruino)
- **Files**: 
  - `firmware/puckApp.js` - Main application
  - `firmware/puckBTService.js` - Bluetooth service
  - `firmware/beacon.js` - Beacon functionality
- **Recording**: Bluetooth streaming to smartphone/computer

## Annotations

### Event Labels
- **Binary**: Handwashing vs non-handwashing
- **Not WHO Steps**: Does not classify individual WHO movements
- **Event Detection**: Focus on presence of handwashing (not quality/steps)
- **Labels File**: `iWoar/labels.csv`

### Annotation Process
- **Manual**: Post-hoc labeling of handwashing events
- **Ground Truth**: Annotated by participants or observers
- **Event Boundaries**: Start/end timestamps for each handwashing event

## Dataset Structure

### File Organization
```
wearpuck/
├── data/
│   ├── participant1/
│   │   ├── imu.csv
│   │   ├── bme.csv (humidity, temp, pressure)
│   │   ├── timestamps.csv
│   │   └── ...
│   └── ...
├── iWoar/
│   ├── labels.csv (event annotations)
│   ├── experiments.py (ML pipeline)
│   └── modules/ (preprocessing, training)
└── firmware/ (Espruino JavaScript)
```

### Data Format
- **IMU CSV**: timestamp, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
- **BME CSV**: timestamp, humidity, temperature, pressure
- **Timestamps CSV**: Synchronization information
- **Labels CSV**: event_id, start_time, end_time, label (handwashing/not)

## Sample Statistics

### IMU Characteristics
- **Accelerometer**: Espruino Puck.js built-in
- **Gyroscope**: Espruino Puck.js built-in
- **Sampling Rate**: Variable (typically 10-50 Hz)

### Environmental Sensor Characteristics
- **Humidity**: BME280 (0-100% RH)
- **Temperature**: BME280 (-40 to +85°C)
- **Pressure**: BME280 (300-1100 hPa)
- **Sampling Rate**: Lower than IMU (typically 1-10 Hz)

### Event Statistics
- **Total Events**: 43 handwashing events
- **Event Duration**: Variable (typically 20-60 seconds)
- **Highly Imbalanced**: ~10 hours total, only 43 events (~1% handwashing)

## Public Availability

### Download Access
- **Platform**: GitHub repository (included in code/wearPuck)
- **License**: Open source (license not specified)
- **Format**: Raw CSV files
- **Download**: Clone repository

### Download Instructions
```bash
git clone https://github.com/kristofvl/wearPuck.git
cd wearPuck
# Data files included in iWoar/data/
```

## Preprocessing & Training

### ML Pipeline
- **Script**: `iWoar/experiments.py`
- **Preprocessing**: `iWoar/modules/prepare.py`
- **Training**: `iWoar/modules/run_ml.py`
- **Models**: Random Forest, SVM (scikit-learn)

### Running Experiments
```bash
cd iWoar
python experiments.py
# Results generated in folder
```

## Usage Examples

### Load IMU Data
```python
import pandas as pd

# Load IMU data
imu_df = pd.read_csv("data/participant1/imu.csv")

# Extract sensor values
acc = imu_df[['acc_x', 'acc_y', 'acc_z']].values
gyro = imu_df[['gyro_x', 'gyro_y', 'gyro_z']].values
```

### Load Environmental Data
```python
import pandas as pd

# Load BME280 data
bme_df = pd.read_csv("data/participant1/bme.csv")

# Extract sensor values
humidity = bme_df['humidity'].values
temperature = bme_df['temperature'].values
pressure = bme_df['pressure'].values
```

### Load Event Labels
```python
import pandas as pd

# Load event labels
labels_df = pd.read_csv("iWoar/labels.csv")

# Filter handwashing events
handwash_events = labels_df[labels_df['label'] == 'handwashing']
```

## Key Features
- **Multimodal**: IMU + environmental sensors
- **Open hardware**: Complete design files (Espruino + BME280)
- **Open source**: Code + firmware included
- **Environmental cues**: Humidity spike unique to handwashing
- **Privacy-preserving**: No video/images
- **Low-cost**: ~$40 hardware (Puck.js + BME280)

## Limitations
- **Highly imbalanced**: Only 43 events in ~10 hours (class imbalance)
- **Small event count**: Limited handwashing samples
- **Event detection only**: Does not classify WHO steps
- **No quality assessment**: Binary presence/absence only
- **Wearable dependency**: Requires custom hardware

## Related Datasets
- **UWash**: Smartwatch IMU (quality assessment)
- **OCDetect**: Smartwatch IMU (compulsive handwashing)
- **PSKUS/METC**: Vision-based (comparison)

## Applications
- **Event detection**: Binary handwashing detection
- **Sensor fusion**: IMU + environmental for robustness
- **Wearable systems**: Privacy-preserving monitoring
- **Humidity-based detection**: Novel environmental cue
- **Low-cost deployment**: Accessible hardware

## Research Publication
- **Conference**: iWoar 2024
- **Focus**: Multimodal wearable handwashing detection
- **Reproducibility**: Steps documented in README

## Citation
```
wearPuck: Open Source Sensing Platform for Handwashing Detection
Repository: https://github.com/kristofvl/wearPuck
Paper: iWoar 2024
```
