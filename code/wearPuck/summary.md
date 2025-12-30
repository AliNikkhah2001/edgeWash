# wearPuck Multimodal Wearable Toolkit

## Overview
Firmware and data tooling for the wearPuck device capturing IMU and environmental signals for handwashing analytics.

## Code Structure
- **Type**: Self-contained codebase (hardware + firmware + ML pipeline)
- **Implementation**: Firmware (JavaScript for Espruino) + Python ML scripts
- **Code Included**: Yes - firmware, data processing, ML experiments
- **Dependencies**: Python (scikit-learn, pandas, numpy), Espruino Puck.js

## Hardware Platform
- **Device**: Espruino Puck.js (wrist-worn)
- **Additional Sensors**: BME280 (humidity, temperature, pressure)
- **Form Factor**: Wristband prototype
- **Open-source Hardware**: Design files included

## Model Architecture
- **Approach**: **Classical ML** (not deep learning)
- **Models**: Random Forest, SVM (scikit-learn)
- **Input**: Multimodal sensor fusion
  - IMU: Accelerometer, gyroscope
  - Environmental: Humidity, temperature, pressure
- **Focus**: Event detection (handwashing vs non-handwashing)

## Video/Temporal Handling
- **Modality**: **No video** - wearable sensors only
- **Temporal Model**: Classical ML on windowed features
- **3D Convolutions**: N/A (sensor data, not video)
- **Sequence Modeling**: Feature engineering on time windows
- **Event Detection**: Binary classification (handwashing event or not)

## Classes & WHO Steps
- **Focus**: **Binary event detection** (handwashing vs non-handwashing)
- **Not WHO step classification** - detects presence of handwashing only
- **Highly imbalanced**: More non-handwashing than handwashing events

## Datasets Used
- **wearPuck Multimodal Handwash Dataset**
  - **Size**: 43 handwashing events over ~10 hours
  - **Highly imbalanced**: Many more non-handwashing samples
  - **Sensors**: IMU (acc/gyro) + humidity/temp/pressure
  - **Format**: Raw CSV files from wearPuck device
  - **Public Availability**: Yes (open-source dataset)

## Training Details

### Data Collection
- **Firmware**: `firmware/puckApp.js`, `firmware/puckBTService.js`, `firmware/beacon.js`
- **Data Format**: CSV exports from wearPuck device
- **Preprocessing**: `read_data.py`, `merge_data.py`

### ML Experiments
- **Script**: `iWoar/experiments.py`
- **Preprocessing**: `iWoar/modules/prepare.py`
- **Training**: `iWoar/modules/run_ml.py`
- **Models**: Random Forest, SVM (scikit-learn)
- **Features**: Statistical features from sensor time windows

### Hyperparameters
- **Classical ML**: Standard scikit-learn defaults
- **Window Size**: Configurable in preprocessing
- **Feature Engineering**: Time-domain statistics

## Running Instructions

### 1. Clone Repository
```bash
git clone https://github.com/kristofvl/wearPuck.git
cd wearPuck
```

### 2. Run ML Experiments (on provided dataset)
```bash
cd iWoar
python experiments.py
```

## Key Features
- **Multimodal sensor fusion** (IMU + environmental)
- **Open-source hardware** (Espruino Puck.js + BME280)
- **Classical ML** (Random Forest, SVM)
- **Humidity spike detection** (unique environmental cue)
- **Event detection** (not step classification)
- **Low-cost wearable** (accessible hardware)
- **Complete pipeline** (firmware + data + ML)

## Limitations
- **Highly imbalanced dataset** (43 events in ~10 hours)
- **Event detection only** (not WHO step classification)
- **Classical ML** (no deep learning)
- **Limited dataset size**

## Availability
- **Code**: Open (license not specified)
- **Dataset**: Yes (included in repo as CSVs)
- **Hardware Design**: Yes (encasing STL files)
- **Firmware**: Yes (Espruino JavaScript)
- **External API**: No
