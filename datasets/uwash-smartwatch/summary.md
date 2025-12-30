# UWash Smartwatch Dataset

## Overview
Smartwatch IMU dataset for handwashing quality assessment with accelerometer, gyroscope, and magnetometer streams.

## Size & Scale
- **Participants**: Multiple (exact count in dataset)
- **Sessions**: Multiple per participant
- **Sensors**: 3-axis accelerometer, gyroscope, magnetometer
- **Total Events**: Multiple handwashing sessions per participant
- **Sampling Rate**: Variable (smartwatch-dependent)
- **Duration**: Variable per session (typically 20-60 seconds)

## Data Collection

### Hardware
- **Device**: Smartwatch with IMU sensors
- **Sensors**: 
  - Accelerometer (3-axis)
  - Gyroscope (3-axis)
  - Magnetometer (3-axis)
- **Placement**: Wrist-worn (dominant hand)
- **Recording**: Native smartwatch app

### Collection Protocol
- **Participants**: Volunteers wearing smartwatch
- **Instructions**: Follow WHO handwashing guidelines
- **Environment**: Various (home, lab, hospital)
- **Natural Usage**: Participants perform normal handwashing

## Annotations

### Quality Assessment Classes (10 classes)
- Focus on handwashing **quality**, not just step detection
- Quality scores based on WHO compliance
- Classes represent quality levels (e.g., 1-10 scale or discrete categories)
- Annotated by experts based on WHO guidelines

### Data Preprocessing
1. **decode_sensor_data.py**: Decode raw IMU from smartwatch
2. **shift_data.py**: Temporal alignment and synchronization
3. **augment_data.py**: Data augmentation for IMU signals

## Dataset Structure

### Raw Data Format
- **CSVs**: Raw sensor readings with timestamps
- **Columns**: timestamp, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, mag_x, mag_y, mag_z
- **Sampling**: Variable rate (smartwatch-dependent)

### Preprocessed Format
- **NumPy Arrays**: Segmented time windows
- **Train/Test Splits**: Separate files for training and evaluation
- **Normalization**: Preprocessed and normalized sensor values

### File Organization
```
uwash-smartwatch/
├── raw/
│   ├── participant1/
│   │   ├── session1.csv
│   │   └── ...
│   └── ...
├── preprocessed/
│   ├── train/
│   └── test/
└── labels.csv (quality scores)
```

## Sample Statistics

### IMU Characteristics
- **Accelerometer**: ±2g, ±4g, or ±8g range
- **Gyroscope**: ±250, ±500, or ±2000 dps range
- **Magnetometer**: ±4, ±8, ±12, or ±16 gauss range
- **Sampling Rate**: Typically 50-100 Hz

### Session Characteristics
- **Duration**: 20-60 seconds per session
- **Segments**: Windowed into fixed-length segments (64 or 128 samples)
- **Quality Distribution**: Balanced across quality levels (or natural distribution)

## Public Availability

### Download Access
- **Platform**: Google Drive (link in UWash README)
- **License**: Research use (cite paper)
- **Download Link**: https://drive.google.com/file/d/1ZRdRiwXp4xbFUWIIjIQ0OEK6gK0cwODN/view
- **Format**: Raw CSV + preprocessing scripts

### Download Instructions
```bash
# Download from Google Drive link
# Unzip Dataset_raw.zip
# Run preprocessing scripts as documented in UWash codebase
```

## Preprocessing Pipeline

### Step 1: Decode Sensor Data
```bash
python pre_validation/decode_sensor_data.py
# Decodes raw smartwatch sensor readings
```

### Step 2: Temporal Alignment
```bash
python pre_validation/shift_data.py
# Aligns sensor streams temporally
```

### Step 3: Data Augmentation
```bash
python pre_validation/augment_data.py
# Augments IMU data (rotation, scaling, noise)
```

## Usage Examples

### Load Raw Data
```python
import pandas as pd

# Load raw sensor data
df = pd.read_csv("participant1/session1.csv")

# Extract sensor values
acc = df[['acc_x', 'acc_y', 'acc_z']].values
gyro = df[['gyro_x', 'gyro_y', 'gyro_z']].values
mag = df[['mag_x', 'mag_y', 'mag_z']].values
```

### Load Preprocessed Data
```python
import numpy as np

# Load preprocessed segments
train_data = np.load("preprocessed/train/segments.npy")
train_labels = np.load("preprocessed/train/labels.npy")

# Shape: (num_samples, sequence_length, num_features)
print(train_data.shape)  # e.g., (1000, 64, 9)
```

## Key Features
- **Wearable-based**: No video/cameras required
- **Privacy-preserving**: IMU data only (no visual information)
- **Quality assessment**: Scores handwashing quality (not just detection)
- **Multi-sensor**: Acc + gyro + mag fusion
- **Public**: Available via Google Drive

## Limitations
- **Requires smartwatch**: Participants must wear device
- **Wrist-worn**: Limited to dominant hand
- **Quality scores**: Subjective (expert annotations)
- **No visual context**: Cannot see actual hand movements
- **Preprocessing required**: Raw data needs extensive preprocessing

## Related Datasets
- **wearPuck**: Multimodal (IMU + environmental sensors)
- **OCDetect**: IMU for compulsive handwashing detection
- **PSKUS/METC**: Vision-based (comparison benchmark)

## Applications
- **Quality assessment**: Automated WHO compliance scoring
- **Wearable systems**: Privacy-preserving monitoring
- **Training feedback**: Real-time coaching via smartwatch
- **Behavior change**: Long-term habit tracking
- **Sensor fusion**: Combine with other modalities

## Citation
```
UWash: You Can Wash Hands Better - Accurate Daily Handwashing Assessment with Smartwatches
Paper: http://arxiv.org/abs/2112.06657
Dataset: Google Drive (link in README)
```
