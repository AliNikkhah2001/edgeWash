# OCDetect Wrist-Worn IMU Dataset

## Overview
Smartwatch IMU dataset for detecting compulsive handwashing behaviors (OCD) with accelerometer and gyroscope data.

## Size & Scale
- **Focus**: Obsessive-Compulsive Disorder (OCD) detection
- **Participants**: Multiple (including OCD patients and controls)
- **Sensors**: 3-axis accelerometer, gyroscope
- **Device**: Smartwatch (wrist-worn)
- **Sessions**: Multiple per participant
- **Duration**: Variable per session

## Data Collection

### Hardware
- **Device**: Commercial smartwatch with IMU
- **Sensors**:
  - Accelerometer (3-axis)
  - Gyroscope (3-axis)
- **Placement**: Wrist-worn (dominant hand)
- **Sampling Rate**: Variable (smartwatch-dependent, typically 50-100 Hz)

### Collection Protocol
- **Participants**: OCD patients + healthy controls
- **Environment**: Various (home, clinic)
- **Natural Behavior**: Participants perform daily activities
- **Focus**: Detecting compulsive handwashing patterns
- **Duration**: Extended monitoring periods

### Clinical Context
- **OCD**: Obsessive-Compulsive Disorder
- **Compulsive Handwashing**: Repetitive, excessive handwashing
- **Detection Goal**: Identify abnormal handwashing frequency/duration
- **Clinical Value**: Monitor symptom severity, treatment effectiveness

## Annotations

### Labels
- **Binary**: Compulsive handwashing vs normal handwashing
- **Frequency**: Number of handwashing events per day
- **Duration**: Length of each handwashing episode
- **Intensity**: Severity of compulsive behavior
- **Not WHO Steps**: Does not classify individual movements

### Annotation Process
- **Clinical Review**: Mental health professionals
- **Ground Truth**: Patient self-reports + observer logs
- **Criteria**: DSM-5 OCD diagnostic criteria
- **Validation**: Cross-reference with clinical assessments

## Dataset Structure

### File Organization
```
ocdetect-wrist/
├── participants/
│   ├── ocd_patient1/
│   │   ├── imu_data.csv
│   │   └── annotations.csv
│   ├── control1/
│   └── ...
└── metadata/
    └── participant_info.csv
```

### Data Format
- **IMU CSV**: timestamp, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
- **Annotations CSV**: event_id, start_time, end_time, label, severity
- **Metadata**: participant demographics, OCD diagnosis status

## Sample Statistics

### IMU Characteristics
- **Accelerometer**: ±2g, ±4g, or ±8g range
- **Gyroscope**: ±250, ±500, or ±2000 dps range
- **Sampling Rate**: 50-100 Hz
- **Sessions**: Multiple per participant (days/weeks)

### Participant Characteristics
- **OCD Patients**: Diagnosed with compulsive handwashing
- **Controls**: Healthy individuals (no OCD)
- **Age**: Variable (adults)
- **Gender**: Mixed

## Public Availability

### Access Status
- **PUBLIC**: Available for research
- **Platform**: Research repository or project website
- **License**: Research use (cite paper)
- **Format**: Raw IMU CSV + annotations

### Download Instructions
```bash
# Check project website or paper for download link
# Typically: Google Drive, Zenodo, or project GitHub
```

## Usage Examples

### Load IMU Data
```python
import pandas as pd

# Load IMU data
imu_df = pd.read_csv("participants/ocd_patient1/imu_data.csv")

# Extract sensor values
acc = imu_df[['acc_x', 'acc_y', 'acc_z']].values
gyro = imu_df[['gyro_x', 'gyro_y', 'gyro_z']].values
```

### Load Annotations
```python
import pandas as pd

# Load annotations
anno_df = pd.read_csv("participants/ocd_patient1/annotations.csv")

# Filter compulsive handwashing events
compulsive = anno_df[anno_df['label'] == 'compulsive']
```

## Key Features
- **Clinical focus**: OCD compulsive handwashing detection
- **Wearable-based**: Smartwatch IMU (privacy-preserving)
- **Long-term monitoring**: Extended sessions (days/weeks)
- **Binary classification**: Compulsive vs normal
- **Public**: Available for research

## Limitations
- **OCD-specific**: Limited to compulsive behavior detection
- **No WHO steps**: Does not classify individual movements
- **Smartwatch-dependent**: Requires wearable device
- **Binary only**: No fine-grained quality assessment
- **Clinical population**: May not generalize to general handwashing

## Related Datasets
- **UWash**: Smartwatch IMU, quality assessment, public
- **wearPuck**: Wrist-worn, multimodal, event detection
- **PSKUS/METC**: Vision-based, WHO steps, public

## Applications
- **OCD detection**: Identify compulsive handwashing patterns
- **Symptom monitoring**: Track OCD severity over time
- **Treatment evaluation**: Measure treatment effectiveness
- **Wearable systems**: Privacy-preserving mental health monitoring
- **Research**: Study compulsive behavior patterns

## Clinical Relevance

### OCD Handwashing
- **Symptom**: Excessive, repetitive handwashing
- **Frequency**: Multiple times per hour (vs few times per day)
- **Duration**: Prolonged episodes (>5 minutes)
- **Impact**: Skin damage, life disruption

### Detection Benefits
- **Objective monitoring**: Quantify symptom severity
- **Treatment tracking**: Measure progress over time
- **Early intervention**: Detect relapse patterns
- **Personalized care**: Tailor treatment to individual

## Research Paper
- **Focus**: Compulsive handwashing detection using wrist-worn IMU
- **Methods**: Machine learning on smartwatch accelerometer/gyroscope
- **Results**: High accuracy distinguishing compulsive vs normal
- **Applications**: Mental health monitoring, OCD treatment

## Citation
```
OCDetect: Compulsive Handwashing Detection via Wrist-Worn IMU
(Check project website or paper for full citation)
```

## Comparison with Other Wearable Datasets

### vs UWash
- **OCDetect**: OCD detection (compulsive behavior)
- **UWash**: Quality assessment (WHO compliance)
- **Both**: Smartwatch IMU, privacy-preserving

### vs wearPuck
- **OCDetect**: Clinical focus (OCD)
- **wearPuck**: General event detection
- **OCDetect**: Long-term monitoring
- **wearPuck**: Short sessions

## Access Information
- **Status**: PUBLIC (research use)
- **Download**: Check project website or paper for link
- **Alternative**: Use UWash (quality assessment) or wearPuck (event detection)
