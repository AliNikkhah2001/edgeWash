# Vision-Based Hand Hygiene Monitoring with Depth Cameras (Stanford PAIR 2015)

## Overview
Privacy-focused hand hygiene monitoring system using overhead depth sensors near soap dispensers in hospital settings.

## Research Contributions
- **Privacy-preserving approach**: Uses depth-only cameras (no RGB) to protect patient/staff identity
- **Dispenser-use detection**: Automatic detection of hand presence and soap dispenser interaction
- **Hospital deployment**: Real-world validation in clinical environment
- **Lightweight CNN**: Demonstrated depth-only approach feasibility for resource-constrained deployment

## Problem Statement
- Hospital staff hand hygiene compliance monitoring requires automated systems
- Traditional RGB camera systems raise privacy concerns in clinical settings
- Need for automated detection of:
  - Hand presence near sink
  - Soap dispenser usage
  - Duration of handwashing events

## Methodology

### Data Collection
- **Setup**: Overhead depth cameras mounted above soap dispensers
- **Environment**: Hospital rooms (clinical setting)
- **Duration**: ~20 hours of depth video captured
- **Sensors**: Depth-only cameras (no RGB for privacy)
- **Capture angle**: Overhead view near dispensers

### Model Architecture
- **Type**: Lightweight CNN for depth frame classification
- **Input**: Single depth frames (per-frame classification)
- **Output**: Binary predictions (hand presence / dispenser use)
- **Temporal Handling**: No explicit sequence model
  - Frame-level predictions aggregated via event logic
  - Duration tracked externally
- **3D Convolutions**: No (standard 2D CNN on depth frames)

### Labeling & Annotations
- **Events**: Dispenser-use and hand-presence events
- **Ground Truth**: Manual annotations of key events
- **Focus**: Detection (not WHO step classification)

## Results
- **Accuracy**: High detection accuracy reported (exact metrics not in accessible portions)
- **Privacy**: Successfully demonstrated depth-only monitoring viability
- **Deployment**: Validated in real hospital environment

## Dataset Details
- **Name**: Stanford Depth Hand Hygiene Dataset (2015)
- **Size**: ~20 hours of depth video
- **Format**: Depth video frames from overhead cameras
- **Location**: Hospital near soap dispensers
- **Public Availability**: **No** - not publicly released
- **Privacy**: Depth-only (no RGB) for patient/staff anonymity

## Technical Structure

### Pipeline
1. **Capture**: Overhead depth cameras record continuously
2. **Frame Extraction**: Depth frames extracted from video
3. **CNN Inference**: Per-frame classification (hand presence / dispenser use)
4. **Event Aggregation**: Frame predictions aggregated into events
5. **Compliance Reporting**: Duration and frequency tracking

### Key Design Choices
- **Depth-only**: Privacy preservation over visual quality
- **Overhead mounting**: Optimal view of sink and dispenser area
- **Lightweight CNN**: Suitable for edge/embedded deployment
- **Frame-based**: Simple per-frame classification (no temporal modeling)

## Limitations
- **Dataset not public**: Cannot reproduce experiments
- **No temporal modeling**: Ignores motion patterns across frames
- **Dispenser-focused**: Limited to near-dispenser monitoring (not full WHO steps)
- **Depth-only**: May miss fine-grained hand movements
- **Event-level only**: Does not classify individual WHO movements

## Applications
- **Hospital compliance monitoring**: Automated hand hygiene tracking
- **Privacy-sensitive environments**: Healthcare, clinical settings
- **Dispenser usage analytics**: Frequency and duration tracking
- **Staff compliance auditing**: Objective measurement without human observers

## Related Work Comparison
- **vs. RGB systems**: Better privacy, but less visual detail
- **vs. wearable sensors**: No device required, but fixed location
- **vs. multi-step classifiers**: Simpler (presence/dispenser only), not WHO steps

## Availability
- **Paper**: PDF link in repo (access may be restricted)
- **Code**: Not publicly available
- **Dataset**: Not publicly available
- **Trained Weights**: Not provided
- **External API**: No

## Notes
- **Era**: Pre-deep learning boom (2015), uses early CNN approaches
- **Focus**: Privacy and deployment feasibility over fine-grained recognition
- **Impact**: Demonstrated depth-only viability for healthcare monitoring
