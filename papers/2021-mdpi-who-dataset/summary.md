# Hand-Washing Video Dataset Annotated According to the WHO Guidelines (Data 2021)

## Overview
Dataset paper presenting the PSKUS Hospital Handwashing Dataset with 3,185 annotated episodes following WHO guidelines.

## Research Contributions
- **Large-scale hospital dataset**: 3,185 real-world handwashing episodes
- **Frame-level annotations**: Detailed WHO movement codes per frame
- **Double annotation**: 1,094 videos annotated by two people for reliability
- **Public release**: Available on Zenodo for research use
- **Statistics and metadata**: Comprehensive summary.csv and statistics.csv files

## Problem Statement
- Lack of publicly available handwashing datasets with WHO annotations
- Need for large-scale real-world data for training and benchmarking
- Frame-level annotations required for temporal modeling
- Hospital environment data critical for clinical deployment validation

## Dataset Collection

### Data Collection System
- **Locations**: 9 sites at Pauls Stradins Clinical University Hospital (Latvia)
- **Cameras**: AirLive POE 100CAM, Axis M3046V IP cameras
- **Control**: Raspberry Pi 4 single-board computers
- **Placement**: Overhead above sinks
- **Trigger**: Motion detection >10 seconds
- **Duration**: 3 months continuous collection
- **Total Captured**: 32,471 videos

### Annotations
- **Annotated Videos**: 2,293 files total
  - 1,199 annotated once
  - 1,094 annotated twice (double annotation for reliability)
- **Total Episodes**: 3,185 handwashing episodes
- **Annotation Tool**: Custom Python + OpenCV GUI
- **Annotators**: Infectious disease specialists, medical professionals, RSU students
- **Guidelines**: Developed with local infectious disease specialists

## Dataset Structure

### File Organization
- **Video Files**: MP4/AVI format
- **JSON Files**: Frame-level annotations per video
- **CSV Files**: 
  - `summary.csv`: Episode-level metadata
  - `statistics.csv`: Dataset-wide statistics

### Annotations Per Frame
- **Movement Code**: WHO step being performed
- **Frame Time**: Timestamp within video
- **Is Washing**: Binary flag (washing vs non-washing)
- **Additional Flags**: Ring, watch, lacquered nails (inappropriate for medical staff)

### WHO Movement Classes (7 total)
1. Palm to palm
2. Palm over dorsum with fingers interlaced
3. Palm to palm with fingers interlaced
4. Back of fingers to opposing palms
5. Rotational rubbing of thumb
6. Fingertips to palm
7. Turning off faucet with paper towel

Plus "Other" class for non-washing actions.

## Dataset Statistics

### Video Characteristics
- **Resolution**: 320×240 or 640×480 pixels
- **Frame Rate**: 30 FPS
- **Total Episodes**: 3,185
- **Average Episode Duration**: Variable (typically 20-60 seconds)

### Annotation Quality
- **Double Annotation**: 1,094 videos (59% of annotated set)
- **Purpose**: Inter-annotator reliability assessment
- **Benefit**: Quality control and ambiguity identification

### Distribution
- **Multiple Locations**: 9 hospital sites
- **Diverse Participants**: Hospital staff (doctors, nurses, technicians)
- **Real-world Conditions**: Variable lighting, camera angles, hand positions

## Data Collection Protocol

### Hardware Setup
- **Camera**: Overhead mounting above sink
- **Storage**: Micro SD card on Raspberry Pi
- **Network**: PoE switch for power and connectivity
- **Recording**: Triggered by motion (>10 seconds)

### Annotation Workflow
1. **Video Collection**: Monthly retrieval from hospital sites
2. **Central Storage**: Transfer to annotation server
3. **Annotation**: Custom GUI for frame-level labeling
4. **Quality Control**: Double annotation for subset
5. **Export**: JSON + CSV for each video

## Usage and Applications

### Training Deep Learning Models
- **Frame-level classification**: Per-frame WHO step recognition
- **Temporal modeling**: LSTM/GRU on frame sequences
- **Transfer learning**: Pre-train on PSKUS, fine-tune on smaller datasets

### Benchmarking
- **Standard test set**: Enable fair comparison across methods
- **Evaluation metrics**: Accuracy, precision, recall, F1 per class
- **Generalization**: Hospital environment (challenging real-world conditions)

### Research Directions
- **Action recognition**: WHO step classification
- **Temporal segmentation**: Detect step boundaries
- **Quality assessment**: Compliance with WHO duration requirements
- **Multi-task learning**: Detect inappropriate accessories (rings, watches)

## Technical Details

### Data Format
- **Video Codecs**: H.264 (MP4), MJPEG (AVI)
- **JSON Schema**: 
  ```json
  {
    "frame_number": int,
    "frame_time": float,
    "movement_code": int,
    "is_washing": bool
  }
  ```
- **CSV Columns**: Episode ID, duration, movement counts, accessories

### Download and Access
- **Platform**: Zenodo (public repository)
- **License**: Open for research use (restrictions on commercial use)
- **Size**: ~Several GB (videos + annotations)
- **Download**: Automated via fetch.sh script in repo

## Validation and Quality

### Inter-Annotator Agreement
- **Method**: Cohen's kappa on double-annotated subset
- **Purpose**: Measure annotation consistency
- **Findings**: High agreement on clear movements, lower on transitions

### Challenges Identified
- **Ambiguous Transitions**: Hand position changes between steps
- **Occlusions**: Hands overlap, fingers hidden
- **Camera Angle**: Overhead view misses some movements
- **Motion Blur**: Fast hand movements

## Limitations
- **Single Hospital**: Limited to one clinical environment (Latvia)
- **Overhead Cameras**: Fixed angle may miss fine-grained movements
- **Resolution**: 320×240 or 640×480 (not HD)
- **Annotations**: Some ambiguity at step transitions
- **No Depth**: RGB only (no depth data for privacy-preserving)

## Related Datasets Comparison
- **vs. Kaggle**: Larger scale, hospital environment, WHO-compliant
- **vs. METC**: Real hospital vs lab, more participants
- **vs. Egocentric**: Fixed camera vs wearable
- **vs. Synthetic**: Real-world variation vs perfect annotations

## Future Directions
- **Multi-camera**: Additional angles for better coverage
- **Higher Resolution**: HD cameras for finer details
- **Depth Cameras**: Privacy-preserving depth-only variant
- **Automated Pre-annotation**: Speed up annotation with semi-supervised methods

## Availability
- **Paper**: Data journal, MDPI, 2021
- **Dataset**: Yes (Zenodo, public)
- **Download Script**: `datasets/pskus-hospital/fetch.sh`
- **License**: Open research use
- **Citation Required**: Yes (see below)

## Citation
```
M. Lulla, A. Rutkovskis, A. Slavinska, A. Vilde, A. Gromova, M. Ivanovs, 
A. Skadins, R. Kadikis and A. Elsts
"Hand Washing Video Dataset Annotated According to the World Health Organization's Handwashing Guidelines"
Data, 6(4), p.38, 2021
```

## Download Instructions
```bash
cd datasets/pskus-hospital
./fetch.sh
# Dataset will be downloaded from Zenodo and extracted
```
