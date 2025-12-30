# PSKUS Hospital Handwashing Dataset

## Overview
Real-world hospital videos with 3,185 hand-washing episodes annotated frame-by-frame using WHO movement codes.

## Size & Scale
- **Total Episodes**: 3,185 annotated handwashing episodes
- **Total Videos Captured**: 32,471 (subset of 3,185 annotated)
- **Annotated Videos**: 2,293 files
  - 1,199 annotated once
  - 1,094 annotated twice (double annotation for reliability)
- **Total Annotations**: 6,690 (including double annotations)
- **Total Washing Time**: 83,804 seconds
- **WHO Movements Time**: 27,517 seconds (movements 1-7)

## Data Collection

### Hardware Setup
- **Cameras**: AirLive POE 100CAM, Axis M3046V IP cameras
- **Controller**: Raspberry Pi 4 single-board computers
- **Storage**: Micro SD cards (local storage)
- **Network**: Netgear 5-Port PoE Gigabit Ethernet switch
- **Placement**: Overhead above sinks (9 sites)

### Collection Protocol
- **Location**: Pauls Stradins Clinical University Hospital (Latvia)
- **Duration**: 3 months continuous deployment
- **Trigger**: Motion detection >10 seconds
- **Recording**: Automatic (motion-triggered)
- **Resolution**: 640×480 or 320×240 pixels
- **Frame Rate**: 30 FPS
- **Storage**: Local SD cards, monthly manual collection

### Deployment Sites
- **Total Sites**: 9 hospital locations
- **Environment**: Real clinical setting (sinks in hospital)
- **Participants**: Hospital staff (doctors, nurses, technicians)
- **Natural Behavior**: Staff unaware of specific recording times

## Annotations

### Annotation Process
- **Tool**: Custom Python + OpenCV GUI
- **Annotators**: 
  - Infectious disease specialists
  - Medical professionals
  - Riga Stradins University students
- **Guidelines**: Developed with local infectious disease specialists
- **Quality Control**: 1,094 videos double-annotated

### Frame-Level Labels
Each frame annotated with:
- **Movement Code**: WHO step (1-7) or "Other" (0)
- **Frame Time**: Timestamp within video
- **Is Washing**: Binary flag (washing vs non-washing)
- **Additional Flags**: Ring, watch, lacquered nails (inappropriate for medical staff)

### WHO Movement Classes (7 + Other)
1. **Palm to palm** rubbing
2. **Palm over dorsum** with interlaced fingers
3. **Palm to palm** with fingers interlaced
4. **Back of fingers** to opposing palms
5. **Rotational rubbing** of thumb
6. **Fingertips** to palm (circular motion)
7. **Turning off faucet** with paper towel
8. **Other** (non-washing actions, rinsing, etc.)

## Dataset Structure

### File Organization
```
PSKUS/
├── DataSet1/
│   ├── video1.mp4
│   ├── video1.json (frame-level annotations)
│   └── ...
├── DataSet2/
├── ...
├── summary.csv (episode metadata)
├── statistics.csv (dataset-wide stats)
└── statistics-with-locations.csv (location-based splits)
```

### File Formats
- **Videos**: MP4 or AVI (H.264/MJPEG codecs)
- **Annotations**: JSON (per video)
  ```json
  {
    "frame_number": int,
    "frame_time": float,
    "movement_code": int,
    "is_washing": bool
  }
  ```
- **Metadata**: CSV (summary.csv, statistics.csv)

### CSV Files
- **summary.csv**: Episode-level metadata (ID, duration, movement counts)
- **statistics.csv**: Dataset-wide statistics (class distributions, durations)
- **statistics-with-locations.csv**: Location-based splits (for train/test)

## Sample Statistics

### Video Characteristics
- **Resolution**: 320×240 or 640×480 pixels
- **Frame Rate**: 30 FPS
- **Duration**: Variable (typically 20-60 seconds per episode)
- **Total Duration**: 83,804 seconds (~23.3 hours)

### Class Distribution
- **WHO Movements**: Detailed counts in statistics.csv
- **Balanced**: Reasonable coverage of all WHO steps
- **Real-world**: Natural distribution (not artificially balanced)

### Annotation Quality
- **Double Annotation**: 1,094 videos (59% of annotated set)
- **Inter-Annotator Agreement**: High (Cohen's kappa reported in paper)
- **Ambiguities**: Identified at step transitions

## Public Availability

### Download Access
- **Platform**: Zenodo (open repository)
- **License**: Open research use (cite paper)
- **Size**: ~Several GB (compressed)
- **Download Script**: `datasets/pskus-hospital/fetch.sh`

### Download Instructions
```bash
cd datasets/pskus-hospital
./fetch.sh
# Dataset will be downloaded from Zenodo and extracted
```

### Citation Requirement
```
M. Lulla, A. Rutkovskis, A. Slavinska, A. Vilde, A. Gromova, M. Ivanovs, 
A. Skadins, R. Kadikis and A. Elsts
"Hand Washing Video Dataset Annotated According to the World Health Organization's Handwashing Guidelines"
Data, 6(4), p.38, 2021
```

## Usage Examples

### Training Data Split
- **Recommended**: Use `statistics-with-locations.csv` for location-based splits
- **Purpose**: Ensure train/test videos from different camera locations
- **Benefit**: Better generalization testing

### Frame Extraction
```python
# Extract frames from videos
import cv2

video = cv2.VideoCapture("DataSet1/video1.mp4")
while True:
    ret, frame = video.read()
    if not ret:
        break
    # Process frame...
```

### Load Annotations
```python
import json

with open("DataSet1/video1.json", "r") as f:
    annotations = json.load(f)

for anno in annotations:
    frame_num = anno["frame_number"]
    movement_code = anno["movement_code"]
    is_washing = anno["is_washing"]
```

## Key Features
- **Real-world data**: Hospital setting (not lab)
- **Frame-level annotations**: Dense temporal labels
- **Large-scale**: 3,185 episodes
- **Double annotation**: 1,094 videos for reliability
- **Public**: Available on Zenodo
- **WHO-compliant**: Follows official guidelines

## Limitations
- **Single hospital**: Limited to one environment (Latvia)
- **Overhead cameras**: Fixed angle (may miss some movements)
- **Resolution**: 320×240 or 640×480 (not HD)
- **Annotation ambiguity**: Some transitions unclear
- **No depth**: RGB only (no depth/privacy-preserving)

## Related Datasets
- **METC Lab Dataset**: Lab-based, similar structure
- **Kaggle WHO6**: Public, smaller, resorted to WHO taxonomy
- **HHA300**: Quality scores, not public
- **Class23**: Open-room, not public

## Applications
- **Deep learning**: Train CNNs, LSTMs, transformers
- **Benchmarking**: Standard test set for WHO step recognition
- **Transfer learning**: Pre-train on PSKUS, fine-tune on smaller datasets
- **Temporal segmentation**: Detect step boundaries
- **Quality assessment**: Compliance with WHO guidelines
