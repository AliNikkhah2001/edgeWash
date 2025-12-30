# METC Lab Handwashing Dataset

## Overview
Lab-collected WHO handwash recordings from the Medical Education Technology Center (Riga Stradins University) with multiple camera interfaces.

## Size & Scale
- **Total Videos**: Multiple sessions (exact count in summary.csv)
- **Camera Interfaces**: 3 different interfaces/setups
- **Participants**: Lab volunteers and students
- **Environment**: Controlled lab setting (Medical Education Technology Center)
- **Resolution**: Various (typically 640×480 or higher)
- **Frame Rate**: 30 FPS

## Data Collection

### Hardware Setup
- **Location**: Medical Education Technology Center, Riga Stradins University
- **Cameras**: Multiple camera interfaces (Interface_number_1, 2, 3)
- **Environment**: Controlled lab (consistent lighting, fixed camera angles)
- **Participants**: Students and volunteers (controlled sessions)

### Collection Protocol
- **Setting**: Lab-based (not real-world hospital)
- **Instructions**: Participants follow WHO guidelines
- **Supervision**: Controlled sessions with guidance
- **Quality**: Higher quality than real-world (better lighting, angles)

## Annotations

### Annotation Process
- **Tool**: Custom annotation tool (similar to PSKUS)
- **Annotators**: Medical professionals and students
- **Guidelines**: WHO guidelines (same as PSKUS)
- **Label Scheme**: Aligned with PSKUS (6 steps + "Other")

### Frame-Level Labels
Each frame annotated with:
- **Movement Code**: WHO step (1-6) or "Other" (0)
- **Frame Time**: Timestamp within video
- **Is Washing**: Binary flag

### WHO Movement Classes (6 + Other)
Same as PSKUS:
1. Palm to palm
2. Palm over dorsum with interlaced fingers
3. Palm to palm with fingers interlaced
4. Back of fingers to opposing palms
5. Rotational rubbing of thumb
6. Fingertips to palm
7. Other (non-washing, rinsing, faucet control)

## Dataset Structure

### File Organization
```
METC/
├── Interface_number_1/
│   ├── video1.mp4
│   ├── video1.json
│   └── ...
├── Interface_number_2/
├── Interface_number_3/
├── summary.csv (episode metadata)
└── statistics.csv (dataset stats)
```

### File Formats
- **Videos**: MP4 format
- **Annotations**: JSON (per video, same format as PSKUS)
- **Metadata**: CSV (summary.csv, statistics.csv)

## Sample Statistics

### Video Characteristics
- **Resolution**: Various (typically ≥640×480)
- **Frame Rate**: 30 FPS
- **Duration**: Variable (standard WHO duration ~20-40 seconds)
- **Quality**: Higher than hospital data (controlled environment)

### Camera Interfaces
- **Interface 1**: Specific camera setup/angle
- **Interface 2**: Alternative setup/angle
- **Interface 3**: Third setup/angle
- **Purpose**: Multi-view diversity for generalization

## Public Availability

### Download Access
- **Platform**: Zenodo (open repository)
- **License**: Open research use (cite paper)
- **Download Script**: `datasets/metc-lab/fetch.sh`

### Download Instructions
```bash
cd datasets/metc-lab
./fetch.sh
# Dataset will be downloaded from Zenodo and extracted
```

### Citation Requirement
```
METC Lab Handwashing Dataset
Medical Education Technology Center, Riga Stradins University
Available on Zenodo
```

## Usage Examples

### Frame Extraction
```python
import cv2

video = cv2.VideoCapture("Interface_number_1/video1.mp4")
while True:
    ret, frame = video.read()
    if not ret:
        break
    # Process frame...
```

### Load Annotations
```python
import json

with open("Interface_number_1/video1.json", "r") as f:
    annotations = json.load(f)

for anno in annotations:
    movement_code = anno["movement_code"]
```

## Key Features
- **Lab-based**: Controlled environment (consistent quality)
- **Multi-interface**: 3 camera setups for diversity
- **WHO-compliant**: Same label scheme as PSKUS
- **Public**: Available on Zenodo
- **Frame-level annotations**: Dense temporal labels

## Limitations
- **Lab setting**: Not real-world (may not generalize to hospitals)
- **Controlled**: Less natural behavior than hospital data
- **Smaller scale**: Fewer videos than PSKUS
- **Participant diversity**: Limited to lab volunteers

## Related Datasets
- **PSKUS Hospital**: Real-world hospital (larger, more diverse)
- **Kaggle WHO6**: Public, smaller, similar structure
- **Synthetic**: Large-scale, perfect annotations

## Applications
- **Training**: Controlled data for initial model training
- **Transfer learning**: Pre-train on METC, fine-tune on hospital
- **Ablation studies**: Compare lab vs hospital performance
- **Benchmarking**: Lab-to-wild generalization testing
