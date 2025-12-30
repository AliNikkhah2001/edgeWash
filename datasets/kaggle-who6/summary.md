# Kaggle Handwash (Resorted to WHO 6+1 Classes)

## Overview
Public subset of the Kaggle hand-wash videos, re-sorted into 7 folders to align with WHO steps (left/right merged; wrist/rinse in "Other").

## Size & Scale
- **Total Videos**: ~292 short video clips
- **Classes**: 7 folders (WHO 6 steps + "Other")
- **Duration**: Short clips (typically 5-15 seconds each)
- **Total Duration**: Variable (few minutes total)
- **Source**: Kaggle Hand Wash Dataset (public competition)

## Data Collection

### Original Source
- **Platform**: Kaggle (public dataset)
- **Original Structure**: 12 classes (left/right hand variants separate)
- **Resorting**: Reorganized to match WHO 6-step taxonomy
- **Purpose**: Align with PSKUS/METC label schemes

### Preprocessing
- **Left/Right Merge**: Combine left/right hand variants into single classes
- **Wrist/Rinse**: Moved to "Other" class (not core WHO movements)
- **Result**: 7 folders (6 WHO steps + 1 "Other")

## Annotations

### Class Structure (7 folders)
1. **Step 1**: Palm to palm
2. **Step 2**: Palm over dorsum with interlaced fingers
3. **Step 3**: Palm to palm with fingers interlaced
4. **Step 4**: Back of fingers to opposing palms
5. **Step 5**: Rotational rubbing of thumb
6. **Step 6**: Fingertips to palm
7. **Other**: Wrist washing, rinsing, non-core movements

### Resorting from Original
- **Original**: 12 classes (left/right variants separate)
- **Resorted**: 7 classes (left/right merged)
- **Rationale**: Match PSKUS/METC taxonomy (no left/right distinction)

## Dataset Structure

### File Organization
```
kaggle-who6/
├── step1/ (palm to palm)
├── step2/ (palm over dorsum)
├── step3/ (interlaced)
├── step4/ (back of fingers)
├── step5/ (thumb rub)
├── step6/ (fingertips)
└── other/ (wrist, rinse, etc.)
```

### File Formats
- **Videos**: Short clips (5-15 seconds)
- **Format**: MP4 or AVI
- **Resolution**: Variable (typically 640×480 or 1080p)
- **Frame Rate**: Variable (typically 30 FPS)

## Sample Statistics

### Video Characteristics
- **Total Clips**: ~292
- **Per Class**: ~40-50 clips per WHO step
- **Duration**: Short (5-15 seconds)
- **Quality**: Variable (Kaggle submissions)

### Environment
- **Setting**: Various (home, lab, portable sinks)
- **Lighting**: Variable (indoor, outdoor)
- **Backgrounds**: Diverse (different sinks, settings)

## Public Availability

### Download Access
- **Platform**: GitHub mirror + Kaggle
- **License**: Public (Kaggle open data)
- **Download Script**: `datasets/kaggle-who6/fetch.sh`
- **Mirror**: https://github.com/atiselsts/data/raw/master/kaggle-dataset-6classes.tar

### Download Instructions
```bash
cd datasets/kaggle-who6
./fetch.sh
# Dataset will be downloaded from GitHub mirror and extracted
```

## Usage Examples

### Load Videos
```python
import cv2
import os

# List videos in step1 folder
step1_videos = os.listdir("kaggle-who6/step1/")

# Load first video
video = cv2.VideoCapture(f"kaggle-who6/step1/{step1_videos[0]}")
while True:
    ret, frame = video.read()
    if not ret:
        break
    # Process frame...
```

### Training Split
```python
from sklearn.model_selection import train_test_split

# Split videos into train/val/test
all_videos = []
labels = []

for class_idx, class_name in enumerate(["step1", "step2", ..., "other"]):
    videos = os.listdir(f"kaggle-who6/{class_name}/")
    all_videos.extend([f"{class_name}/{v}" for v in videos])
    labels.extend([class_idx] * len(videos))

train_val, test = train_test_split(all_videos, test_size=0.1)
train, val = train_test_split(train_val, test_size=0.15)
```

## Key Features
- **Public**: Fully open (Kaggle + GitHub mirror)
- **Small**: ~292 clips (quick experiments)
- **WHO-aligned**: Resorted to match PSKUS/METC taxonomy
- **Diverse**: Various environments and settings
- **Short clips**: Easy to process (5-15 seconds each)

## Limitations
- **Small scale**: Only ~292 videos (much smaller than PSKUS)
- **No frame-level labels**: Only clip-level labels (entire clip = one class)
- **Variable quality**: Kaggle submissions (inconsistent)
- **No temporal annotations**: Cannot train temporal segmentation
- **Resorting artifacts**: Original 12-class structure lost

## Related Datasets
- **PSKUS Hospital**: Much larger (3,185 episodes), frame-level labels
- **METC Lab**: Lab-based, frame-level labels
- **Original Kaggle**: 12 classes (before resorting)

## Applications
- **Quick experiments**: Baseline models on small dataset
- **Ablation studies**: Compare small vs large dataset performance
- **Transfer learning**: Pre-train on Kaggle, fine-tune on PSKUS
- **Proof of concept**: Fast prototyping before large-scale training
- **Benchmarking**: Test model generalization across datasets

## Citation
```
Kaggle Hand Wash Dataset (resorted to WHO 6+1 classes)
Original: https://www.kaggle.com/realtimear/hand-wash-dataset
Resorted mirror: https://github.com/atiselsts/data/raw/master/kaggle-dataset-6classes.tar
```

## Download Size
- **Compressed**: ~Few hundred MB
- **Uncompressed**: ~1-2 GB (short clips)
