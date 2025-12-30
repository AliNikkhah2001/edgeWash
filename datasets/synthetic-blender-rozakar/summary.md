# Synthetic Hand-Washing Gesture Dataset (Özakar & Gedikli, 2025)

## Overview
CGI-rendered dataset with 96,000 frames (64 minutes) across 8 WHO-inspired gestures, 4 characters, and 4 environments (RGB, depth, masks).

## Size & Scale
- **Total Frames**: 96,000 frames
- **Total Duration**: 64 minutes (25 FPS)
- **Characters**: 4 diverse 3D hand models
- **Environments**: 4 realistic bathroom settings
- **Gestures**: 8 WHO-inspired handwashing movements
- **Modalities**: RGB, depth, depth-isolated, hand segmentation masks

## Data Generation

### Rendering Pipeline
- **Tool**: Blender 3D rendering engine
- **Resolution**: 960×540 pixels
- **Frame Rate**: 25 FPS
- **Rendering**: Cycles renderer (realistic lighting)

### Character Variations
- **Count**: 4 diverse character models
- **Diversity**: Varied skin tones, hand sizes, appearances
- **Animation**: Keyframe animation for 8 gestures per character

### Environment Variations
- **Count**: 4 realistic bathroom scenes
- **Lighting**: Different lighting conditions per environment
- **Backgrounds**: Varied sinks, walls, fixtures
- **Purpose**: Domain diversity for generalization

## Annotations

### Gesture Classes (8 WHO-Inspired)
1. **Palm to palm** rubbing
2. **Right palm over left dorsum** with interlaced fingers
3. **Left palm over right dorsum** with interlaced fingers
4. **Palm to palm** with fingers interlaced
5. **Back of fingers** to opposing palms
6. **Rotational rubbing of right thumb**
7. **Rotational rubbing of left thumb**
8. **Fingertips to palm** (circular motion)

### Perfect Annotations
- **Automatic**: Generated during rendering (no manual labeling)
- **Frame-level**: Every frame labeled with gesture class
- **Multi-modal**: RGB, depth, and masks perfectly aligned
- **No errors**: CGI ensures 100% label accuracy

## Dataset Structure

### File Organization
```
synthetic-blender-rozakar/
├── character1/
│   ├── environment1/
│   │   ├── gesture1/
│   │   │   ├── rgb/
│   │   │   ├── depth/
│   │   │   ├── depth_isolated/
│   │   │   └── masks/
│   │   └── ...
│   └── ...
├── character2/
└── ...
```

### Modalities
1. **RGB Images**: Standard color frames (960×540)
2. **Depth Maps**: Per-pixel depth (normalized 0-1)
3. **Depth-Isolated**: Depth with background removed
4. **Hand Segmentation Masks**: Binary masks (hands = white, background = black)

### File Formats
- **Images**: PNG (lossless)
- **Depth**: PNG (grayscale, normalized)
- **Masks**: PNG (binary)
- **Organization**: Folders by character/environment/gesture

## Sample Statistics

### Combinations
- **Total**: 4 characters × 4 environments × 8 gestures = 128 unique combinations
- **Frames per Gesture**: Variable (typically few hundred frames)
- **Total Frames**: 96,000

### Distribution
- **Balanced**: Equal frames per gesture (12,000 frames per gesture)
- **Systematic**: All character/environment combinations covered

## Public Availability

### Download Access
- **Platform**: Google Drive (5 download links)
- **License**: CC BY (Attribution)
- **Size**: ~Several GB (compressed)
- **Download Links** (in README):
  - https://drive.google.com/file/d/1EW3JQvElcuXzawxEMRkA8YXwK_Ipiv-p/view
  - https://drive.google.com/file/d/163TsrDe4q5KTQGCv90JRYFkCs7AGxFip/view
  - https://drive.google.com/file/d/1GxyTYfSodumH78NbjWdmbjm8JP8AOkAY/view
  - https://drive.google.com/file/d/1IoRsgBBr8qoC3HO-vEr6E7K4UZ6ku6-1/view
  - https://drive.google.com/file/d/1svCYnwDazy5FN1DYSgqbGscvDKL_YnID/view

### Download Instructions
```bash
# Download all 5 parts from Google Drive links
# Extract to dataset folder
# Organize by character/environment/gesture
```

## Pre-trained Models

### Included Weights
- **Inception-V3**: RGB image classifier (TensorFlow/Keras)
- **YOLOv8n**: Detection and segmentation (Ultralytics)
- **PointNet**: Point cloud classifier (PyTorch)
- **Training Scripts**: Provided in upstream repo

## Usage Examples

### Load RGB Images
```python
import cv2
import os

# List RGB images for gesture 1, character 1, environment 1
rgb_path = "character1/environment1/gesture1/rgb/"
rgb_images = sorted(os.listdir(rgb_path))

# Load first image
img = cv2.imread(f"{rgb_path}/{rgb_images[0]}")
```

### Load Depth Maps
```python
import cv2

# Load depth map (grayscale PNG)
depth = cv2.imread("character1/environment1/gesture1/depth/frame001.png", cv2.IMREAD_GRAYSCALE)

# Normalize to 0-1 range
depth_normalized = depth.astype(float) / 255.0
```

### Load Segmentation Masks
```python
import cv2

# Load binary mask
mask = cv2.imread("character1/environment1/gesture1/masks/frame001.png", cv2.IMREAD_GRAYSCALE)

# Threshold to binary (hands = 255, background = 0)
mask_binary = (mask > 127).astype(np.uint8) * 255
```

## Key Features
- **Large-scale**: 96,000 frames (much larger than real datasets)
- **Perfect annotations**: No labeling errors (CGI)
- **Multi-modal**: RGB, depth, masks (all aligned)
- **Systematic variations**: All character/environment combinations
- **Public**: CC BY license (fully open)
- **Pre-trained models**: Baseline weights provided

## Limitations
- **Synthetic-to-real gap**: Domain shift (CGI vs real hands)
- **8 gestures only**: Not exact WHO 6-step taxonomy
- **No temporal modeling**: Frame-level only (no videos)
- **Limited real validation**: Few real-world test samples
- **CGI artifacts**: Rendering may not capture all nuances

## Related Datasets
- **PSKUS/METC**: Real-world, frame-level labels
- **Kaggle WHO6**: Real, smaller, clip-level
- **HHA300**: Real, quality scores, not public

## Applications
- **Pre-training**: Bootstrap models before fine-tuning on real data
- **Data augmentation**: Mix synthetic with real for robustness
- **Domain adaptation**: Study synthetic-to-real transfer
- **Ablation studies**: Controlled experiments (vary characters, environments)
- **Annotation cost reduction**: Free perfect labels

## Research Paper
- **Title**: "Hand Washing Gesture Recognition Using Synthetic Dataset"
- **Authors**: Rüstem Özakar, Eyüp Gedikli
- **Journal**: Journal of Imaging, MDPI, 2025
- **DOI**: 10.3390/jimaging11070208
- **URL**: https://www.mdpi.com/2313-433X/11/7/208

## Citation
```
@Article{jimaging11070208,
  AUTHOR = {Özakar, Rüstem and Gedikli, Eyüp},
  TITLE = {Hand Washing Gesture Recognition Using Synthetic Dataset},
  JOURNAL = {Journal of Imaging},
  VOLUME = {11},
  YEAR = {2025},
  NUMBER = {7},
  ARTICLE-NUMBER = {208},
  URL = {https://www.mdpi.com/2313-433X/11/7/208},
  DOI = {10.3390/jimaging11070208}
}
```
