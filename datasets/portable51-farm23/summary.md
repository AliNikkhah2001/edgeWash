# Portable51 & Farm23 Shadow Augmentation Datasets

## Overview
Two shadow robustness datasets (Portable51, Farm23) for testing handwashing models under challenging shadow conditions.

## Size & Scale
- **Portable51**: 51 videos (portable sink, controlled shadows)
- **Farm23**: 23 videos (farm setting, natural shadows)
- **Total**: 74 videos across 2 environments
- **Purpose**: Shadow robustness evaluation
- **Annotation**: WHO step labels with shadow characteristics

## Data Collection

### Portable51 Dataset
- **Environment**: Portable sink in controlled setting
- **Shadow Conditions**: Artificial shadows (controlled)
- **Videos**: 51 recordings
- **Participants**: Lab volunteers
- **Camera**: Fixed overhead position
- **Lighting**: Variable (simulating different shadow conditions)

### Farm23 Dataset
- **Environment**: Farm setting (outdoor sink)
- **Shadow Conditions**: Natural outdoor shadows
- **Videos**: 23 recordings
- **Participants**: Farm workers
- **Camera**: Fixed position above outdoor sink
- **Lighting**: Natural sunlight (time-of-day variations)

## Annotations

### WHO Step Labels
Standard WHO 6-step taxonomy:
1. Palm to palm
2. Palm over dorsum with interlaced fingers
3. Palm to palm with fingers interlaced
4. Back of fingers to opposing palms
5. Rotational rubbing of thumb
6. Fingertips to palm
7. Other (non-washing, rinsing)

### Shadow Characteristics
- **Shadow Intensity**: Quantified shadow strength
- **Shadow Coverage**: Percentage of hands occluded
- **Shadow Type**: Artificial vs natural
- **Challenge Level**: Easy, medium, hard (based on occlusion)

## Dataset Structure

### File Organization
```
portable51-farm23/
├── portable51/
│   ├── videos/
│   │   ├── video001.mp4
│   │   └── ...
│   └── annotations/
│       └── labels.csv
├── farm23/
│   ├── videos/
│   │   ├── video001.mp4
│   │   └── ...
│   └── annotations/
│       └── labels.csv
└── shadow_analysis/
    └── shadow_metrics.csv
```

### File Formats
- **Videos**: MP4 format
- **Annotations**: CSV (frame-level WHO step labels)
- **Shadow Metrics**: CSV (shadow characteristics per frame)
- **Resolution**: Variable (typically 640×480 or 1080p)
- **Frame Rate**: 30 FPS

## Sample Statistics

### Portable51 Characteristics
- **Total Videos**: 51
- **Duration**: Variable per video (20-60 seconds)
- **Shadow Type**: Artificial (controlled)
- **Lighting**: Variable (simulated shadow conditions)

### Farm23 Characteristics
- **Total Videos**: 23
- **Duration**: Variable per video (20-60 seconds)
- **Shadow Type**: Natural (outdoor)
- **Lighting**: Sunlight (time-of-day variations)

## Public Availability

### Access Status
- **NOT PUBLIC**: Datasets not publicly released
- **Reason**: Research-specific, limited distribution
- **Request**: May be available upon request to authors
- **Institution**: Research group (papers/2024-mmsp-shadow-augmentation)

### Citation Requirement
```
Portable51 & Farm23 Shadow Augmentation Datasets
Paper: papers/2024-mmsp-shadow-augmentation
Contact authors for access
```

## Usage Notes

### Shadow Robustness Testing
- **Purpose**: Evaluate model performance under shadow occlusion
- **Benchmark**: Test trained models on shadow-augmented data
- **Ablation**: Compare performance with/without shadow augmentation
- **Domain**: Portable51 (controlled) vs Farm23 (natural)

### Research Applications
- **Robustness**: Test shadow resilience
- **Augmentation**: Validate shadow augmentation techniques
- **Domain adaptation**: Indoor vs outdoor shadows
- **Generalization**: Test cross-domain performance

## Key Features
- **Shadow-focused**: Specifically designed for shadow robustness
- **Dual environments**: Portable sink (controlled) + farm (natural)
- **WHO annotations**: Frame-level step labels
- **Shadow metrics**: Quantified shadow characteristics
- **Challenging**: Tests model robustness to occlusion

## Limitations
- **NOT PUBLIC**: Not available for download
- **Small scale**: Only 74 videos total
- **Shadow-specific**: Limited to shadow conditions
- **Limited diversity**: 2 environments only
- **Research-only**: Not general-purpose benchmark

## Related Datasets
- **PSKUS Hospital**: Larger, public, indoor (minimal shadows)
- **METC Lab**: Lab-based, controlled (minimal shadows)
- **Class23**: Open-room, variable lighting
- **Stanford Depth**: Depth-based (less sensitive to shadows)

## Applications
- **Shadow robustness testing**: Evaluate model performance under occlusion
- **Augmentation validation**: Test shadow augmentation techniques
- **Domain adaptation**: Indoor vs outdoor generalization
- **Benchmarking**: Shadow-specific performance metrics
- **Research**: Study shadow effects on WHO step recognition

## Research Paper
- **Title**: "Shadow Augmentation for Robust Hand Hygiene Recognition"
- **Year**: 2024
- **Conference**: MMSP 2024
- **Focus**: Shadow robustness and augmentation techniques
- **Paper**: papers/2024-mmsp-shadow-augmentation/paper.pdf

## Shadow Augmentation Techniques

### Tested Methods
- **CutMix**: Random rectangular shadow regions
- **Shadow synthesis**: Realistic shadow generation
- **Occlusion augmentation**: Simulated hand occlusion
- **Performance**: Improved model robustness

### Results
- **Baseline**: Poor performance on shadow-heavy videos
- **With augmentation**: Significant improvement in shadow conditions
- **Generalization**: Better cross-domain performance

## Access Information
- **Status**: NOT PUBLIC
- **Request**: Contact authors via paper (2024-mmsp-shadow-augmentation)
- **Alternative**: Use PSKUS/METC (public, indoor lighting) or generate synthetic shadows
