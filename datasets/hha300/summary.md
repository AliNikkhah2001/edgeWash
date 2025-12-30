# HHA300 Hand Hygiene Assessment Dataset

## Overview
Hospital hand hygiene dataset with 300 videos and quality scores (0-10 scale) based on WHO compliance.

## Size & Scale
- **Total Videos**: 300 videos
- **Environment**: Hospital setting
- **Quality Scores**: 0-10 scale (WHO compliance)
- **Participants**: Hospital staff (healthcare workers)
- **Duration**: Variable per video (typically 20-60 seconds)
- **Resolution**: Variable (typically 640×480 or higher)
- **Frame Rate**: 30 FPS

## Data Collection

### Hardware Setup
- **Cameras**: Fixed cameras above sinks
- **Placement**: Overhead or side-view in hospitals
- **Environment**: Real clinical setting
- **Recording**: Continuous or motion-triggered

### Collection Protocol
- **Location**: Hospital hand hygiene stations
- **Participants**: Healthcare workers (doctors, nurses)
- **Natural Behavior**: Staff performing routine handwashing
- **Instructions**: Follow standard hospital protocols
- **Privacy**: Limited identifiable information

## Annotations

### Quality Assessment
- **Scale**: 0-10 quality scores
- **Basis**: WHO compliance (7 steps + duration)
- **Annotators**: Infectious disease experts
- **Criteria**: 
  - Completion of all WHO steps
  - Adequate duration (20+ seconds)
  - Technique quality
  - Coverage of all hand surfaces

### WHO Compliance Factors
- **Step completion**: All 7 WHO movements performed
- **Duration**: Minimum 20 seconds recommended
- **Technique**: Proper execution of each movement
- **Coverage**: All hand surfaces cleaned

### Annotation Process
- **Expert review**: Infectious disease specialists
- **Guidelines**: WHO hand hygiene guidelines
- **Scoring**: Holistic quality assessment (not frame-level)
- **Consistency**: Multiple reviewers for reliability

## Dataset Structure

### File Organization
```
hha300/
├── videos/
│   ├── video001.mp4
│   ├── video002.mp4
│   └── ...
└── annotations/
    └── quality_scores.csv
```

### Quality Scores CSV
- **Columns**: video_id, quality_score, duration, notes
- **Format**: CSV with video-level quality scores
- **Range**: 0 (poor) to 10 (perfect WHO compliance)

## Sample Statistics

### Video Characteristics
- **Total**: 300 videos
- **Duration**: Variable (20-60 seconds typical)
- **Quality**: Real hospital conditions (variable lighting)
- **Participants**: Healthcare workers

### Quality Distribution
- **Range**: 0-10 scale
- **Distribution**: Variable (natural hospital data)
- **High scores**: Perfect WHO compliance (rare)
- **Low scores**: Missing steps or poor technique

## Public Availability

### Access Status
- **NOT PUBLIC**: Dataset not publicly released
- **Reason**: Privacy concerns (hospital staff)
- **Request**: May be available upon request to authors
- **Institution**: Hospital research group

### Citation Requirement
```
HHA300 Hand Hygiene Assessment Dataset
Contact authors for access (paper reference TBD)
```

## Usage Notes

### Research Applications
- **Quality prediction**: Train models to predict compliance scores
- **Technique assessment**: Automated WHO compliance checking
- **Benchmarking**: Compare quality assessment methods
- **Feedback systems**: Real-time coaching

### Challenges
- **Not public**: Limited access
- **Quality subjective**: Expert scores may vary
- **Hospital privacy**: Restrictions on sharing
- **Small scale**: 300 videos (smaller than PSKUS)

## Key Features
- **Quality-focused**: Holistic WHO compliance scores
- **Hospital data**: Real clinical environment
- **Expert annotations**: Infectious disease specialists
- **Comprehensive**: All WHO criteria considered

## Limitations
- **NOT PUBLIC**: Not available for download
- **Small scale**: Only 300 videos
- **Quality scores only**: No frame-level WHO step labels
- **Subjective**: Expert scoring may vary
- **Privacy**: Hospital staff concerns

## Related Datasets
- **PSKUS Hospital**: Larger (3,185 episodes), frame-level labels, public
- **METC Lab**: Lab-based, frame-level labels, public
- **UWash**: Smartwatch IMU, quality assessment, public
- **Class23**: Open-room, not public

## Applications
- **Quality prediction**: Automated WHO compliance scoring
- **Feedback systems**: Real-time coaching during handwashing
- **Training**: Healthcare worker technique improvement
- **Benchmarking**: Compare quality assessment methods
- **Research**: Study factors affecting compliance

## Comparison with Other Datasets

### vs PSKUS
- **HHA300**: Quality scores (holistic assessment)
- **PSKUS**: Frame-level WHO step labels
- **HHA300**: 300 videos (smaller)
- **PSKUS**: 3,185 episodes (much larger, public)

### vs UWash
- **HHA300**: Vision-based (hospital cameras)
- **UWash**: IMU-based (smartwatch sensors)
- **Both**: Quality assessment focus

## Access Information
- **Status**: NOT PUBLIC
- **Request**: Contact authors via paper
- **Alternative**: Use PSKUS (public, larger) or UWash (public, IMU-based)

## Research Value
- **Quality assessment**: Gold standard for WHO compliance
- **Hospital validation**: Real-world clinical data
- **Expert labels**: High-quality annotations
- **Benchmarking**: Standard for quality prediction models
