# Stanford Depth Camera Hand Hygiene Dataset

## Overview
Privacy-preserving depth camera dataset capturing ~20 hours of hospital hand hygiene with silhouette-based anonymization.

## Size & Scale
- **Total Duration**: ~20 hours of recordings
- **Environment**: Hospital setting (Stanford Hospital)
- **Camera Type**: Depth cameras (Kinect-like sensors)
- **Privacy**: Silhouette-based (no RGB/identifiable features)
- **Participants**: Hospital staff and visitors
- **Resolution**: Depth map resolution (typically 320×240 or 640×480)

## Data Collection

### Hardware Setup
- **Sensors**: Depth cameras (Microsoft Kinect or similar)
- **Placement**: Above or beside hand hygiene stations
- **Modality**: Depth-only (no RGB)
- **Recording**: Continuous or motion-triggered
- **Privacy Design**: No color/facial features captured

### Collection Protocol
- **Location**: Stanford Hospital hand hygiene stations
- **Duration**: Extended deployment (~20 hours total)
- **Participants**: Natural hospital users (staff, visitors)
- **Behavior**: Spontaneous hand hygiene (no instructions)
- **Privacy-First**: Depth-only capture for anonymization

### Privacy Features
- **No RGB**: Only depth information captured
- **Silhouettes**: Body/hand shapes only (no identifiable features)
- **Anonymization**: Inherent privacy protection
- **Hospital Approved**: Meets privacy requirements for deployment

## Annotations

### Annotation Process
- **Annotators**: Trained reviewers
- **Guidelines**: WHO hand hygiene guidelines
- **Labels**: Handwashing events and durations
- **Quality**: Compliance assessment (presence/absence)

### Event Detection
- **Binary**: Hand hygiene event vs no event
- **Duration**: Event start/end timestamps
- **Not WHO Steps**: Does not classify individual movements
- **Focus**: Detection and duration, not step-by-step analysis

## Dataset Structure

### File Organization
```
stanford-depth/
├── depth_videos/
│   ├── session001.depth
│   ├── session002.depth
│   └── ...
└── annotations/
    └── events.csv
```

### File Formats
- **Depth Videos**: Depth map sequences (proprietary or standard format)
- **Annotations**: CSV with event timestamps
- **Resolution**: Depth sensor resolution (320×240 or 640×480)
- **Frame Rate**: Typically 30 FPS

## Sample Statistics

### Video Characteristics
- **Total Duration**: ~20 hours
- **Event Count**: Multiple events per hour
- **Depth Range**: 0.5-4 meters (typical depth camera range)
- **Quality**: Consistent (controlled hospital lighting)

### Depth Camera Characteristics
- **Technology**: Time-of-flight or structured light
- **Range**: 0.5-4 meters (optimal for sink monitoring)
- **Resolution**: 320×240 or 640×480 depth pixels
- **Privacy**: No color/RGB data captured

## Public Availability

### Access Status
- **NOT PUBLIC**: Dataset not publicly released
- **Reason**: Hospital privacy policies
- **Request**: May be available upon request to authors
- **Institution**: Stanford University research group

### Citation Requirement
```
Stanford Depth Camera Hand Hygiene Dataset
Paper: papers/2015-stanford-depth
Contact authors for access
```

## Usage Notes

### Privacy Advantages
- **Depth-only**: No identifiable features (faces, skin tone)
- **Silhouettes**: Body shapes only
- **Hospital-approved**: Meets strict privacy requirements
- **Public deployment**: Feasible in sensitive environments

### Research Applications
- **Privacy-preserving monitoring**: Hospital/clinic deployment
- **Event detection**: Binary presence/absence
- **Duration monitoring**: Compliance with 20+ second guideline
- **Transfer learning**: Depth-to-RGB domain adaptation

## Key Features
- **Privacy-preserving**: Depth-only (no RGB)
- **Hospital deployment**: Real clinical setting (~20 hours)
- **Silhouette-based**: Anonymous body/hand shapes
- **Event detection**: Handwashing presence and duration
- **Pioneering**: Early work in privacy-preserving hand hygiene

## Limitations
- **NOT PUBLIC**: Not available for download
- **Depth-only**: Limited information compared to RGB
- **Event detection**: No WHO step classification
- **Small annotations**: Limited labeled events
- **Technology-specific**: Requires depth cameras for inference

## Related Datasets
- **PSKUS Hospital**: RGB, larger scale, frame-level labels, public
- **METC Lab**: RGB, controlled, frame-level labels, public
- **Synthetic**: RGB+depth, perfect labels, public
- **HHA300**: RGB, quality scores, not public

## Applications
- **Privacy-preserving monitoring**: Deploy in hospitals without RGB
- **Event detection**: Binary handwashing detection
- **Duration monitoring**: Compliance with time guidelines
- **Research**: Depth vs RGB performance comparison
- **Policy compliance**: Meet strict privacy requirements

## Research Paper
- **Year**: 2015
- **Institution**: Stanford University
- **Focus**: Privacy-preserving hand hygiene monitoring
- **Innovation**: Depth-only approach for hospital deployment
- **Paper**: papers/2015-stanford-depth/paper.pdf

## Comparison with RGB Datasets

### vs PSKUS (RGB)
- **Stanford**: Depth-only, privacy-preserving
- **PSKUS**: RGB, more visual information
- **Stanford**: Event detection only
- **PSKUS**: Frame-level WHO step labels

### Privacy Trade-off
- **Depth**: Better privacy, less information
- **RGB**: More information, privacy concerns
- **Hospital deployment**: Depth often preferred

## Access Information
- **Status**: NOT PUBLIC
- **Request**: Contact Stanford authors via paper
- **Alternative**: Use PSKUS (public, RGB) or Synthetic (public, RGB+depth)

## Historical Significance
- **Early work**: One of first privacy-preserving approaches (2015)
- **Depth cameras**: Pioneered depth-only hand hygiene monitoring
- **Influence**: Inspired later privacy-focused research
