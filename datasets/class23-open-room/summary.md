# Class23 Open-Room Handwash Dataset

## Overview
Open-room deployment dataset with 105 videos collected from uncontrolled environments (cafeterias, restrooms) with overhead camera views.

## Size & Scale
- **Total Videos**: 105 videos
- **Environment**: Open-room (cafeterias, restrooms)
- **Camera Placement**: Overhead above sinks
- **Participants**: Natural users (staff, students, visitors)
- **Setting**: Uncontrolled real-world (not lab)
- **Duration**: Variable per video (typically 20-60 seconds)

## Data Collection

### Hardware Setup
- **Cameras**: Overhead IP cameras
- **Placement**: Above sinks in public spaces
- **Environment**: Open-room (high traffic areas)
  - Cafeterias
  - Public restrooms
  - Hospital areas
- **Recording**: Automatic (motion-triggered or continuous)

### Collection Protocol
- **Location**: Public spaces (cafeterias, restrooms)
- **Duration**: Extended deployment period
- **Participants**: Natural users (uncontrolled)
- **Behavior**: Spontaneous handwashing (no instructions)
- **Privacy**: Overhead angle (less identifiable)

## Annotations

### Annotation Process
- **Annotators**: Trained annotators
- **Guidelines**: WHO guidelines
- **Frame-Level**: WHO step codes per frame
- **Quality Control**: Review process for consistency

### WHO Movement Classes
Standard WHO 6-step taxonomy:
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
class23-open-room/
├── videos/
│   ├── cafeteria/
│   ├── restroom/
│   └── ...
└── annotations/
    ├── video1.json
    └── ...
```

### File Formats
- **Videos**: MP4 or AVI
- **Annotations**: JSON (per video, frame-level labels)
- **Resolution**: Variable (typically 640×480 or higher)
- **Frame Rate**: 30 FPS

## Sample Statistics

### Video Characteristics
- **Total**: 105 videos
- **Per Environment**: Distribution across locations
- **Duration**: Variable (20-60 seconds typical)
- **Quality**: Variable (uncontrolled lighting, occlusions)

### Environment Challenges
- **Occlusions**: People blocking camera view
- **Variable lighting**: Different times of day
- **Crowding**: Multiple people at sinks
- **Natural behavior**: Not following WHO perfectly

## Public Availability

### Access Status
- **NOT PUBLIC**: Dataset not publicly released
- **Reason**: Privacy concerns (public spaces)
- **Request**: May be available upon request to authors
- **Institution**: Research group (specific institution TBD from paper)

### Citation Requirement
```
Class23 Open-Room Handwashing Dataset
Paper: (reference from papers/2021-jimaging-chengzhang or related)
Contact authors for access
```

## Usage Notes

### Challenges
- **Open-room**: More realistic but noisier than lab
- **Uncontrolled**: Variable lighting, occlusions
- **Privacy**: Overhead helps but still concerns
- **Annotation difficulty**: Natural behavior harder to label

### Research Value
- **Realistic**: Real-world deployment conditions
- **Generalization**: Tests model robustness
- **Domain adaptation**: Lab-to-wild transfer
- **Benchmarking**: Uncontrolled environment baseline

## Key Features
- **Open-room**: Public spaces (not lab)
- **Natural behavior**: Uncontrolled (realistic)
- **Overhead cameras**: Less invasive angle
- **WHO annotations**: Frame-level labels
- **Multiple environments**: Cafeterias, restrooms

## Limitations
- **NOT PUBLIC**: Not available for download
- **Small scale**: Only 105 videos
- **Privacy concerns**: Public space recordings
- **Variable quality**: Uncontrolled conditions
- **Annotation challenges**: Natural behavior complex

## Related Datasets
- **PSKUS Hospital**: Hospital setting, larger scale, public
- **METC Lab**: Lab-based, controlled, public
- **HHA300**: Quality scores, not public
- **Stanford Depth**: Depth cameras, privacy-preserving, not public

## Applications
- **Real-world testing**: Benchmark for uncontrolled environments
- **Domain adaptation**: Study lab-to-wild transfer
- **Robustness testing**: Occlusions, variable lighting
- **Privacy research**: Overhead camera evaluation

## Research Paper
- **Conference/Journal**: See papers/2021-jimaging-chengzhang
- **Focus**: Open-room deployment and challenges
- **Contributions**: Real-world benchmarking

## Access Information
- **Status**: NOT PUBLIC
- **Request**: Contact authors via paper correspondence
- **Alternative**: Use PSKUS (public, hospital) or METC (public, lab)
