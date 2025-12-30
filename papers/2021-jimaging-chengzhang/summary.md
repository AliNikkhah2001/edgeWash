# Designing a Computer-Vision Application: Hand-Hygiene Assessment in Open-Room Environment (J. Imaging 2021)

## Overview
Case study of deploying a hand-hygiene assessment system in an open-room setting, addressing challenges beyond sink-mounted cameras.

## Research Contributions
- **Open-room deployment**: Beyond fixed sink locations (food lab, open floor plan)
- **Multi-view capture**: 3 camera angles for better coverage
- **Deployment challenges**: Practical considerations for real-world systems
- **Computer vision application design**: System architecture and integration

## Problem Statement
- Previous systems focused on sink-mounted cameras (fixed location)
- Food manufacturing environments have open floor plans
- Workers move freely, not tied to specific sinks
- Multi-view coverage needed for robust detection
- System integration challenges (hardware, software, networking)

## Methodology

### Deployment Environment
- **Location**: Food manufacturing laboratory (open-room)
- **Participants**: 23 workers (Class23 dataset)
- **Camera Setup**: 3 fixed cameras with overlapping fields of view
- **Recording**: 105 untrimmed videos (multiple angles per session)
- **Resolution**: 1080p @ 30 FPS

### System Architecture
- **Hardware**: IP cameras with PoE (Power over Ethernet)
- **Processing**: Centralized server with GPU for real-time inference
- **Network**: Gigabit Ethernet for video streaming
- **Storage**: NAS (Network Attached Storage) for video archives
- **Software**: Custom pipeline for multi-camera synchronization

### Model Architecture
- **Base**: Two-stream network (RGB + Optical Flow)
- **Backbone**: ResNet-152 (pretrained on ImageNet)
- **Multi-view**: Separate predictions from each camera angle
- **Fusion**: Late fusion of multi-camera predictions

### Data Collection Protocol
- **Instruction**: Workers given WHO handwashing guidelines
- **Recording**: Natural behavior (minimal supervision)
- **Multi-angle**: 3 cameras capture same washing event
- **Synchronization**: Timestamps for aligning camera feeds
- **Annotations**: Frame-level labels for 8 action classes

## Challenges Addressed

### Multi-View Integration
- **Camera calibration**: Spatial alignment of camera views
- **Temporal synchronization**: Frame-level alignment across cameras
- **Occlusion handling**: Multiple views reduce occlusion impact
- **View fusion**: Combining predictions from different angles

### Open-Room Considerations
- **Background clutter**: Workers, equipment, movement in frame
- **Variable lighting**: Natural and artificial light sources
- **Worker mobility**: Not fixed to one location
- **Privacy**: Balancing monitoring with worker privacy

### System Integration
- **Real-time requirements**: Low-latency inference for feedback
- **Scalability**: Multiple cameras, multiple sinks
- **Network bandwidth**: Streaming 3× 1080p feeds
- **Storage management**: Long-term video archives

## Results
- **Multi-view benefit**: Improved accuracy vs single camera
- **Occlusion robustness**: Multiple angles reduce missed detections
- **Real-world validation**: Successful deployment in food lab
- **System usability**: Practical deployment insights

## Dataset Details
- **Name**: Class23 Open-Room Hand Hygiene Dataset
- **Size**: 105 untrimmed videos (23 participants × 3 cameras + repeats)
- **Environment**: Open-room food manufacturing lab
- **Cameras**: 3 synchronized cameras (multi-view)
- **Resolution**: 1080p @ 30 FPS
- **Public Availability**: **No** - consent restrictions
- **Annotations**: Frame-level labels for 8 action classes

## Technical Structure

### Multi-Camera Pipeline
1. **Capture**: 3 cameras record simultaneously
2. **Streaming**: Video feeds to central server
3. **Synchronization**: Timestamp alignment
4. **Processing**: Two-stream CNN on each camera feed
5. **Fusion**: Combine predictions across views
6. **Output**: Unified action detection + compliance report

### Key Design Choices
- **Multi-view**: Redundancy for occlusion handling
- **Open-room**: Flexible deployment (not sink-specific)
- **Centralized processing**: GPU server for real-time inference
- **Two-stream**: Motion-aware action recognition

## Limitations
- **Dataset not public**: Cannot reproduce experiments
- **Fixed cameras**: Still requires infrastructure setup
- **3 cameras only**: Limited view diversity
- **No privacy-preserving**: RGB video raises privacy concerns
- **Centralized processing**: Requires powerful server

## Applications
- **Food manufacturing**: Open-room worker monitoring
- **Multi-sink environments**: Coverage beyond single sink
- **Quality control**: Compliance auditing in open spaces
- **Training environments**: Feedback for workers in food labs

## Related Work Comparison
- **vs. Sink-mounted**: Flexible deployment, multi-view coverage
- **vs. Egocentric**: Fixed infrastructure, no wearables required
- **vs. Single camera**: Better occlusion handling, redundancy
- **vs. Privacy-preserving**: RGB provides detail but raises privacy concerns

## Future Directions
- **More cameras**: Expand coverage, reduce occlusions
- **Privacy-preserving**: Depth cameras or pose estimation
- **Edge processing**: Distributed inference on camera devices
- **Automated calibration**: Simplify multi-camera setup

## Availability
- **Paper**: Journal of Imaging, 2021
- **Code**: Not publicly released
- **Dataset**: Not publicly released (Class23, consent restrictions)
- **Trained Weights**: Not provided
- **External API**: No

## Citation
```
Paper published in Journal of Imaging, 2021
Focus on deployment case study and system design
```
