# Automated Quality Assessment of Hand Washing Using Deep Learning (2020)

## Overview
Frame-level WHO step recognition on hospital footage using compact CNNs (MobileNetV2 and Xception).

## Research Contributions
- **First large-scale hospital dataset**: 1,854 annotated handwashing episodes from real clinical environment
- **Transfer learning approach**: Demonstrated ImageNet → WHO step transfer learning viability
- **Mobile-ready models**: MobileNetV2 for resource-constrained deployment
- **Frame-level recognition**: Simple per-frame classification (no temporal modeling)
- **Double annotation**: 1,094 videos annotated by two people for reliability
- **Mobile application concept**: Proposed real-time feedback system for medical professionals

## Problem Statement
- Medical staff often fails to follow WHO handwashing guidelines (even in hospitals)
- Lack of compliance causes preventable infections (4.1M annually in Europe, 37,000 deaths)
- Manual observation (Hawthorne effect): Staff performs better only when watched
- Need automated quality control for:
  - Total handwashing duration
  - Duration of each WHO movement
  - Compliance with all required movements

## Methodology

### Data Collection System
- **Hardware**: 
  - IP cameras: AirLive POE 100CAM, Axis M3046V
  - Control: Raspberry Pi 4 single-board computers
  - Storage: Micro SD cards
  - Network: Netgear 5-Port PoE Gigabit Ethernet switch
- **Deployment**: 9 sites at Pauls Stradins Clinical University Hospital (Latvia)
- **Placement**: Cameras mounted above sinks
- **Recording Trigger**: Motion detection >10 seconds
- **Duration**: 3 months of continuous data collection
- **Resolution**: 640×480 pixels @ 30 FPS
- **Storage**: Local micro SD cards, monthly manual collection

### Dataset Structure
- **Total Videos**: 32,471 captured (1,854 annotated)
- **Annotated**: 2,293 files total
  - 1,199 annotated once
  - 1,094 annotated twice (for reliability)
- **Format**: 
  - Video files (.mp4 or .avi)
  - JSON annotation files (frame-level labels)
  - CSV statistics files (per-video metadata)

### Annotation Process
- **Custom Annotation Tool**: Python + OpenCV GUI
- **Annotators**: Infectious disease specialists, medical professionals, RSU students
- **Guidelines**: Developed with local infectious disease specialists
- **Labels**: 7 WHO movement classes per frame
  - Palm to palm
  - Palm over dorsum with fingers interlaced
  - Palm to palm with fingers interlaced
  - Back of fingers to opposing palm
  - Rotational rubbing of thumb
  - Fingertips to palm
  - Turning off faucet with paper towel
- **Additional Flags**: Ring, watch, lacquered nails (inappropriate for medical staff)

### Model Architecture

#### MobileNetV2
- **Base**: ImageNet pretrained (Keras implementation)
- **Type**: Compact CNN for mobile deployment
- **Input**: 224×224 RGB frames
- **Output**: 7 WHO movement classes (frame-level)
- **Transfer Learning**: Fine-tune top layers on hospital data

#### Xception
- **Base**: ImageNet pretrained (Keras implementation)
- **Type**: Larger CNN for higher accuracy
- **Input**: 299×299 RGB frames
- **Output**: 7 WHO movement classes (frame-level)
- **Trade-off**: Slower but more accurate than MobileNetV2

### Training Details

#### Dataset Split (Preliminary Experiments)
- **Subset**: 378 videos from full dataset
- **Frames**: 309,315 total frames
- **Split**: 70% train (216,520), 20% val (61,863), 10% test (30,932)
- **Frame Extraction**: Videos split into individual frames
- **Resizing**: 
  - MobileNetV2: 224×224
  - Xception: 299×299

#### Augmentation
- **Random Flip**: Horizontal/vertical
- **Random Rotation**: ±20 degrees
- **Purpose**: Increase generalization, account for camera angle variations

#### MobileNetV2 Training
- **Epochs**: 50 (max)
- **Early Stopping**: 10 epochs without validation improvement
- **Loss**: Categorical cross-entropy
- **Optimizer**: RMSprop
- **Learning Rate**: 0.0001
- **Batch Size**: Not specified (default Keras)
- **Actual Training**: Stopped after 3 epochs (early stopping triggered)

#### Xception Training
- **Epochs**: 10
- **Loss**: Categorical cross-entropy
- **Optimizer**: Adam
- **Learning Rate**: 0.001 (default)
- **Batch Size**: Not specified

#### Hyperparameter Tuning
- **Status**: No tuning performed (preliminary results only)
- **Future Work**: Hyperparameter search planned

### Temporal Handling
- **Approach**: **None** - purely per-frame classification
- **3D Convolutions**: No
- **Sequence Modeling**: No (RNN/LSTM/GRU)
- **Duration Tracking**: External application logic (not in model)
- **Time Series**: Not modeled by CNN

## Results

### Frame-Level Accuracy (309,315 test frames)
| Model | Accuracy |
|-------|----------|
| MobileNetV2 | 64.03% |
| Xception | 66.83% |

**Interpretation**:
- Preliminary results (no hyperparameter tuning)
- Xception +2.8% better than MobileNetV2
- Lower than typical image classification (due to fine-grained WHO steps, motion blur, occlusions)

### Challenges Observed
- **Similar movements**: Some WHO steps visually similar (e.g., interlaced variations)
- **Motion blur**: Handwashing involves fast motion
- **Occlusions**: Hands overlap, obscuring individual fingers
- **Camera angle**: Fixed overhead view may miss some movements
- **Background clutter**: Sink, soap dispenser, towels in frame

### Future Improvements (Suggested in Paper)
- Hyperparameter tuning (learning rate, batch size, epochs)
- Larger training set (use full 1,854 annotated videos)
- Temporal modeling (sequence-based architectures)
- Multi-task learning (predict duration + movement simultaneously)

## Application Design (Proposed)

### Mobile Application Concept
- **Platform**: Smartphone/tablet mounted above sink
- **Input**: Live camera feed
- **Output**: Real-time feedback on handwashing quality

### State Machine Logic
**States**:
1. **Waiting**: Watching for handwashing start (motion detection)
2. **In-progress**: Washing ongoing, tracking movements and duration
3. **OK**: Total duration reached, all movements detected
4. **Failed**: Washing ended prematurely (missing steps or insufficient duration)

**Transitions**:
- Waiting → In-progress: Motion detected, washing=1
- In-progress → OK: Duration threshold met + all movements present
- In-progress → Failed: Washing stopped (washing=0) before completion
- OK/Failed → Waiting: After 5-second display, return to waiting

### Feedback Mechanisms
- **Visual**: On-screen progress indicators
- **Sound**: Audio cues for state transitions
  - Sound 1: Washing started
  - Sound 2: Movement recognized
  - Sound 3: Washing complete (OK)
  - Sound 4: Washing failed
- **Vibration**: Optional haptic feedback

### Adaptability
- **Configurable Thresholds**: 
  - Total duration (e.g., 20-30 seconds)
  - Per-movement duration (e.g., 3-5 seconds each)
- **No Retraining Required**: 
  - CNN recognizes movements (fixed)
  - Application logic handles duration/compliance (configurable)
- **WHO Guideline Updates**: 
  - If durations change: Update config only
  - If new movements added: Retrain CNN

## Dataset Details
- **Name**: PSKUS Hospital Handwashing Dataset
- **Public Availability**: Not released with this paper (2020)
  - Later released (2021) on Zenodo as standalone dataset paper
- **Size**: 1,854 annotated episodes (subset of 32,471 captured)
- **Environment**: Hospital sinks in Latvia
- **Annotations**: Frame-level WHO movement codes
- **Double Annotation**: 1,094 videos for reliability assessment

## Technical Structure

### Data Collection Pipeline
1. **Trigger**: Motion detection >10 seconds
2. **Recording**: 640×480 @ 30 FPS to SD card
3. **Monthly Collection**: Manual retrieval from deployment sites
4. **Storage**: Central server for annotation
5. **Annotation**: Custom Python/OpenCV tool
6. **Export**: JSON (labels) + CSV (statistics) per video

### Training Pipeline
1. **Frame Extraction**: Videos → individual frames
2. **Resizing**: 224×224 (MobileNetV2) or 299×299 (Xception)
3. **Augmentation**: Flip, rotate
4. **Loading**: Keras ImageDataGenerator
5. **Training**: Transfer learning from ImageNet weights
6. **Evaluation**: Frame-level accuracy on test set

### Deployment Pipeline (Proposed)
1. **Camera Input**: Live video stream
2. **Frame Processing**: Extract frames at 30 FPS
3. **CNN Inference**: Per-frame movement prediction
4. **State Machine**: Track movements and duration
5. **Feedback**: Real-time visual/audio cues
6. **Logging**: Compliance reports for auditing

## Limitations
- **Preliminary Results**: No hyperparameter tuning
- **Small Training Set**: Only 378 of 1,854 videos used
- **Frame-Based Only**: Ignores temporal dependencies
- **Lower Accuracy**: 64-67% (needs improvement for deployment)
- **Single Hospital**: Limited to one clinical environment
- **Fixed Camera Angle**: Overhead only

## Applications
- **Hospital Compliance Monitoring**: Automated hand hygiene auditing
- **Medical Training**: Provide feedback to learners
- **Quality Control**: Objective measurement of handwashing quality
- **Research**: Study handwashing behavior patterns
- **Infection Prevention**: Reduce hospital-acquired infections

## Related Work Comparison
- **vs. Wearable Sensors**: No device required, but fixed location
- **vs. Depth Cameras**: RGB provides more visual detail
- **vs. Multi-Step Classifiers**: Simpler (frame-based), but less accurate
- **vs. Manual Observation**: Objective, no Hawthorne effect

## Future Directions (from paper)
- **Hyperparameter Tuning**: Optimize learning rate, batch size, etc.
- **Full Dataset**: Train on all 1,854 annotated videos
- **Temporal Modeling**: LSTM/GRU for sequence-based recognition
- **Multi-Camera**: Combine multiple angles for better coverage
- **Mobile Deployment**: TensorFlow Lite for on-device inference
- **Real-World Trials**: Deploy and evaluate mobile app in hospitals

## Funding
- **Project**: VPP-COVID-2020/1-0004
- **Title**: "Integration of reliable technologies for protection against Covid-19 in healthcare and high risk areas"
- **Country**: Latvia

## Acknowledgements
- RSU (Riga Stradins University) staff and students for video labeling

## Availability
- **Paper**: arXiv:2011.11383 (preprint, 2020)
- **Code**: Not publicly released
- **Dataset**: Not released with this paper
  - Later released as PSKUS dataset (2021, Zenodo)
- **Trained Weights**: Not provided
- **External API**: No

## Citation
```
Maksims Ivanovs, Roberts Kadikis, Atis Elsts, Martins Lulla, Aleksejs Rutkovskis
"Automated Quality Assessment of Hand Washing Using Deep Learning"
arXiv preprint arXiv:2011.11383, 2020
```

## Related Publications (Same Research Group)
- **2021**: PSKUS dataset paper (Data journal, MDPI)
- **2022**: IPTA conference paper (CNN architecture comparison)
