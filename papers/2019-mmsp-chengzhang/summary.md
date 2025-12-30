# Hand-Hygiene Activity Recognition in Egocentric Video (MMSP 2019)

Two-stage egocentric video pipeline for temporal localization and classification of hand-hygiene actions.

## Data
- **Participants:** 100; each recorded twice (200 untrimmed videos).
- **Capture:** chest-mounted GoPro, 1080p @ 30 FPS; downsampled to 480x270.
- **Classes:** 8 actions (touch faucet with elbow/hand, rinse, rub without water, rub with water, apply soap, dry with towel, non-hygiene).
- **Labels:** frame-level labels; trimmed clips created (1380 train, 675 test); 135/65 video split.

## Method
- **Stage 1:** low-cost hand/motion cues for temporal localization.
- **Stage 2:** two-stream CNN (RGB + optical flow) with ResNet-152 pretrained on ImageNet; 224x224 inputs.
- **Temporal handling:** optical flow stream; sparse vs dense frame sampling with score fusion.

## Results
- **Accuracy:** fusion ~87% on trimmed clips; detection accuracy ~80% on untrimmed videos.

## Availability
- **Data/code:** dataset not public; code and weights not released; no API.
