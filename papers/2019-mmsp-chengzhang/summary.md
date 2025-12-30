# Hand-Hygiene Activity Recognition in Egocentric Video (MMSP 2019)

Two-stage egocentric video pipeline that localizes hand-hygiene actions and then classifies them with a two-stream CNN.

## Dataset
- **Participants:** 100; each recorded twice (200 untrimmed videos).
- **Capture:** chest-mounted GoPro, 1080p @ 30 FPS; downsampled to 480x270.
- **Labels:** frame-level annotations for 8 action classes.
- **Trimmed clips:** 1,380 train / 675 test clips (one action per clip); 135/65 train/test video split.
- **Classes (8):** touch faucet with elbow/hand, rinse hands, rub hands without water, rub hands with water, apply soap, dry hands, non-hygiene.

## Pipeline
1. **Stage 1 - Localization (binary):**
   - **Hand-mask CNN:** trained on 134 hand-mask images; stack length `L=5`.
   - **Input sizes:** 32x18 or 64x36 hand masks.
   - **Training:** batch size 128, learning rate 1e-5.
   - **Motion histogram:** optical-flow histograms inside/outside hand mask.
   - **Classifier:** Random Forest (30 estimators, max depth 40), bin sizes 9/12/16.
   - **Decision rule:** frame positive only if both hand-mask CNN and motion hist agree.
2. **Stage 2 - Recognition:**
   - **Two-stream CNN:** ResNet-152 RGB + ResNet-152 optical flow (ImageNet-pretrained).
   - **Sampling:** sparse (25 frames per clip) or dense (all frames); score fusion.
   - **Unit search:** untrimmed videos split into 30-frame units; units with >15 positive frames are checked first; neighbor search recovers low-motion actions.

## Temporal Handling
- **Motion modeling:** explicit optical-flow stream.
- **Sequence logic:** 30-frame units with search over neighbors; no 3D conv or RNNs.

## Results
- **Trimmed clips:** fusion accuracy ~87% (sparse/dense sampling).
- **Untrimmed videos:** detection accuracy ~80% with reduced unit inspection (~76–82% PV).

## Availability
- **Dataset:** not public.
- **Code/weights:** not released; no external API.
