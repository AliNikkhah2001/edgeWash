# Shadow Augmentation for Handwashing Action Recognition (MMSP 2024)

## Overview
Studies how shadow-induced domain shift degrades handwashing action recognition and proposes shadow augmentation to improve model robustness.

## Research Contributions
- **Domain shift analysis**: Quantified shadow-induced performance degradation
- **Shadow augmentation method**: Data augmentation strategy to improve robustness
- **Cross-environment validation**: Tested on outdoor/portable sink scenarios
- **Practical deployment focus**: Addresses real-world lighting variations

## Problem Statement
- Handwashing action recognition systems trained indoors fail outdoors
- Strong shadows cast by sunlight degrade model accuracy
- Indoor-trained models lack robustness to shadow variations
- Need for domain adaptation strategies for outdoor deployment

## Methodology

### Datasets
1. **Indoor Training Data**: Standard handwashing datasets (PSKUS/METC)
2. **Shadow Test Datasets**:
   - **Portable51**: 51 clips at portable sinks (outdoor)
   - **Farm23**: 23 clips at farm environments (outdoor)
   - **Public Availability**: Not released (non-public)

### Shadow Domain Shift Analysis
- **Indoor → Outdoor**: Significant accuracy drop observed
- **Shadow Characteristics**: Hard edges, high contrast, dynamic movement
- **Failure Modes**: Model confuses shadows with hand movements

### Shadow Augmentation Strategy
- **Synthetic Shadow Generation**: Add artificial shadows during training
- **Shadow Parameters**: Position, angle, intensity, edge hardness
- **Augmentation Probability**: Applied to training data probabilistically
- **Goal**: Make model invariant to shadow presence

### Model Architecture
- **Base Models**: Standard CNNs (MobileNetV2, ResNet) tested
- **Training**: With and without shadow augmentation
- **Evaluation**: Indoor (standard) vs outdoor (shadow) test sets

## Results
- **Without Augmentation**: Large accuracy drop on shadow test sets
- **With Shadow Augmentation**: Improved robustness to shadows
- **Trade-offs**: Slight indoor accuracy loss for significant outdoor gains
- **Generalization**: Better cross-domain performance

## Dataset Details
- **Portable51 Shadow Dataset**:
  - 51 portable sink clips (outdoor)
  - Strong sunlight shadows
  - Public Availability: No
- **Farm23 Shadow Dataset**:
  - 23 farm environment clips
  - Variable lighting conditions
  - Public Availability: No

## Technical Structure
1. **Baseline Training**: Indoor data only
2. **Shadow Augmentation Training**: Indoor data + synthetic shadows
3. **Evaluation**: Indoor test + outdoor shadow test
4. **Metrics**: Accuracy, per-class performance, confusion analysis

## Key Findings
- **Shadow impact**: Significant degradation without augmentation
- **Augmentation effectiveness**: Restores performance on shadow data
- **Minimal indoor loss**: Small accuracy trade-off on standard data
- **Practical value**: Enables outdoor deployment

## Limitations
- **Shadow datasets not public**: Cannot reproduce experiments
- **Synthetic shadows**: May not fully capture real shadow diversity
- **Limited outdoor scenarios**: Only portable sinks and farms tested
- **No multi-domain data**: Training still requires indoor data

## Applications
- **Outdoor handwashing monitoring**: Portable sinks, camping, farms
- **Variable lighting**: Robust to time-of-day changes
- **Cross-environment deployment**: Single model for indoor/outdoor
- **Domain adaptation**: General approach for shadow robustness

## Related Work Comparison
- **vs. Domain randomization**: Focused specifically on shadows
- **vs. Multi-domain training**: Uses augmentation instead of diverse data
- **vs. Style transfer**: Simpler synthetic shadow generation

## Availability
- **Paper**: MMSP 2024 conference proceedings
- **Code**: Not publicly released
- **Datasets**: Not publicly released (Portable51, Farm23)
- **Trained Weights**: Not provided
- **External API**: No

## Citation
```
Paper presented at IEEE International Workshop on Multimedia Signal Processing (MMSP), 2024
```
