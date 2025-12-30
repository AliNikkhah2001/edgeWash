# Hand Washing Gesture Recognition Using Synthetic Dataset (J. Imaging 2025)

## Overview
Introduces a 96k-frame synthetic dataset with RGB, depth, and masks covering 8 WHO-inspired gestures, generated using Blender rendering.

## Research Contributions
- **Large-scale synthetic dataset**: 96,000 frames (64 minutes) with perfect annotations
- **Multi-modal data**: RGB, depth, depth-isolated, hand segmentation masks
- **Blender-based generation**: Controlled rendering pipeline for data augmentation
- **Pre-trained models**: Inception-V3, YOLOv8, PointNet weights provided
- **Domain adaptation study**: Synthetic-to-real transfer learning experiments
- **Public release**: CC BY license (fully open)

## Problem Statement
- Real-world handwashing datasets are limited in size and diversity
- Manual annotation is labor-intensive and error-prone
- Need for large-scale training data with perfect ground truth
- Domain adaptation from synthetic to real data is under-explored for handwashing

## Methodology

### Synthetic Data Generation
- **Tool**: Blender 3D rendering engine
- **Characters**: 4 diverse character models (varied skin tones, hand sizes)
- **Environments**: 4 realistic bathroom settings (lighting, backgrounds)
- **Gestures**: 8 WHO-inspired handwashing movements
- **Total Combinations**: 4 characters × 4 environments × 8 gestures
- **Frame Count**: 96,000 frames (64 minutes of video)

### Gesture Classes (8 WHO-Inspired)
1. Palm to palm
2. Right palm over left dorsum
3. Left palm over right dorsum
4. Palm to palm with fingers interlaced
5. Back of fingers to opposing palms
6. Rotational rubbing of right thumb
7. Rotational rubbing of left thumb
8. Fingertips to palm

### Data Modalities
- **RGB Images**: Standard color frames (realistic rendering)
- **Depth Maps**: Per-pixel depth information
- **Depth-Isolated**: Depth with background removed
- **Hand Segmentation Masks**: Binary masks for hands (perfect ground truth)

### Model Architectures Tested

#### Inception-V3 (RGB Classification)
- **Framework**: TensorFlow/Keras
- **Input**: 96×96 RGB images
- **Pre-trained**: ImageNet weights
- **Fine-tuning**: Top layers on synthetic data
- **Hyperparameters**:
  - Dropout: 0.3
  - Dense layer: 64 units
  - Optimizer: Adam (lr=1e-4)
  - Cross-validation: 5-fold

#### YOLOv8 (Detection + Segmentation)
- **Framework**: Ultralytics
- **Variants**: YOLOv8n (nano) for classification and segmentation
- **Pre-trained**: COCO weights
- **Training**:
  - Epochs: 5
  - Flip augmentation: Disabled (flipud=0, fliplr=0)
  - YAML configs: 5 training configurations provided

#### PointNet (Point Cloud Classification)
- **Framework**: PyTorch
- **Input**: 512-point clouds from depth maps
- **Training**:
  - Optimizer: Adam (lr=1e-4)
  - Epochs: 3
  - Cross-validation: 5-fold
  - Point sampling: Uniform from depth surfaces

## Results

### Synthetic Data Performance
- **Inception-V3**: High accuracy on synthetic test set (exact metrics in paper)
- **YOLOv8**: Strong detection and segmentation on synthetic data
- **PointNet**: Effective point cloud classification

### Synthetic-to-Real Transfer
- **Real Test Data**: Small set of real RGB and point cloud samples provided
- **Transfer Performance**: Models trained on synthetic data tested on real samples
- **Domain Gap**: Performance drop observed (as expected), but synthetic pretraining provides good initialization

### Key Findings
- **Perfect annotations**: Synthetic data provides error-free ground truth
- **Diversity**: Character and environment variations improve generalization
- **Multi-modal benefits**: Depth and masks enhance robustness
- **Transfer learning**: Synthetic pretraining + real fine-tuning shows promise

## Dataset Details
- **Name**: Synthetic Hand-Washing Gesture Dataset (Özakar & Gedikli, 2025)
- **Size**: 96,000 frames (64 minutes)
- **Format**: Organized by gesture class folders
- **Modalities**: RGB, depth, depth-isolated, hand masks
- **Public Availability**: Yes (Google Drive, 5 download links)
- **License**: CC BY (Attribution)
- **Download Links** (in paper README):
  - https://drive.google.com/file/d/1EW3JQvElcuXzawxEMRkA8YXwK_Ipiv-p/view
  - https://drive.google.com/file/d/163TsrDe4q5KTQGCv90JRYFkCs7AGxFip/view
  - https://drive.google.com/file/d/1GxyTYfSodumH78NbjWdmbjm8JP8AOkAY/view
  - https://drive.google.com/file/d/1IoRsgBBr8qoC3HO-vEr6E7K4UZ6ku6-1/view
  - https://drive.google.com/file/d/1svCYnwDazy5FN1DYSgqbGscvDKL_YnID/view

## Technical Structure

### Data Generation Pipeline
1. **Character Modeling**: 4 diverse 3D hand models in Blender
2. **Environment Setup**: 4 realistic bathroom scenes
3. **Animation**: Keyframe animation for 8 gestures per character/environment
4. **Rendering**: RGB, depth, and mask rendering in parallel
5. **Export**: Frame extraction and organization by class

### Training Pipeline
1. **Data Preparation**: 
   - RGB: Pickle files via `create_rgb_pickle.py`
   - Point Cloud: Pickle files via `create_pcd_pickle.py`
   - YOLO: Format conversion via `ready_for_training.py`
2. **Model Training**: Architecture-specific scripts
3. **Evaluation**: 5-fold cross-validation on synthetic data
4. **Transfer Testing**: Evaluation on real-world samples

## Limitations
- **Synthetic-to-real gap**: Domain shift remains (lighting, hand variations)
- **8 gestures only**: Not exact WHO 6-step taxonomy
- **Frame-based**: No temporal/video modeling
- **Limited real test data**: Few real samples for validation
- **CGI artifacts**: Rendering may not capture all real-world nuances

## Applications
- **Pre-training for real data**: Bootstrap models with large synthetic dataset
- **Data augmentation**: Mix synthetic with real data for robustness
- **Annotation cost reduction**: Synthetic data provides free labels
- **Research baseline**: Public dataset for benchmarking
- **Domain adaptation studies**: Test synthetic-to-real transfer techniques

## Related Work Comparison
- **vs. Real datasets**: Larger scale, perfect annotations, but domain gap
- **vs. Manual augmentation**: More diverse, automated generation
- **vs. GAN-based synthesis**: Explicit control over scene parameters

## Future Directions
- **More characters**: Expand diversity (ages, hand sizes, skin tones)
- **More environments**: Outdoor, hospital, kitchen settings
- **Dynamic lighting**: Variable illumination conditions
- **Temporal modeling**: Generate video sequences for temporal classifiers
- **Fine-tuning strategies**: Optimize synthetic-to-real transfer

## Availability
- **Paper**: Journal of Imaging, MDPI, 2025
- **DOI**: 10.3390/jimaging11070208
- **URL**: https://www.mdpi.com/2313-433X/11/7/208
- **Code**: Yes (GitHub, training scripts included)
- **Dataset**: Yes (Google Drive, CC BY license)
- **Pre-trained Weights**: Yes (Inception-V3, YOLOv8n, PointNet)
- **External API**: No

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
  ISSN = {2313-433X},
  DOI = {10.3390/jimaging11070208}
}
```
