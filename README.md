# Handwashing Research Hub

Aggregated code, papers, datasets, models, and experiment ideas for automated handwashing assessment.
_Last updated: 2025-12-30 19:48 UTC_

## Structure
- `code/`: cloned codebases and pipelines
- `papers/`: papers with `summary.md`, `tags.md`, and `paper.pdf`
- `datasets/`: storage location for raw/processed datasets (gitignored)
- `models/`: exported weights and model cards
- `evaluation/`: benchmarks and result artifacts
- `ideas/`: future experiment notes and design sketches

## Codebases
- [ ] **TensorFlow Handwash Monitoring Demo** (`code/Handwash-Monitoring`) — tags: depth, cnn, dispenser-detection, tensorflow — source: https://github.com/SidhantSarkar/Handwash-Monitoring: Overview Lightweight TF/Keras prototype that detects hand presence and dispenser use from depth video near sinks using transfer learning.
- [ ] **UWash Smartwatch Handwashing Assessment** (`code/UWash`) — tags: wearable, imu, smartwatch, quality-assessment — source: https://github.com/aiotgroup/UWash: Overview IMU-based smartwatch pipeline for detecting and scoring handwashing quality/compliance, with preprocessing and model training scripts.
- [ ] **WashWise - Real-Time Handwashing Step Tracker** (`code/WashWise`) — tags: yolo, real-time, who-steps, streamlit, roboflow — source: https://github.com/aarnavshah12/WashWise: Overview YOLOv8-based demo from Roboflow for real-time WHO step classification on sink camera footage with Streamlit GUI.
- [ ] **EdgeWash WHO Movement Classifier** (`code/edgewash`) — tags: who-steps, edge, classification, vision — source: https://github.com/AliNikkhah2001/edgeWash: Overview Lightweight edge-focused pipeline for WHO hand-washing movement recognition with exportable on-device models.
- [ ] **EDI-Riga Handwashing Movement Classifiers** (`code/edi-riga-handwash`) — tags: hospital, who-steps, classification, pytorch — source: https://github.com/edi-riga/handwash: Overview Hospital-focused WHO step classifiers with training scripts and pretrained checkpoints maintained by EDI-Riga (identical to EdgeWash codebase - same research group).
- [ ] **Hand-Wash Compliance Detection with YOLO** (`code/hand-wash-compliance-yolo`) — tags: yolo, compliance-detection, vision — source: https://github.com/dpt-xyz/hand-wash-compliance-yolo: Overview YOLO-based detector for monitoring hand presence and compliance near sinks, including dataset prep and training utilities.
- [ ] **HandWash Multimodal Surgical Rub Recognition** (`code/huiwen-HandWash`) — tags: surgical, multimodal, feedback, vision — source: https://github.com/huiwen99/HandWash: Overview Multimodal hand-rub recognition project combining RGB video and additional cues, plus a feedback UI for user guidance.
- [ ] **Synthetic Hand-Washing Data Generation** (`code/synthetic-hand-washing`) — tags: synthetic-data, generation, augmentation — source: https://github.com/r-ozakar/synthetic-hand-washing: Overview Rendering and augmentation toolkit for creating synthetic handwashing datasets plus training recipes for baseline models.
- [ ] **wearPuck Multimodal Wearable Toolkit** (`code/wearPuck`) — tags: wearable, imu, environmental, hardware — source: https://github.com/kristofvl/wearPuck: Overview Firmware and data tooling for the wearPuck device capturing IMU and environmental signals for handwashing analytics.

## Papers
- [ ] **Vision-Based Hand Hygiene Monitoring with Depth Cameras (Stanford PAIR 2015)** (`papers/2015-stanford-depth`) — tags: depth, privacy, hospital, non-public, cnn: Overview Privacy-focused hand hygiene monitoring system using overhead depth sensors near soap dispensers in hospital settings.
- [ ] **Hand-Hygiene Activity Recognition in Egocentric Video (MMSP 2019)** (`papers/2019-mmsp-chengzhang`) — tags: egocentric-video, handwashing, temporal-detection, two-stream-cnn, food-safety: Two-stage egocentric video pipeline that localizes hand-hygiene actions and then classifies them with a two-stream CNN.
- [ ] **Automated Quality Assessment of Hand Washing Using Deep Learning (2020)** (`papers/2020-arxiv-2011.11383`) — tags: handwashing, quality-assessment, mobilenet, transfer-learning, hospital: Overview Frame-level WHO step recognition on hospital footage using compact CNNs (MobileNetV2 and Xception).
- [ ] **Designing a Computer-Vision Application: Hand-Hygiene Assessment in an Open-Room Environment (J. Imaging 2021)** (`papers/2021-jimaging-chengzhang`) — tags: handwashing, deployment, dataset-design, open-world, computer-vision: Open-room deployment study with a new side-view dataset and baseline action-recognition models.
- [ ] **Hand-Washing Video Dataset Annotated According to the WHO Guidelines (Data 2021)** (`papers/2021-mdpi-who-dataset`) — tags: dataset, who-steps, hospital, annotations, public: Introduces the PSKUS hospital dataset with frame-level WHO step annotations and baseline CNN results.
- [ ] **Hand Hygiene Assessment via Joint Step Segmentation and Key Action Scorer (2023)** (`papers/2022-arxiv-2209.12221`) — tags: handwashing, step-segmentation, quality-assessment, fine-grained-actions, healthcare: Fine-grained framework that segments WHO steps and scores key actions using a convolution-transformer.
- [ ] **Handwashing Action Detection System for an Autonomous Social Robot (2022)** (`papers/2022-arxiv-2210.15804`) — tags: handwashing, action-detection, robotics, real-time, feedback: CNN with attention for real-time handwashing step recognition to drive robot feedback.
- [ ] **Shadow Augmentation for Handwashing Action Recognition: From Synthetic to Real Datasets (MMSP 2024)** (`papers/2024-mmsp-shadow-augmentation`) — tags: handwashing, data-augmentation, domain-robustness, synthetic-data, shadow-invariance: Study of shadow-induced domain shift and a shadow augmentation strategy for robust handwashing recognition.
- [ ] **Hand Washing Gesture Recognition Using Synthetic Dataset (J. Imaging 2025)** (`papers/2025-mdpi-synthetic`) — tags: synthetic, data-augmentation, rgb, depth, yolo: Introduces a large synthetic dataset and benchmarks InceptionV3, YOLOv8n, and PointNet for gesture recognition.

## Datasets
- [ ] **Class23 Open-Room Hand Hygiene Dataset (2024, restricted)** (`datasets/class23-open-room`) — tags: open-room, multi-view, restricted, rgb-video: Open-room, side-view hand-hygiene dataset with 105 untrimmed videos from 23 participants across three camera viewpoints (1080p/30 FPS).
- [ ] **HHA300 Hand Hygiene Assessment Dataset (2023, non-public)** (`datasets/hha300`) — tags: who-steps, quality-scoring, non-public, rgb-video, research: Fine-grained assessment dataset with 300 handwashing videos (60 participants) annotated at frame-level and video-level.
- [ ] **Kaggle Handwash (Resorted to WHO 6+1 Classes)** (`datasets/kaggle-who6`) — tags: who-steps, public, rgb-video, kaggle, 7-classes: Public subset of the Kaggle hand-wash videos re-sorted into 7 WHO-style classes.
- [ ] **METC Lab Handwashing Dataset** (`datasets/metc-lab`) — tags: who-steps, lab, rgb-video, public, zenodo: Lab-collected WHO handwashing videos from RSU METC with multi-camera interfaces and frame-level labels.
- [ ] **OCDetect Wrist Motion Dataset (2023)** (`datasets/ocdetect-wrist`) — tags: wearable, imu, public, ocd, event-detection: Real-world smartwatch dataset for handwashing event detection using wrist motion data.
- [ ] **Portable51 & Farm23 Shadow Datasets (2024, non-public)** (`datasets/portable51-farm23`) — tags: outdoor, shadows, non-public, robustness: Outdoor handwashing datasets used to study shadow-induced domain shift in action recognition.
- [ ] **PSKUS Hospital Handwashing Dataset** (`datasets/pskus-hospital`) — tags: who-steps, hospital, rgb-video, public, zenodo: Large hospital dataset with 3,185 annotated handwashing episodes (30 FPS; 320x240 or 640x480).
- [ ] **Stanford Depth Hand Hygiene Dataset (2015, non-public)** (`datasets/stanford-depth`) — tags: depth, privacy-preserving, non-public, hospital: Depth-only hospital dataset for dispenser-use and hand-presence detection.
- [ ] **Synthetic Hand-Washing Gesture Dataset (Özakar & Gedikli, 2025)** (`datasets/synthetic-blender-rozakar`) — tags: synthetic, rgb, depth, segmentation, public: Open synthetic dataset of 96,000 rendered frames (64 minutes) across 8 gestures, 4 characters, and 4 environments.
- [ ] **UWash Smartwatch Dataset (2023)** (`datasets/uwash-smartwatch`) — tags: imu, wearable, smartwatch, public, quality-assessment: Smartwatch IMU dataset for handwashing quality assessment with raw and preprocessed splits.
- [ ] **wearPuck Multimodal Handwash Dataset (2024)** (`datasets/wearpuck`) — tags: imu, environmental, wearable, public, multimodal: Wearable sensor dataset for handwashing event detection using IMU + environmental signals.

## Models
- Staging area for trained weights and model cards.

## Evaluation
- Add benchmark summaries or result notebooks here.

## Ideas
- Roadmap (ideas/roadmap.md)
- Todo (ideas/todo.md)

## TODOs
- [ ] Data: download & checksum PSKUS, METC, Kaggle WHO6; wire fetch scripts into CI smoke check.
  - [ ] Add lightweight sample slices for quick tests (5 clips per class).
  - [ ] Normalize label maps across datasets (WHO6+Other).
- [ ] Models: train baseline MobileNetV2 (frames) and GRU (clips) on combined hospital+lab; log metrics to `evaluation/`.
  - [ ] Fine-tune two-stream (RGB+OF) with shadow augmentation recipes from 2024 paper.
  - [ ] Benchmark synthetic pretraining (r-ozakar) then finetune on real data.
- [ ] Tracking: implement step-duration tracker (start/end, per-step seconds, coverage) fed by classifier outputs.
  - [ ] Export JSON/CSV compliance report per video.
  - [ ] Add threshold configs (per-step seconds, total duration).
- [ ] UI/Pages: publish GitHub Pages summary and embed latest README; add demo GIFs once tracker exists.
  - [ ] Add model cards for each released checkpoint.
- [ ] Sensors: prototype fusion (wearPuck humidity + IMU) baseline and compare to vision-only triggers.

## Automation
- `scripts/build_readme.py` regenerates this README from folder metadata.
- `.github/workflows/build-readme.yml` runs the generator on each push and commits changes.
- `.github/workflows/pages.yml` builds GitHub Pages from the generated docs.

To add new assets, drop them in the appropriate folder with minimal metadata; the automation will refresh this page.
