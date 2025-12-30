# Handwashing Research Hub

Aggregated code, papers, datasets, models, and experiment ideas for automated handwashing assessment.
_Last updated: 2025-12-30 20:21 UTC_

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
- [ ] **Designing a Computer-Vision Application: Hand-Hygiene Assessment in Open-Room Environment (J. Imaging 2021)** (`papers/2021-jimaging-chengzhang`) — tags: handwashing, deployment, dataset-design, open-world, computer-vision: Overview Case study of deploying a hand-hygiene assessment system in an open-room setting, addressing challenges beyond sink-mounted cameras.
- [ ] **Hand-Washing Video Dataset Annotated According to the WHO Guidelines (Data 2021)** (`papers/2021-mdpi-who-dataset`) — tags: dataset, who-steps, hospital, annotations, public: Overview Dataset paper presenting the PSKUS Hospital Handwashing Dataset with 3,185 annotated episodes following WHO guidelines.
- [ ] **Hand Hygiene Assessment via Joint Step Segmentation and Key Action Scorer (2023)** (`papers/2022-arxiv-2209.12221`) — tags: handwashing, step-segmentation, quality-assessment, fine-grained-actions, healthcare: Overview Fine-grained framework that segments WHO handwashing steps and scores key actions jointly using a multi-stage conv-transformer architecture.
- [ ] **Handwashing Action Detection System for an Autonomous Social Robot (2022)** (`papers/2022-arxiv-2210.15804`) — tags: handwashing, action-detection, robotics, real-time, feedback: Overview Vision model that detects WHO handwashing steps on video to drive an autonomous social robot for real-time coaching and feedback.
- [ ] **Shadow Augmentation for Handwashing Action Recognition (MMSP 2024)** (`papers/2024-mmsp-shadow-augmentation`) — tags: handwashing, data-augmentation, domain-robustness, synthetic-data, shadow-invariance: Overview Studies how shadow-induced domain shift degrades handwashing action recognition and proposes shadow augmentation to improve model robustness.
- [ ] **Hand Washing Gesture Recognition Using Synthetic Dataset (J. Imaging 2025)** (`papers/2025-mdpi-synthetic`) — tags: synthetic, data-augmentation, rgb, depth, yolo: Overview Introduces a 96k-frame synthetic dataset with RGB, depth, and masks covering 8 WHO-inspired gestures, generated using Blender rendering.

## Datasets
- [ ] **Class23 Open-Room Handwash Dataset** (`datasets/class23-open-room`) — tags: open-room, multi-view, restricted, rgb-video: Overview Open-room deployment dataset with 105 videos collected from uncontrolled environments (cafeterias, restrooms) with overhead camera views.
- [ ] **HHA300 Hand Hygiene Assessment Dataset** (`datasets/hha300`) — tags: who-steps, quality-scoring, non-public, rgb-video, research: Overview Hospital hand hygiene dataset with 300 videos and quality scores (0-10 scale) based on WHO compliance.
- [ ] **Kaggle Handwash (Resorted to WHO 6+1 Classes)** (`datasets/kaggle-who6`) — tags: who-steps, public, rgb-video, kaggle, 7-classes: Overview Public subset of the Kaggle hand-wash videos, re-sorted into 7 folders to align with WHO steps (left/right merged; wrist/rinse in "Other").
- [ ] **METC Lab Handwashing Dataset** (`datasets/metc-lab`) — tags: who-steps, lab, rgb-video, public, zenodo: Overview Lab-collected WHO handwash recordings from the Medical Education Technology Center (Riga Stradins University) with multiple camera interfaces.
- [ ] **OCDetect Wrist-Worn IMU Dataset** (`datasets/ocdetect-wrist`) — tags: wearable, imu, public, ocd, event-detection: Overview Smartwatch IMU dataset for detecting compulsive handwashing behaviors (OCD) with accelerometer and gyroscope data.
- [ ] **Portable51 & Farm23 Shadow Augmentation Datasets** (`datasets/portable51-farm23`) — tags: outdoor, shadows, non-public, robustness: Overview Two shadow robustness datasets (Portable51, Farm23) for testing handwashing models under challenging shadow conditions.
- [ ] **PSKUS Hospital Handwashing Dataset** (`datasets/pskus-hospital`) — tags: who-steps, hospital, rgb-video, public, zenodo: Overview Real-world hospital videos with 3,185 hand-washing episodes annotated frame-by-frame using WHO movement codes.
- [ ] **Stanford Depth Camera Hand Hygiene Dataset** (`datasets/stanford-depth`) — tags: depth, privacy-preserving, non-public, hospital: Overview Privacy-preserving depth camera dataset capturing ~20 hours of hospital hand hygiene with silhouette-based anonymization.
- [ ] **Synthetic Hand-Washing Gesture Dataset (Özakar & Gedikli, 2025)** (`datasets/synthetic-blender-rozakar`) — tags: synthetic, rgb, depth, segmentation, public: Overview CGI-rendered dataset with 96,000 frames (64 minutes) across 8 WHO-inspired gestures, 4 characters, and 4 environments (RGB, depth, masks).
- [ ] **UWash Smartwatch Dataset** (`datasets/uwash-smartwatch`) — tags: imu, wearable, smartwatch, public, quality-assessment: Overview Smartwatch IMU dataset for handwashing quality assessment with accelerometer, gyroscope, and magnetometer streams.
- [ ] **wearPuck Multimodal Handwash Dataset** (`datasets/wearpuck`) — tags: imu, environmental, wearable, public, multimodal: Overview Open-source wrist-worn sensor dataset capturing 43 hand-washing events over ~10 hours of recordings (highly imbalanced) with IMU and environmental sensors.

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
