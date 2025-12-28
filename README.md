# Handwashing Research Hub

Aggregated code, papers, datasets, models, and experiment ideas for automated handwashing assessment.
_Last updated: 2025-12-28 12:32 UTC_

## Structure
- `code/`: cloned codebases and pipelines
- `papers/`: papers with `summary.md`, `tags.md`, and `paper.pdf`
- `datasets/`: storage location for raw/processed datasets (gitignored)
- `models/`: exported weights and model cards
- `evaluation/`: benchmarks and result artifacts
- `ideas/`: future experiment notes and design sketches

## Codebases
- [ ] **TensorFlow Handwash Monitoring Demo** (`code/Handwash-Monitoring`) — tags: depth, cnn, dispenser-detection, tensorflow — source: https://github.com/SidhantSarkar/Handwash-Monitoring: Lightweight TF/Keras prototype that detects hand presence and dispenser use from depth video near sinks. Includes training and inference scripts for processing depth frames.
- [ ] **UWash Smartwatch Handwashing Assessment** (`code/UWash`) — tags: wearable, imu, smartwatch, quality-assessment — source: https://github.com/aiotgroup/UWash: IMU-based smartwatch pipeline for detecting and scoring handwashing quality/compliance, with preprocessing and model training scripts.
- [ ] **WashWise Real-Time Step Tracker** (`code/WashWise`) — tags: yolo, real-time, who-steps, streamlit, roboflow — source: https://github.com/aarnavshah12/WashWise: YOLOv8-based demo from Roboflow for real-time WHO step classification on sink camera footage with a Streamlit UI.
- [ ] **EdgeWash WHO Movement Classifier** (`code/edgewash`) — tags: who-steps, edge, classification, vision — source: https://github.com/AliNikkhah2001/edgeWash: Lightweight edge-focused pipeline for WHO hand-washing movement recognition with exportable on-device models.
- [ ] **EDI-Riga Handwashing Movement Classifiers** (`code/edi-riga-handwash`) — tags: hospital, who-steps, classification, pytorch — source: https://github.com/edi-riga/handwash: Hospital-focused WHO step classifiers with training scripts and pretrained checkpoints maintained by EDI-Riga.
- [ ] **Hand-Wash Compliance Detection with YOLO** (`code/hand-wash-compliance-yolo`) — tags: yolo, compliance-detection, vision — source: https://github.com/dpt-xyz/hand-wash-compliance-yolo: YOLO-based detector for monitoring hand presence and compliance near sinks, including dataset prep and training utilities.
- [ ] **HandWash Multimodal Surgical Rub Recognition** (`code/huiwen-HandWash`) — tags: surgical, multimodal, feedback, vision — source: https://github.com/huiwen99/HandWash: Multimodal hand-rub recognition project combining RGB video and additional cues, plus a feedback UI for user guidance.
- [ ] **Synthetic Hand-Washing Data Generation** (`code/synthetic-hand-washing`) — tags: synthetic-data, generation, augmentation — source: https://github.com/r-ozakar/synthetic-hand-washing: Rendering and augmentation toolkit for creating synthetic handwashing datasets plus training recipes for baseline models.
- [ ] **wearPuck Multimodal Wearable Toolkit** (`code/wearPuck`) — tags: wearable, imu, environmental, hardware — source: https://github.com/kristofvl/wearPuck: Firmware and data tooling for the wearPuck device capturing IMU and environmental signals for handwashing analytics.

## Papers
- [ ] **Vision-Based Hand Hygiene Monitoring with Depth Cameras (Stanford PAIR 2015)** (`papers/2015-stanford-depth`) — tags: depth, privacy, hospital, non-public, cnn: Early depth-only system using overhead sensors near soap dispensers to detect hand presence and dispenser use. Demonstrates lightweight CNNs on depth frames achieving high precision while preserving privacy; dataset not public.
- [ ] **Hand-Hygiene Activity Recognition in Egocentric Video (MMSP 2019)** (`papers/2019-mmsp-chengzhang`) — tags: egocentric-video, handwashing, temporal-detection, two-stream-cnn, food-safety: Two-stage pipeline for detecting hand-hygiene actions in egocentric videos from food-manufacturing environments. Temporal localization uses motion and hand-mask cues, followed by a two-stream CNN with search to classify actions, reaching ~80% detection accuracy on a 100-participant dataset.
- [ ] **Automated Quality Assessment of Hand Washing Using Deep Learning (2020)** (`papers/2020-arxiv-2011.11383`) — tags: handwashing, quality-assessment, mobilenet, transfer-learning, hospital: Early work on frame-level WHO step recognition using MobileNetV2 and Xception. Evaluates hospital footage; reports ~64% frame accuracy and highlights challenges with real-world variability.
- [ ] **Designing a Computer-Vision Application: Hand-Hygiene Assessment in an Open-Room Environment (J. Imaging 2021)** (`papers/2021-jimaging-chengzhang`) — tags: handwashing, deployment, dataset-design, open-world, computer-vision: Case study of deploying a hand-hygiene assessment system in an open-room setting. Discusses dataset design, environmental challenges, and end-to-end engineering considerations for reliable step recognition beyond controlled lab conditions.
- [ ] **Hand-Washing Video Dataset Annotated According to the WHO Guidelines (Data 2021)** (`papers/2021-mdpi-who-dataset`) — tags: dataset, who-steps, hospital, annotations, public: Presents the PSKUS hospital dataset with 3,185 annotated episodes and detailed statistics. Defines the 6 WHO movement classes (+ Other) and provides per-frame labels to enable step-wise recognition research.
- [ ] **Hand Hygiene Assessment via Joint Step Segmentation and Key Action Scorer (2023)** (`papers/2022-arxiv-2209.12221`) — tags: handwashing, step-segmentation, quality-assessment, fine-grained-actions, healthcare: Fine-grained framework that segments WHO handwashing steps and scores key actions jointly. Uses multi-stage learning with segmentation cues to improve per-step assessment quality, outperforming coarse video-level classifiers on hospital hand-hygiene videos.
- [ ] **Handwashing Action Detection System for an Autonomous Social Robot (2022)** (`papers/2022-arxiv-2210.15804`) — tags: handwashing, action-detection, robotics, real-time, feedback: Vision model that detects WHO handwashing steps on video to drive an autonomous social robot delivering hygiene coaching. The authors combine action detection with a lightweight deployment target to give real-time feedback during hand-rub sessions and validate performance on in-the-wild recordings.
- [ ] **Shadow Augmentation for Handwashing Action Recognition: From Synthetic to Real Datasets (MMSP 2024)** (`papers/2024-mmsp-shadow-augmentation`) — tags: handwashing, data-augmentation, domain-robustness, synthetic-data, shadow-invariance: Studies how shadow-induced domain shift degrades handwashing action recognition and proposes shadow augmentation to harden models. Systematically varies synthetic shadow attributes, then transfers augmentation to real data to improve robustness across architectures and datasets.
- [ ] **Hand Washing Gesture Recognition Using Synthetic Dataset (J. Imaging 2025)** (`papers/2025-mdpi-synthetic`) — tags: synthetic, data-augmentation, rgb, depth, yolo: Introduces a 96k-frame synthetic dataset with RGB, depth, and masks covering 8 WHO-inspired gestures. Benchmarks InceptionV3, YOLOv8n (cls/seg), and PointNet; synthetic-trained YOLOv8n-seg reaches ~79% on a small real test set.

## Datasets
- [ ] **Class23 Open-Room Hand Hygiene Dataset (2024, restricted)** (`datasets/class23-open-room`) — tags: open-room, multi-view, restricted, rgb-video: 105 untrimmed videos from 23 participants washing hands in a food lab across three camera setups. Captured at 1080p/30FPS; used for open-room action recognition/detection. Not publicly released due to consent restrictions.
- [ ] **HHA300 Hand Hygiene Assessment Dataset (2023, non-public)** (`datasets/hha300`) — tags: who-steps, quality-scoring, non-public, rgb-video, research: 300 videos (60 participants) labeled frame-by-frame with WHO steps plus quality scores for each step. Introduced with a multi-stage conv-transformer for joint segmentation and scoring. Dataset is not publicly released.
- [ ] **Kaggle Handwash (Resorted to WHO 6+1 Classes)** (`datasets/kaggle-who6`) — tags: who-steps, public, rgb-video, kaggle, 7-classes: Public subset of the Kaggle hand-wash videos, re-sorted into 7 folders to align with WHO steps (left/right merged; wrist/rinse in "Other"). ~292 short videos.
- [ ] **METC Lab Handwashing Dataset** (`datasets/metc-lab`) — tags: who-steps, lab, rgb-video, public, zenodo: Lab-collected WHO handwash recordings from the Medical Education Technology Center (Riga Stradins University). Multiple camera interfaces, annotated with WHO step labels; aligns with PSKUS label scheme (6 steps + "Other").
- [ ] **OCDetect Wrist Motion Dataset (2023)** (`datasets/ocdetect-wrist`) — tags: wearable, imu, public, ocd, event-detection: Smartwatch motion dataset for detecting compulsive handwashing behaviors. 22 participants over 28 days (~3,000 handwashing events). Focuses on event detection rather than WHO step coverage.
- [ ] **Portable51 & Farm23 Shadow Datasets (2024, non-public)** (`datasets/portable51-farm23`) — tags: outdoor, shadows, non-public, robustness: Outdoor handwashing videos recorded at portable sinks (51 clips) and farm environments (23 clips) to study shadow-induced domain shift. Used in the 2024 shadow augmentation paper.
- [ ] **PSKUS Hospital Handwashing Dataset** (`datasets/pskus-hospital`) — tags: who-steps, hospital, rgb-video, public, zenodo: Real-world hospital videos (Latvia) with 3,185 hand-washing episodes annotated frame-by-frame using WHO movement codes. Includes `summary.csv`, `statistics.csv`, and per-frame labels (`is_washing`, `movement_code`, `frame_time`). Resolution 320x240 or 640x480 at 30 FPS.
- [ ] **Stanford Depth Hand Hygiene Dataset (2015, non-public)** (`datasets/stanford-depth`) — tags: depth, privacy-preserving, non-public, hospital: ~20 hours of depth video from overhead cameras near soap dispensers in a hospital. Used to detect dispenser use and hand presence with lightweight CNNs on depth frames.
- [ ] **Synthetic Hand-Washing Gesture Dataset (Özakar & Gedikli, 2025)** (`datasets/synthetic-blender-rozakar`) — tags: synthetic, rgb, depth, segmentation, public: CGI-rendered dataset with 96,000 frames (64 minutes) across 8 WHO-inspired gestures, 4 characters, and 4 environments (RGB, depth, masks). Ships with pre-trained models (InceptionV3, YOLOv8n, PointNet) in the upstream repo.
- [ ] **UWash Smartwatch Dataset (2023)** (`datasets/uwash-smartwatch`) — tags: imu, wearable, smartwatch, public, quality-assessment: Smartwatch IMU dataset for handwashing quality assessment. Multiple sessions per participant with accelerometer/gyroscope/magnetometer streams; processed via provided scripts into train/test splits for 10 classes.
- [ ] **wearPuck Multimodal Handwash Dataset (2024)** (`datasets/wearpuck`) — tags: imu, environmental, wearable, public, multimodal: Open-source wrist-worn sensor dataset capturing 43 hand-washing events over ~10 hours of recordings (highly imbalanced). Sensors: IMU (acc/gyro) + humidity/temperature/pressure. Collected with the wearPuck device; provides raw CSVs and preprocessing scripts.

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
