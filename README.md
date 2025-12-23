# Handwashing Research Hub

Aggregated code, papers, datasets, models, and experiment ideas for automated handwashing assessment.

## Structure
- `code/`: cloned codebases and pipelines
- `papers/`: papers with `summary.md`, `tags.md`, and `paper.pdf`
- `datasets/`: storage location for raw/processed datasets (gitignored)
- `models/`: exported weights and model cards
- `evaluation/`: benchmarks and result artifacts
- `ideas/`: future experiment notes and design sketches

## Codebases
- **EdgeWash (WHO hand-washing movement classification)** (`code/edgewash`) — source: https://github.com/AliNikkhah2001/edgeWash
- **Hand-wash compliance detection with YOLO** (`code/hand-wash-compliance-yolo`) — source: https://github.com/dpt-xyz/hand-wash-compliance-yolo
- **HandWash (surgical hand-rub recognition with multimodal cues)** (`code/huiwen-HandWash`) — source: https://github.com/huiwen99/HandWash
- **Synthetic hand-washing data generation** (`code/synthetic-hand-washing`) — source: https://github.com/r-ozakar/synthetic-hand-washing

## Papers
- **Hand-Hygiene Activity Recognition in Egocentric Video (MMSP 2019)** (`papers/2019-mmsp-chengzhang`) — tags: egocentric-video, handwashing, temporal-detection, two-stream-cnn, food-safety: Two-stage pipeline for detecting hand-hygiene actions in egocentric videos from food-manufacturing environments. Temporal localization uses motion and hand-mask cues, followed by a two-stream CNN with search to classify actions, reaching ~80% detection accuracy on a 100-participant dataset.
- **Designing a Computer-Vision Application: Hand-Hygiene Assessment in an Open-Room Environment (J. Imaging 2021)** (`papers/2021-jimaging-chengzhang`) — tags: handwashing, deployment, dataset-design, open-world, computer-vision: Case study of deploying a hand-hygiene assessment system in an open-room setting. Discusses dataset design, environmental challenges, and end-to-end engineering considerations for reliable step recognition beyond controlled lab conditions.
- **Hand Hygiene Assessment via Joint Step Segmentation and Key Action Scorer (2023)** (`papers/2022-arxiv-2209.12221`) — tags: handwashing, step-segmentation, quality-assessment, fine-grained-actions, healthcare: Fine-grained framework that segments WHO handwashing steps and scores key actions jointly. Uses multi-stage learning with segmentation cues to improve per-step assessment quality, outperforming coarse video-level classifiers on hospital hand-hygiene videos.
- **Handwashing Action Detection System for an Autonomous Social Robot (2022)** (`papers/2022-arxiv-2210.15804`) — tags: handwashing, action-detection, robotics, real-time, feedback: Vision model that detects WHO handwashing steps on video to drive an autonomous social robot delivering hygiene coaching. The authors combine action detection with a lightweight deployment target to give real-time feedback during hand-rub sessions and validate performance on in-the-wild recordings.
- **Shadow Augmentation for Handwashing Action Recognition: From Synthetic to Real Datasets (MMSP 2024)** (`papers/2024-mmsp-shadow-augmentation`) — tags: handwashing, data-augmentation, domain-robustness, synthetic-data, shadow-invariance: Studies how shadow-induced domain shift degrades handwashing action recognition and proposes shadow augmentation to harden models. Systematically varies synthetic shadow attributes, then transfers augmentation to real data to improve robustness across architectures and datasets.

## Models
- Staging area for trained weights and model cards.

## Evaluation
- Add benchmark summaries or result notebooks here.

## Ideas
- Roadmap (ideas/roadmap.md)

## Automation
- `scripts/build_readme.py` regenerates this README from folder metadata.
- `.github/workflows/build-readme.yml` runs the generator on each push and commits changes.

To add new assets, drop them in the appropriate folder with minimal metadata; the automation will refresh this page.
