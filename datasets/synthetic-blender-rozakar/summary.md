# Synthetic Hand-Washing Gesture Dataset (Özakar & Gedikli, 2025)

Open synthetic dataset of 96,000 rendered frames (64 minutes) across 8 gestures, 4 characters, and 4 environments.

## Overview
- **Size:** 96,000 frames at 25 FPS; 960x540 resolution.
- **Breakdown:** 12,000 frames per gesture; 750 frames per environment per gesture.
- **Modalities:** RGB, depth, isolated depth, hand masks; point clouds derived from depth.
- **Labels:** per-frame gesture class (8).

## Structure
- **Assets:** multiple Google Drive archives (RGB, depth, masks, point clouds, pretrained models).
- **Derived data:** ROI crops (150x150 → 96x96) used for Inception/PointNet training.

## Availability
- **Access:** public via Google Drive links in upstream repo.
- **Download:** `./fetch.sh` (requires `gdown`).

## Notes
- **Use cases:** synthetic-to-real transfer; baselines in `code/synthetic-hand-washing`.
