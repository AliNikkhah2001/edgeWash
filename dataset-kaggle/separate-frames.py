#!/usr/bin/env python3
"""Streaming-friendly preprocessing for the Kaggle handwash dataset.

This script replaces the legacy "copy every video and dump every frame" flow
with a lighter pipeline that:

- **avoids duplicating videos** (uses hardlinks/symlinks only when requested),
- **samples a small, even spread of frames** instead of extracting every frame,
- **resizes and JPEG-compresses frames** to keep disk usage minimal,
- **logs progress with a progress bar** and structured messages, and
- **parallelises** video processing across CPU cores.

Outputs mirror the expected train/val/test folder structure so the existing
training scripts continue to work.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import numpy as np
from tqdm import tqdm

TEST_PROPORTION = 0.3
DEFAULT_FRAMES_PER_VIDEO = 48
MAX_FPS = 5  # upper bound when sampling densely recorded videos
TARGET_LONG_EDGE = 360  # resize while preserving aspect ratio
JPEG_QUALITY = 85


@dataclass
class PreprocessConfig:
    input_dir: Path
    output_dir: Path
    test_proportion: float = TEST_PROPORTION
    frames_per_video: int = DEFAULT_FRAMES_PER_VIDEO
    max_fps: int = MAX_FPS
    target_long_edge: int = TARGET_LONG_EDGE
    jpeg_quality: int = JPEG_QUALITY
    seed: int = 123
    keep_video_links: bool = False

    @property
    def frames_dir(self) -> Path:
        return self.output_dir / "frames"

    @property
    def videos_dir(self) -> Path:
        return self.output_dir / "videos"


@dataclass
class FrameSummary:
    video: Path
    subset: str
    class_id: str
    frames_written: int


def configure_logging() -> None:
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)s: %(message)s",
        level=logging.INFO,
    )


def ensure_dirs(config: PreprocessConfig, class_ids: Iterable[str]) -> None:
    for subset in ("trainval", "test"):
        for class_id in class_ids:
            (config.frames_dir / subset / class_id).mkdir(parents=True, exist_ok=True)
            if config.keep_video_links:
                (config.videos_dir / subset / class_id).mkdir(parents=True, exist_ok=True)


def compute_resize_dims(frame_shape: Tuple[int, int, int], target_long_edge: int) -> Tuple[int, int]:
    height, width = frame_shape[:2]
    if target_long_edge <= 0:
        return width, height
    long_edge = max(height, width)
    scale = target_long_edge / long_edge
    new_w, new_h = int(round(width * scale)), int(round(height * scale))
    return max(new_w, 1), max(new_h, 1)


def choose_frame_indices(frame_count: int, fps: float, config: PreprocessConfig) -> List[int]:
    if frame_count == 0:
        return []
    step = max(int(round(fps / config.max_fps)), 1) if fps > 0 else 1
    dense_indices = list(range(0, frame_count, step))
    if len(dense_indices) <= config.frames_per_video:
        return dense_indices
    linspace_idx = np.linspace(0, len(dense_indices) - 1, config.frames_per_video)
    return sorted({dense_indices[int(i)] for i in linspace_idx})


def write_frames(video_path: Path, dest_dir: Path, frame_indices: List[int], config: PreprocessConfig) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logging.warning("Failed to open %s", video_path)
        return 0

    frames_written = 0
    resize_cache: Tuple[int, int] | None = None
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            break
        if resize_cache is None:
            resize_cache = compute_resize_dims(frame.shape, config.target_long_edge)
        if resize_cache != (frame.shape[1], frame.shape[0]):
            frame = cv2.resize(frame, resize_cache, interpolation=cv2.INTER_AREA)
        out_name = f"f{idx:05d}_{video_path.stem}.jpg"
        out_path = dest_dir / out_name
        cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), config.jpeg_quality])
        frames_written += 1

    cap.release()
    return frames_written


def process_video(video: Path, subset: str, config: PreprocessConfig) -> FrameSummary:
    class_id = video.parent.name
    dest_dir = config.frames_dir / subset / class_id

    cap = cv2.VideoCapture(str(video))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
    cap.release()

    frame_indices = choose_frame_indices(frame_count, fps, config)
    frames_written = write_frames(video, dest_dir, frame_indices, config)

    if config.keep_video_links:
        target_video = config.videos_dir / subset / class_id / video.name
        try:
            os.link(video, target_video)
        except OSError:
            target_video.symlink_to(video.resolve())

    return FrameSummary(video=video, subset=subset, class_id=class_id, frames_written=frames_written)


def _process_with_config(args: Tuple[Path, str, PreprocessConfig]) -> FrameSummary:
    video, subset, config = args
    return process_video(video, subset, config)


def gather_videos(input_dir: Path) -> List[Path]:
    return [p for p in input_dir.rglob("*.mp4") if p.is_file()]


def split_deterministically(videos: List[Path], test_proportion: float, seed: int) -> List[Tuple[Path, str]]:
    rng = random.Random(seed)
    assignments = []
    for video in sorted(videos):
        subset = "test" if rng.random() < test_proportion else "trainval"
        assignments.append((video, subset))
    return assignments


def parse_args() -> PreprocessConfig:
    parser = argparse.ArgumentParser(description="Lightweight Kaggle handwash preprocessing")
    parser.add_argument("input_dir", type=Path, help="Folder containing per-class video subdirectories")
    parser.add_argument("output_dir", type=Path, help="Where to write frames (and optional video links)")
    parser.add_argument("--frames-per-video", type=int, default=DEFAULT_FRAMES_PER_VIDEO,
                        help="Max frames to sample per video (evenly spaced)")
    parser.add_argument("--test-proportion", type=float, default=TEST_PROPORTION,
                        help="Probability of assigning a video to the test split")
    parser.add_argument("--max-fps", type=int, default=MAX_FPS,
                        help="Cap sampling frequency for high-FPS videos")
    parser.add_argument("--target-long-edge", type=int, default=TARGET_LONG_EDGE,
                        help="Resize so the longer edge matches this (0 to disable)")
    parser.add_argument("--jpeg-quality", type=int, default=JPEG_QUALITY, help="JPEG quality for saved frames")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for split reproducibility")
    parser.add_argument("--keep-video-links", action="store_true",
                        help="Create hardlinks/symlinks to videos instead of duplicating them")
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1),
                        help="Number of parallel workers")
    args = parser.parse_args()

    return PreprocessConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        test_proportion=args.test_proportion,
        frames_per_video=args.frames_per_video,
        max_fps=args.max_fps,
        target_long_edge=args.target_long_edge,
        jpeg_quality=args.jpeg_quality,
        seed=args.seed,
        keep_video_links=args.keep_video_links,
    ), args.workers


def main() -> None:
    configure_logging()
    config, workers = parse_args()

    logging.info("Input directory: %s", config.input_dir)
    logging.info("Output directory: %s", config.output_dir)

    videos = gather_videos(config.input_dir)
    if not videos:
        raise SystemExit(f"No .mp4 files found under {config.input_dir}")

    class_ids = sorted({p.parent.name for p in videos})
    ensure_dirs(config, class_ids)

    assignments = split_deterministically(videos, config.test_proportion, config.seed)
    logging.info("Prepared %d videos (%d classes)", len(assignments), len(class_ids))

    task_payloads = [(video, subset, config) for video, subset in assignments]

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(
            tqdm(
                executor.map(_process_with_config, task_payloads),
                total=len(assignments),
                desc="Extracting frames",
            )
        )

    total_frames = sum(r.frames_written for r in results)
    logging.info("Wrote %d frames across %d videos", total_frames, len(results))


if __name__ == "__main__":
    main()
