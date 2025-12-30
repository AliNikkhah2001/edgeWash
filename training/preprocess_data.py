"""
Data preprocessing module for handwashing detection training pipeline.

Handles frame extraction from videos, dataset organization, and train/val/test splitting.
"""

import sys
import json
import logging
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Import configuration
from config import (
    DATASETS,
    RAW_DIR,
    PROCESSED_DIR,
    IMG_SIZE,
    FRAME_SKIP,
    NUM_CLASSES,
    CLASS_NAMES,
    KAGGLE_CLASS_MAPPING,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    RANDOM_SEED,
    LOG_FORMAT,
    LOG_DATE_FORMAT
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt=LOG_DATE_FORMAT
)
logger = logging.getLogger(__name__)

# Set random seed
np.random.seed(RANDOM_SEED)


def extract_frames_from_video(
    video_path: str,
    output_dir: Path,
    img_size: Tuple[int, int] = IMG_SIZE,
    frame_skip: int = FRAME_SKIP,
    max_frames: Optional[int] = None
) -> List[Path]:
    """
    Extract frames from video and save as JPEG images.
    
    Args:
        video_path: Path to input video file
        output_dir: Directory to save extracted frames
        img_size: Target image size (width, height)
        frame_skip: Extract every Nth frame (1=all, 2=every other)
        max_frames: Maximum number of frames to extract (None=all)
    
    Returns:
        List of paths to extracted frame images
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return []
    
    frame_paths = []
    frame_idx = 0
    saved_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % frame_skip == 0:
            # Resize frame
            frame_resized = cv2.resize(frame, img_size)
            
            # Save frame as JPEG
            frame_path = output_dir / f"frame_{saved_idx:06d}.jpg"
            cv2.imwrite(str(frame_path), frame_resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
            frame_paths.append(frame_path)
            saved_idx += 1
            
            if max_frames and saved_idx >= max_frames:
                break
        
        frame_idx += 1
    
    cap.release()
    return frame_paths


def load_kaggle_dataset(kaggle_dir: Path) -> pd.DataFrame:
    """
    Load Kaggle WHO6 dataset from folder structure.
    
    Dataset structure: kaggle-dataset-6classes/step1/, step2/, ..., other/
    
    Args:
        kaggle_dir: Path to Kaggle dataset directory
    
    Returns:
        DataFrame with columns: video_path, class_id, class_name, dataset
    """
    records = []
    
    dataset_root = kaggle_dir / 'kaggle-dataset-6classes'
    if not dataset_root.exists():
        logger.error(f"Kaggle dataset not found: {dataset_root}")
        return pd.DataFrame(records)
    
    # Iterate through class folders
    for class_folder in sorted(dataset_root.iterdir()):
        if not class_folder.is_dir():
            continue
        
        class_name_lower = class_folder.name.lower()
        class_id = KAGGLE_CLASS_MAPPING.get(class_name_lower, 0)
        
        # Find all video files in this class folder
        for video_file in class_folder.glob('*'):
            if video_file.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
                records.append({
                    'video_path': str(video_file),
                    'class_id': class_id,
                    'class_name': CLASS_NAMES[class_id],
                    'dataset': 'kaggle'
                })
    
    return pd.DataFrame(records)


def load_pskus_annotations(pskus_dir: Path) -> List[dict]:
    """
    Load PSKUS Hospital dataset annotations (frame-level).
    
    Args:
        pskus_dir: Path to PSKUS dataset directory
    
    Returns:
        List of annotation dictionaries
    """
    annotations = []
    
    # Iterate through DataSet folders
    for dataset_folder in sorted(pskus_dir.glob('DataSet*')):
        if not dataset_folder.is_dir():
            continue
        
        # Find all video files
        for video_path in dataset_folder.glob('*.mp4'):
            # Find corresponding JSON annotation
            json_path = video_path.with_suffix('.json')
            if not json_path.exists():
                logger.warning(f"No annotations for: {video_path}")
                continue
            
            # Load frame-level annotations
            try:
                with open(json_path, 'r') as f:
                    frame_annotations = json.load(f)
                
                annotations.append({
                    'video_path': str(video_path),
                    'annotations': frame_annotations,
                    'dataset': 'pskus',
                    'type': 'frame-level'
                })
            except Exception as e:
                logger.error(f"Failed to load annotations from {json_path}: {e}")
    
    return annotations


def load_metc_annotations(metc_dir: Path) -> List[dict]:
    """
    Load METC Lab dataset annotations (frame-level).
    
    Args:
        metc_dir: Path to METC dataset directory
    
    Returns:
        List of annotation dictionaries
    """
    annotations = []
    
    # Iterate through Interface folders
    for interface_folder in sorted(metc_dir.glob('Interface_number_*')):
        if not interface_folder.is_dir():
            continue
        
        # Find all video files
        for video_path in interface_folder.glob('*.mp4'):
            json_path = video_path.with_suffix('.json')
            if not json_path.exists():
                logger.warning(f"No annotations for: {video_path}")
                continue
            
            try:
                with open(json_path, 'r') as f:
                    frame_annotations = json.load(f)
                
                annotations.append({
                    'video_path': str(video_path),
                    'annotations': frame_annotations,
                    'dataset': 'metc',
                    'type': 'frame-level'
                })
            except Exception as e:
                logger.error(f"Failed to load annotations from {json_path}: {e}")
    
    return annotations


def preprocess_clip_level_dataset(
    video_df: pd.DataFrame,
    output_dir: Path
) -> pd.DataFrame:
    """
    Preprocess clip-level dataset (Kaggle).
    Extract frames from videos where entire clip has same label.
    
    Args:
        video_df: DataFrame with video_path, class_id columns
        output_dir: Directory to save extracted frames
    
    Returns:
        DataFrame with frame_path, class_id, video_id, frame_idx, dataset columns
    """
    records = []
    
    for idx, row in tqdm(video_df.iterrows(), total=len(video_df), desc="Processing videos"):
        video_path = row['video_path']
        class_id = row['class_id']
        dataset_name = row['dataset']
        
        video_id = Path(video_path).stem
        frames_dir = output_dir / dataset_name / f"video_{idx:04d}_{video_id}"
        
        # Skip if already processed
        if frames_dir.exists() and any(frames_dir.iterdir()):
            existing_frames = sorted(frames_dir.glob('*.jpg'))
            for frame_path in existing_frames:
                frame_idx = int(frame_path.stem.split('_')[-1])
                records.append({
                    'frame_path': str(frame_path),
                    'class_id': class_id,
                    'class_name': CLASS_NAMES[class_id],
                    'video_id': video_id,
                    'frame_idx': frame_idx,
                    'dataset': dataset_name
                })
            continue
        
        # Extract frames
        frame_paths = extract_frames_from_video(video_path, frames_dir)
        
        # All frames in clip have same label
        for frame_idx, frame_path in enumerate(frame_paths):
            records.append({
                'frame_path': str(frame_path),
                'class_id': class_id,
                'class_name': CLASS_NAMES[class_id],
                'video_id': video_id,
                'frame_idx': frame_idx,
                'dataset': dataset_name
            })
    
    return pd.DataFrame(records)


def preprocess_frame_level_dataset(
    annotations: List[dict],
    output_dir: Path,
    dataset_name: str
) -> pd.DataFrame:
    """
    Preprocess frame-level dataset (PSKUS, METC).
    Extract frames and match with frame-level annotations.
    
    Args:
        annotations: List of annotation dictionaries
        output_dir: Directory to save extracted frames
        dataset_name: Name of dataset (for folder organization)
    
    Returns:
        DataFrame with frame_path, class_id, video_id, frame_idx, dataset columns
    """
    records = []
    
    for video_ann in tqdm(annotations, desc=f"Processing {dataset_name}"):
        video_path = video_ann['video_path']
        frame_annotations = video_ann['annotations']
        
        video_id = Path(video_path).stem
        frames_dir = output_dir / dataset_name / video_id
        
        # Skip if already processed
        if frames_dir.exists() and any(frames_dir.iterdir()):
            existing_frames = sorted(frames_dir.glob('*.jpg'))
            for frame_idx, frame_path in enumerate(existing_frames):
                # Get annotation for this frame
                if frame_idx < len(frame_annotations):
                    ann = frame_annotations[frame_idx]
                    class_id = ann.get('movement_code', 0)
                else:
                    class_id = 0  # Default to "Other"
                
                records.append({
                    'frame_path': str(frame_path),
                    'class_id': class_id,
                    'class_name': CLASS_NAMES[class_id],
                    'video_id': video_id,
                    'frame_idx': frame_idx,
                    'dataset': dataset_name
                })
            continue
        
        # Extract frames
        frame_paths = extract_frames_from_video(video_path, frames_dir)
        
        # Match frames with annotations
        for frame_idx, frame_path in enumerate(frame_paths):
            if frame_idx < len(frame_annotations):
                ann = frame_annotations[frame_idx]
                class_id = ann.get('movement_code', 0)
            else:
                class_id = 0
            
            records.append({
                'frame_path': str(frame_path),
                'class_id': class_id,
                'class_name': CLASS_NAMES[class_id],
                'video_id': video_id,
                'frame_idx': frame_idx,
                'dataset': dataset_name
            })
    
    return pd.DataFrame(records)


def split_dataset(
    frames_df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    random_state: int = RANDOM_SEED
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train/val/test by video (to avoid data leakage).
    
    Args:
        frames_df: DataFrame with frame data
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        random_state: Random seed for reproducibility
    
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    # Get unique videos
    unique_videos = frames_df['video_id'].unique()
    video_to_class = frames_df.groupby('video_id')['class_id'].first()
    
    # Stratified split by class
    train_videos, temp_videos = train_test_split(
        unique_videos,
        test_size=(val_ratio + test_ratio),
        random_state=random_state,
        stratify=video_to_class[unique_videos]
    )
    
    val_videos, test_videos = train_test_split(
        temp_videos,
        test_size=test_ratio / (val_ratio + test_ratio),
        random_state=random_state,
        stratify=video_to_class[temp_videos]
    )
    
    # Create split dataframes
    train_df = frames_df[frames_df['video_id'].isin(train_videos)].reset_index(drop=True)
    val_df = frames_df[frames_df['video_id'].isin(val_videos)].reset_index(drop=True)
    test_df = frames_df[frames_df['video_id'].isin(test_videos)].reset_index(drop=True)
    
    return train_df, val_df, test_df


def preprocess_all_datasets(use_kaggle: bool = True, use_pskus: bool = False, use_metc: bool = False) -> dict:
    """
    Preprocess all available datasets.
    
    Args:
        use_kaggle: Process Kaggle dataset
        use_pskus: Process PSKUS dataset
        use_metc: Process METC dataset
    
    Returns:
        Dictionary with paths to processed CSV files
    """
    logger.info("=" * 80)
    logger.info("DATASET PREPROCESSING")
    logger.info("=" * 80)
    
    all_frames = []
    
    # Process Kaggle dataset
    if use_kaggle:
        kaggle_dir = RAW_DIR / 'kaggle'
        kaggle_extracted = kaggle_dir / 'kaggle-dataset-6classes'
        
        if kaggle_extracted.exists():
            logger.info("\n[1/3] Processing Kaggle dataset...")
            kaggle_df = load_kaggle_dataset(kaggle_dir)
            logger.info(f"Loaded {len(kaggle_df)} videos")
            
            kaggle_frames = preprocess_clip_level_dataset(
                kaggle_df,
                PROCESSED_DIR
            )
            logger.info(f"Extracted {len(kaggle_frames)} frames")
            all_frames.append(kaggle_frames)
        else:
            logger.warning("Kaggle dataset not found, skipping")
    
    # Process PSKUS dataset
    if use_pskus:
        pskus_dir = RAW_DIR / 'pskus'
        if (pskus_dir / 'summary.csv').exists():
            logger.info("\n[2/3] Processing PSKUS dataset...")
            pskus_annotations = load_pskus_annotations(pskus_dir)
            logger.info(f"Loaded {len(pskus_annotations)} videos")
            
            pskus_frames = preprocess_frame_level_dataset(
                pskus_annotations,
                PROCESSED_DIR,
                'pskus'
            )
            logger.info(f"Extracted {len(pskus_frames)} frames")
            all_frames.append(pskus_frames)
        else:
            logger.warning("PSKUS dataset not found, skipping")
    
    # Process METC dataset
    if use_metc:
        metc_dir = RAW_DIR / 'metc'
        if (metc_dir / 'summary.csv').exists():
            logger.info("\n[3/3] Processing METC dataset...")
            metc_annotations = load_metc_annotations(metc_dir)
            logger.info(f"Loaded {len(metc_annotations)} videos")
            
            metc_frames = preprocess_frame_level_dataset(
                metc_annotations,
                PROCESSED_DIR,
                'metc'
            )
            logger.info(f"Extracted {len(metc_frames)} frames")
            all_frames.append(metc_frames)
        else:
            logger.warning("METC dataset not found, skipping")
    
    # Combine all datasets
    if not all_frames:
        logger.error("No datasets processed!")
        return {}
    
    combined_df = pd.concat(all_frames, ignore_index=True)
    
    # Save combined dataset
    combined_csv = PROCESSED_DIR / 'frames.csv'
    combined_df.to_csv(combined_csv, index=False)
    logger.info(f"\nCombined dataset: {len(combined_df)} frames")
    logger.info(f"Saved to: {combined_csv}")
    
    # Class distribution
    logger.info("\nClass distribution:")
    class_counts = combined_df['class_name'].value_counts()
    for class_name, count in class_counts.items():
        logger.info(f"  {class_name}: {count} frames ({count/len(combined_df)*100:.1f}%)")
    
    # Split dataset
    logger.info("\nSplitting dataset...")
    train_df, val_df, test_df = split_dataset(combined_df)
    
    # Save splits
    train_csv = PROCESSED_DIR / 'train.csv'
    val_csv = PROCESSED_DIR / 'val.csv'
    test_csv = PROCESSED_DIR / 'test.csv'
    
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    
    logger.info(f"Train: {len(train_df)} frames ({len(train_df['video_id'].unique())} videos)")
    logger.info(f"Val:   {len(val_df)} frames ({len(val_df['video_id'].unique())} videos)")
    logger.info(f"Test:  {len(test_df)} frames ({len(test_df['video_id'].unique())} videos)")
    
    logger.info("\nPreprocessing complete!")
    
    return {
        'combined': str(combined_csv),
        'train': str(train_csv),
        'val': str(val_csv),
        'test': str(test_csv)
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Preprocess handwashing datasets"
    )
    parser.add_argument(
        '--kaggle',
        action='store_true',
        default=True,
        help='Process Kaggle dataset (default: True)'
    )
    parser.add_argument(
        '--pskus',
        action='store_true',
        help='Process PSKUS dataset'
    )
    parser.add_argument(
        '--metc',
        action='store_true',
        help='Process METC dataset'
    )
    
    args = parser.parse_args()
    
    result = preprocess_all_datasets(
        use_kaggle=args.kaggle,
        use_pskus=args.pskus,
        use_metc=args.metc
    )
    
    if result:
        logger.info(f"\nProcessed files:")
        for key, path in result.items():
            logger.info(f"  {key}: {path}")
