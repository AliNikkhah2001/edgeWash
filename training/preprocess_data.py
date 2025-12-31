"""
Data preprocessing module for handwashing detection training pipeline.

Handles frame extraction from videos, dataset organization, and train/val/test splitting.
"""

import sys
import csv
import json
import random
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
    
    def _class_id_from_folder(folder_name: str) -> int:
        name_lower = folder_name.lower()
        if name_lower.isdigit():
            class_id = int(name_lower)
            if 0 <= class_id < len(CLASS_NAMES):
                return class_id
        digits = "".join(ch for ch in name_lower if ch.isdigit())
        if digits:
            class_id = int(digits)
            if 0 <= class_id < len(CLASS_NAMES):
                return class_id
        return KAGGLE_CLASS_MAPPING.get(name_lower, 0)

    # Iterate through class folders
    for class_folder in sorted(dataset_root.iterdir()):
        if not class_folder.is_dir():
            continue
        
        class_id = _class_id_from_folder(class_folder.name)
        
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


def _majority_vote(labels: List[int], total_movements: int) -> int:
    counts = [0] * total_movements
    for el in labels:
        counts[int(el)] += 1
    best = 0
    for i in range(1, total_movements):
        if counts[best] < counts[i]:
            best = i
    majority = (len(labels) + 2) // 2
    if counts[best] < majority:
        return -1
    return best


def _discount_reaction_indeterminacy(labels: List[int], reaction_frames: int) -> List[int]:
    new_labels = [u for u in labels]
    n = len(labels) - 1
    for i in range(n):
        if i == 0 or labels[i] != labels[i + 1] or i == n - 1:
            start = max(0, i - reaction_frames)
            end = i
            for j in range(start, end):
                new_labels[j] = -1
            start = i
            end = min(n + 1, i + reaction_frames)
            for j in range(start, end):
                new_labels[j] = -1
    return new_labels


def _select_frames_to_save(
    is_washing: List[int],
    codes: List[int],
    movement0_prop: float = 1.0
) -> dict:
    old_code = -1
    old_saved = False
    num_snippets = 0
    mapping = {}
    current_snippet = {}
    for i in range(len(is_washing)):
        new_code = codes[i]
        new_saved = (is_washing[i] == 2 and new_code != -1)
        if new_saved != old_saved:
            if new_saved:
                num_snippets += 1
                current_snippet = {}
            else:
                if old_code != 0 or random.random() < movement0_prop:
                    for key in current_snippet:
                        mapping[key] = current_snippet[key]

        if new_saved:
            current_snippet_frame = len(current_snippet)
            current_snippet[i] = (current_snippet_frame, num_snippets, new_code)
        old_saved = new_saved
        old_code = new_code

    if old_saved:
        if old_code != 0 or random.random() < movement0_prop:
            for key in current_snippet:
                mapping[key] = current_snippet[key]

    return mapping


def _find_annotations_dir(video_path: Path) -> Optional[Path]:
    for parent in video_path.parents:
        ann_dir = parent / "Annotations"
        if ann_dir.exists():
            return ann_dir
    return None


def _load_frame_annotations(
    video_path: Path,
    annotator_prefix: str,
    total_annotators: int
) -> Tuple[List[List[Tuple[bool, int]]], int]:
    ann_dir = _find_annotations_dir(video_path)
    if not ann_dir:
        return [], 0
    annotations = []
    for a in range(1, total_annotators + 1):
        annotator_dir = ann_dir / f"{annotator_prefix}{a}"
        json_path = annotator_dir / f"{video_path.stem}.json"
        if not json_path.exists():
            continue
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
            a_annotations = [
                (data["labels"][i]["is_washing"], data["labels"][i]["code"])
                for i in range(len(data["labels"]))
            ]
            annotations.append(a_annotations)
        except Exception as exc:
            logger.warning(f"Failed to load annotations from {json_path}: {exc}")
    return annotations, len(annotations)


def _frame_labels_from_annotations(
    annotations: List[List[Tuple[bool, int]]],
    total_movements: int,
    reaction_frames: int
) -> Tuple[List[int], List[int]]:
    num_annotators = len(annotations)
    if num_annotators == 0:
        return [], []
    num_frames = len(annotations[0])
    is_washing = []
    codes = []
    for frame_num in range(num_frames):
        frame_annotations = [annotations[a][frame_num] for a in range(num_annotators)]
        frame_is_washing_any = any(frame_annotations[a][0] for a in range(num_annotators))
        frame_is_washing_all = all(frame_annotations[a][0] for a in range(num_annotators))
        frame_codes = [frame_annotations[a][1] for a in range(num_annotators)]
        frame_codes = [0 if code == 7 else code for code in frame_codes]

        if frame_is_washing_all:
            frame_is_washing = 2
        elif frame_is_washing_any:
            frame_is_washing = 1
        else:
            frame_is_washing = 0

        is_washing.append(frame_is_washing)
        if frame_is_washing:
            codes.append(_majority_vote(frame_codes, total_movements))
        else:
            codes.append(-1)

    is_washing = _discount_reaction_indeterminacy(is_washing, reaction_frames)
    codes = _discount_reaction_indeterminacy(codes, reaction_frames)
    return is_washing, codes


def _load_pskus_split(pskus_dir: Path) -> Tuple[set, set]:
    csv_path = pskus_dir / "statistics-with-locations.csv"
    if not csv_path.exists():
        repo_root = Path(__file__).resolve().parents[1]
        fallback = repo_root / "code/edgewash/dataset-pskus/statistics-with-locations.csv"
        if fallback.exists():
            csv_path = fallback
            logger.info(f"Using fallback PSKUS split file: {csv_path}")
    testfiles = set()
    trainvalfiles = set()
    try:
        with open(csv_path, "r") as csv_file:
            for row in csv.reader(csv_file):
                if row and row[0] == "filename":
                    continue
                if not row:
                    continue
                filename = row[0]
                location = row[1] if len(row) > 1 else ""
                if location == "Reanimācija":
                    testfiles.add(filename)
                elif location != "unknown":
                    trainvalfiles.add(filename)
    except Exception as exc:
        logger.warning(f"Failed to read PSKUS split CSV {csv_path}: {exc}")
    return testfiles, trainvalfiles


def preprocess_pskus_dataset(pskus_dir: Path, output_dir: Path) -> pd.DataFrame:
    records = []
    random.seed(RANDOM_SEED)

    testfiles, trainvalfiles = _load_pskus_split(pskus_dir)
    movement0_prop = 0.2
    total_annotators = 8
    total_movements = 8
    fps = 30
    reaction_frames = fps // 2

    for video_path in pskus_dir.rglob("*.mp4"):
        filename = video_path.name
        if filename in testfiles:
            split = "test"
        elif filename in trainvalfiles:
            split = "trainval"
        else:
            continue

        annotations, num_annotators = _load_frame_annotations(
            video_path,
            annotator_prefix="Annotator",
            total_annotators=total_annotators
        )
        if num_annotators <= 1:
            continue

        is_washing, codes = _frame_labels_from_annotations(
            annotations,
            total_movements=total_movements,
            reaction_frames=reaction_frames
        )
        mapping = _select_frames_to_save(is_washing, codes, movement0_prop=movement0_prop)
        if not mapping:
            continue

        frames_dir = output_dir / "pskus" / split
        vidcap = cv2.VideoCapture(str(video_path))
        is_success, image = vidcap.read()
        frame_number = 0
        while is_success:
            if frame_number in mapping:
                new_frame_num, snippet_num, code = mapping[frame_number]
                subfolder = str(code)
                out_dir = frames_dir / subfolder
                out_dir.mkdir(parents=True, exist_ok=True)
                filename_out = f"frame_{new_frame_num}_snippet_{snippet_num}_{video_path.stem}.jpg"
                save_path = out_dir / filename_out
                cv2.imwrite(str(save_path), image)
                records.append({
                    "frame_path": str(save_path),
                    "class_id": int(code),
                    "class_name": CLASS_NAMES[int(code)],
                    "video_id": video_path.stem,
                    "frame_idx": new_frame_num,
                    "dataset": "pskus",
                    "split": split
                })
            is_success, image = vidcap.read()
            frame_number += 1
        vidcap.release()

    return pd.DataFrame(records)


def preprocess_metc_dataset(metc_dir: Path, output_dir: Path) -> pd.DataFrame:
    records = []
    random.seed(RANDOM_SEED)

    total_annotators = 1
    total_movements = 7
    fps = 16
    reaction_frames = fps // 2
    test_proportion = 0.25

    for video_path in metc_dir.rglob("*.mp4"):
        split = "test" if random.random() < test_proportion else "trainval"

        annotations, num_annotators = _load_frame_annotations(
            video_path,
            annotator_prefix="Annotator_",
            total_annotators=total_annotators
        )
        if num_annotators == 0:
            continue

        is_washing, codes = _frame_labels_from_annotations(
            annotations,
            total_movements=total_movements,
            reaction_frames=reaction_frames
        )
        mapping = _select_frames_to_save(is_washing, codes, movement0_prop=1.0)
        if not mapping:
            continue

        frames_dir = output_dir / "metc" / split
        vidcap = cv2.VideoCapture(str(video_path))
        is_success, image = vidcap.read()
        frame_number = 0
        while is_success:
            if frame_number in mapping:
                new_frame_num, snippet_num, code = mapping[frame_number]
                subfolder = str(code)
                out_dir = frames_dir / subfolder
                out_dir.mkdir(parents=True, exist_ok=True)
                filename_out = f"frame_{new_frame_num}_snippet_{snippet_num}_{video_path.stem}.jpg"
                save_path = out_dir / filename_out
                cv2.imwrite(str(save_path), image)
                records.append({
                    "frame_path": str(save_path),
                    "class_id": int(code),
                    "class_name": CLASS_NAMES[int(code)],
                    "video_id": video_path.stem,
                    "frame_idx": new_frame_num,
                    "dataset": "metc",
                    "split": split
                })
            is_success, image = vidcap.read()
            frame_number += 1
        vidcap.release()

    return pd.DataFrame(records)


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


def _split_train_val_by_video(
    frames_df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    random_state: int = RANDOM_SEED
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    unique_videos = frames_df["video_id"].unique()
    video_to_class = frames_df.groupby("video_id")["class_id"].first()
    val_size = val_ratio / (train_ratio + val_ratio)
    train_videos, val_videos = train_test_split(
        unique_videos,
        test_size=val_size,
        random_state=random_state,
        stratify=video_to_class[unique_videos]
    )
    train_df = frames_df[frames_df["video_id"].isin(train_videos)].reset_index(drop=True)
    val_df = frames_df[frames_df["video_id"].isin(val_videos)].reset_index(drop=True)
    return train_df, val_df


def split_dataset(
    frames_df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    random_state: int = RANDOM_SEED
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train/val/test by video (to avoid data leakage).
    If frames_df includes a 'split' column with 'trainval'/'test',
    honor that split and only split trainval into train/val.
    """
    if "split" in frames_df.columns and frames_df["split"].notna().any():
        test_df = frames_df[frames_df["split"] == "test"].reset_index(drop=True)
        trainval_df = frames_df[frames_df["split"] != "test"].reset_index(drop=True)
        if trainval_df.empty:
            return frames_df, frames_df.iloc[0:0].copy(), test_df
        train_df, val_df = _split_train_val_by_video(
            trainval_df,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            random_state=random_state
        )
        return train_df, val_df, test_df

    unique_videos = frames_df["video_id"].unique()
    video_to_class = frames_df.groupby("video_id")["class_id"].first()
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
    train_df = frames_df[frames_df["video_id"].isin(train_videos)].reset_index(drop=True)
    val_df = frames_df[frames_df["video_id"].isin(val_videos)].reset_index(drop=True)
    test_df = frames_df[frames_df["video_id"].isin(test_videos)].reset_index(drop=True)
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
        pskus_dir = RAW_DIR / "pskus"
        if any(pskus_dir.glob("DataSet*")):
            logger.info("\n[2/3] Processing PSKUS dataset...")
            pskus_frames = preprocess_pskus_dataset(pskus_dir, PROCESSED_DIR)
            logger.info(f"Extracted {len(pskus_frames)} frames")
            all_frames.append(pskus_frames)
        else:
            logger.warning("PSKUS dataset not found, skipping")
    
    # Process METC dataset
    if use_metc:
        metc_dir = RAW_DIR / "metc"
        if any(metc_dir.glob("Interface_number_*")):
            logger.info("\n[3/3] Processing METC dataset...")
            metc_frames = preprocess_metc_dataset(metc_dir, PROCESSED_DIR)
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
