"""
Data generators module for handwashing detection training pipeline.

Provides efficient data loading and augmentation using TensorFlow/Keras.
"""

import logging
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from typing import Tuple, Optional
import tensorflow as tf
from tensorflow import keras

# Import configuration
from config import (
    IMG_SIZE,
    NUM_CLASSES,
    BATCH_SIZE,
    AUGMENTATION_CONFIG,
    RANDOM_SEED,
    LOG_FORMAT,
    LOG_DATE_FORMAT,
    AUGMENT_MULTIPLIER,
    ENABLE_SHADOW_AUG,
    CONSISTENT_VIDEO_AUG
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
tf.random.set_seed(RANDOM_SEED)


def _sample_aug_params() -> dict:
    hflip_enabled = any(
        AUGMENTATION_CONFIG.get(key, False)
        for key in ("horizontal_flip", "mid_flip", "hflip")
    )
    params = {
        "hflip": hflip_enabled and np.random.rand() > 0.5,
        "angle": 0.0,
        "zoom": 1.0,
        "shear": 0.0,
        "tx": 0,
        "ty": 0,
        "brightness": None,
        "contrast": None,
        "gamma": None,
        "shadow": False,
        "reverse_sequence": AUGMENTATION_CONFIG.get("reverse_sequence", False) and np.random.rand() > 0.5,
    }

    if AUGMENTATION_CONFIG.get("rotation_range", 0) > 0:
        params["angle"] = np.random.uniform(
            -AUGMENTATION_CONFIG["rotation_range"],
            AUGMENTATION_CONFIG["rotation_range"]
        )

    if AUGMENTATION_CONFIG.get("zoom_range", 0) > 0:
        params["zoom"] = np.random.uniform(
            1 - AUGMENTATION_CONFIG["zoom_range"],
            1 + AUGMENTATION_CONFIG["zoom_range"]
        )

    if AUGMENTATION_CONFIG.get("shear_range", 0) > 0:
        params["shear"] = np.random.uniform(
            -AUGMENTATION_CONFIG["shear_range"],
            AUGMENTATION_CONFIG["shear_range"]
        )

    if AUGMENTATION_CONFIG.get("width_shift_range", 0) > 0 or AUGMENTATION_CONFIG.get("height_shift_range", 0) > 0:
        params["tx"] = int(np.random.uniform(
            -AUGMENTATION_CONFIG["width_shift_range"],
            AUGMENTATION_CONFIG["width_shift_range"]
        ) * IMG_SIZE[0])
        params["ty"] = int(np.random.uniform(
            -AUGMENTATION_CONFIG["height_shift_range"],
            AUGMENTATION_CONFIG["height_shift_range"]
        ) * IMG_SIZE[1])

    if "brightness_range" in AUGMENTATION_CONFIG:
        params["brightness"] = np.random.uniform(*AUGMENTATION_CONFIG["brightness_range"])

    if "contrast_range" in AUGMENTATION_CONFIG:
        params["contrast"] = np.random.uniform(*AUGMENTATION_CONFIG["contrast_range"])

    if "gamma_range" in AUGMENTATION_CONFIG:
        params["gamma"] = np.random.uniform(*AUGMENTATION_CONFIG["gamma_range"])

    if ENABLE_SHADOW_AUG and np.random.rand() < 0.5:
        params["shadow"] = True

    return params


def _apply_aug(img: np.ndarray, params: dict) -> np.ndarray:
    if params.get("hflip"):
        img = cv2.flip(img, 1)

    angle = params.get("angle", 0.0)
    if angle:
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    zoom = params.get("zoom", 1.0)
    if zoom != 1.0:
        h, w = img.shape[:2]
        new_h, new_w = int(h * zoom), int(w * zoom)
        img_resized = cv2.resize(img, (new_w, new_h))
        if zoom > 1:
            start_y = (new_h - h) // 2
            start_x = (new_w - w) // 2
            img = img_resized[start_y:start_y + h, start_x:start_x + w]
        else:
            pad_h = (h - new_h) // 2
            pad_w = (w - new_w) // 2
            img = cv2.copyMakeBorder(
                img_resized,
                pad_h, h - new_h - pad_h,
                pad_w, w - new_w - pad_w,
                cv2.BORDER_REFLECT
            )

    tx, ty = params.get("tx", 0), params.get("ty", 0)
    if tx or ty:
        h, w = img.shape[:2]
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    shear = params.get("shear", 0.0)
    if shear:
        h, w = img.shape[:2]
        M = np.float32([[1, shear, 0], [0, 1, 0]])
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    brightness = params.get("brightness")
    if brightness is not None:
        img = np.clip(img.astype(np.float32) * brightness, 0, 255).astype(np.uint8)

    contrast = params.get("contrast")
    if contrast is not None:
        img = np.clip(128 + contrast * (img.astype(np.float32) - 128), 0, 255).astype(np.uint8)

    gamma = params.get("gamma")
    if gamma is not None:
        img = np.clip(((img.astype(np.float32) / 255.0) ** gamma) * 255.0, 0, 255).astype(np.uint8)

    if params.get("shadow"):
        h, w = img.shape[:2]
        x1, y1 = np.random.randint(0, w), 0
        x2, y2 = np.random.randint(0, w), h
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [np.array([[x1, y1], [x2, y2], [0, h], [w, h]])], 255)
        shadow_intensity = np.random.uniform(0.5, 0.9)
        shadow = np.stack([mask] * 3, axis=-1)
        img = np.where(shadow > 0, (img * shadow_intensity).astype(np.uint8), img)

    return img


class FrameDataGenerator(keras.utils.Sequence):
    """
    Custom data generator for frame-based models.

    Efficiently loads and augments frames for training.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        batch_size: int = BATCH_SIZE,
        img_size: Tuple[int, int] = IMG_SIZE,
        num_classes: int = NUM_CLASSES,
        shuffle: bool = True,
        augment: bool = False,
        augment_multiplier: int = 1,
        augment_prob: float = 1.0
    ):
        """
        Initialize frame data generator.

        Args:
            df: DataFrame with frame_path and class_id columns
            batch_size: Batch size
            img_size: Target image size (width, height)
            num_classes: Number of classes
            shuffle: Shuffle data at end of each epoch
            augment: Apply data augmentation
        """
        self.df = df.reset_index(drop=True)
        self.batch_size = batch_size
        self.img_size = img_size
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.augment = augment
        self.augment_multiplier = max(1, augment_multiplier)
        self.augment_prob = max(0.0, min(1.0, augment_prob))
        self.consistent_video_aug = CONSISTENT_VIDEO_AUG and "video_id" in self.df.columns
        self.video_aug_params = {}
        self.indices = np.arange(len(self.df))
        self.on_epoch_end()

    def __len__(self):
        """Return number of batches per epoch."""
        return int(np.floor((len(self.df) * self.augment_multiplier) / self.batch_size))

    def __getitem__(self, index):
        """
        Generate one batch of data.

        Args:
            index: Batch index

        Returns:
            Tuple of (X, y) where X is images and y is labels
        """
        # Sample with replacement to simulate virtual epoch expansion
        batch_indices = np.random.choice(self.indices, size=self.batch_size, replace=True)

        X, y = self._generate_batch(batch_indices)
        return X, y

    def on_epoch_end(self):
        """Shuffle indices after each epoch."""
        if self.shuffle:
            np.random.shuffle(self.indices)
        if self.augment and self.consistent_video_aug:
            self.video_aug_params = {
                vid: _sample_aug_params()
                for vid in self.df["video_id"].dropna().unique().tolist()
            }

    def _get_aug_params(self, video_id: Optional[str] = None) -> dict:
        if self.consistent_video_aug and video_id in self.video_aug_params:
            return self.video_aug_params[video_id]
        return _sample_aug_params()

    def _generate_batch(self, indices):
        """
        Generate batch of images and labels.

        Args:
            indices: Indices of samples to load

        Returns:
            Tuple of (X, y)
        """
        X = np.empty((self.batch_size, *self.img_size, 3), dtype=np.float32)
        y = np.empty((self.batch_size, self.num_classes), dtype=np.float32)

        for i, idx in enumerate(indices):
            row = self.df.iloc[idx]

            # Load image
            img = cv2.imread(row['frame_path'])
            if img is None:
                logger.warning(f"Failed to load: {row['frame_path']}")
                img = np.zeros((*self.img_size[::-1], 3), dtype=np.uint8)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Apply augmentation if enabled
            if self.augment and (self.augment_prob >= 1.0 or np.random.rand() < self.augment_prob):
                video_id = row.get("video_id") if self.consistent_video_aug else None
                params = self._get_aug_params(video_id)
                img = _apply_aug(img, params)

            # Normalize to [0, 1]
            img = img.astype(np.float32) / 255.0

            # Store
            X[i] = img
            y[i] = keras.utils.to_categorical(row['class_id'], self.num_classes)

        return X, y
    


class SequenceDataGenerator(keras.utils.Sequence):
    """
    Custom data generator for sequence-based models (LSTM/GRU).
    
    Generates sequences of frames for temporal modeling.
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        sequence_length: int,
        batch_size: int = BATCH_SIZE,
        img_size: Tuple[int, int] = IMG_SIZE,
        num_classes: int = NUM_CLASSES,
        shuffle: bool = True,
        augment: bool = False,
        augment_multiplier: int = 1,
        augment_prob: float = 1.0
    ):
        """
        Initialize sequence data generator.
        
        Args:
            df: DataFrame with frame_path, class_id, video_id columns
            sequence_length: Number of frames per sequence
            batch_size: Batch size
            img_size: Target image size (width, height)
            num_classes: Number of classes
            shuffle: Shuffle data at end of each epoch
            augment: Apply data augmentation
        """
        self.df = df.reset_index(drop=True)
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.img_size = img_size
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.augment = augment
        self.augment_multiplier = max(1, augment_multiplier)
        self.augment_prob = max(0.0, min(1.0, augment_prob))
        self.consistent_video_aug = CONSISTENT_VIDEO_AUG and "video_id" in self.df.columns
        self.video_aug_params = {}
        
        # Group frames by video
        self.video_groups = self.df.groupby('video_id')
        
        # Create sequences
        self.sequences = self._create_sequences()
        self.indices = np.arange(len(self.sequences))
        self.on_epoch_end()
    
    def _create_sequences(self):
        """
        Create sequences from frames.
        
        Returns:
            List of (frame_indices, class_id) tuples
        """
        sequences = []
        
        for video_id, group in self.video_groups:
            frames = group.sort_values('frame_idx')
            num_frames = len(frames)
            
            # Create overlapping sequences
            for start_idx in range(0, num_frames - self.sequence_length + 1, self.sequence_length // 2):
                end_idx = start_idx + self.sequence_length
                if end_idx <= num_frames:
                    frame_indices = frames.iloc[start_idx:end_idx].index.tolist()
                    # Use most common class in sequence as label
                    class_id = frames.iloc[start_idx:end_idx]['class_id'].mode()[0]
                    sequences.append((frame_indices, class_id, video_id))
        
        return sequences
    
    def __len__(self):
        """Return number of batches per epoch."""
        return int(np.floor((len(self.sequences) * self.augment_multiplier) / self.batch_size))
    
    def __getitem__(self, index):
        """
        Generate one batch of sequences.
        
        Args:
            index: Batch index
        
        Returns:
            Tuple of (X, y) where X is sequences and y is labels
        """
        batch_indices = np.random.choice(self.indices, size=self.batch_size, replace=True)
        
        X, y = self._generate_batch(batch_indices)
        return X, y
    
    def on_epoch_end(self):
        """Shuffle indices after each epoch."""
        if self.shuffle:
            np.random.shuffle(self.indices)
        if self.augment and self.consistent_video_aug:
            self.video_aug_params = {
                vid: _sample_aug_params()
                for vid in self.df["video_id"].dropna().unique().tolist()
            }

    def _get_aug_params(self, video_id: Optional[str] = None) -> dict:
        if self.consistent_video_aug and video_id in self.video_aug_params:
            return self.video_aug_params[video_id]
        return _sample_aug_params()
    
    def _generate_batch(self, indices):
        """
        Generate batch of sequences and labels.
        
        Args:
            indices: Indices of sequences to load
        
        Returns:
            Tuple of (X, y)
        """
        X = np.empty((self.batch_size, self.sequence_length, *self.img_size, 3), dtype=np.float32)
        y = np.empty((self.batch_size, self.num_classes), dtype=np.float32)
        
        for i, idx in enumerate(indices):
            frame_indices, class_id, video_id = self.sequences[idx]
            apply_aug = self.augment and (self.augment_prob >= 1.0 or np.random.rand() < self.augment_prob)
            params = self._get_aug_params(video_id) if apply_aug else None
            if apply_aug and params.get("reverse_sequence"):
                frame_indices = list(reversed(frame_indices))
            
            # Load sequence of frames
            for j, frame_idx in enumerate(frame_indices):
                row = self.df.iloc[frame_idx]
                
                # Load image
                img = cv2.imread(row['frame_path'])
                if img is None:
                    img = np.zeros((*self.img_size[::-1], 3), dtype=np.uint8)
                else:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Apply augmentation if enabled
                if apply_aug:
                    img = _apply_aug(img, params)
                
                # Normalize
                img = img.astype(np.float32) / 255.0
                
                X[i, j] = img
            
            y[i] = keras.utils.to_categorical(class_id, self.num_classes)
        
        return X, y
    


def create_frame_generators(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    batch_size: int = BATCH_SIZE,
    augment_multiplier: int = 1
) -> Tuple[FrameDataGenerator, FrameDataGenerator, FrameDataGenerator]:
    """
    Create frame-based data generators for train/val/test.
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        batch_size: Batch size
    
    Returns:
        Tuple of (train_gen, val_gen, test_gen)
    """
    train_gen = FrameDataGenerator(
        train_df,
        batch_size=batch_size,
        shuffle=True,
        augment=True,
        augment_multiplier=augment_multiplier
    )
    
    val_gen = FrameDataGenerator(
        val_df,
        batch_size=batch_size,
        shuffle=False,
        augment=False,
        augment_multiplier=1
    )
    
    test_gen = FrameDataGenerator(
        test_df,
        batch_size=batch_size,
        shuffle=False,
        augment=False,
        augment_multiplier=1
    )
    
    logger.info(f"Created frame generators:")
    logger.info(f"  Train: {len(train_gen)} batches ({len(train_df)} samples)")
    logger.info(f"  Val:   {len(val_gen)} batches ({len(val_df)} samples)")
    logger.info(f"  Test:  {len(test_gen)} batches ({len(test_df)} samples)")
    
    return train_gen, val_gen, test_gen


def create_sequence_generators(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    sequence_length: int,
    batch_size: int = BATCH_SIZE,
    augment_multiplier: int = 1
) -> Tuple[SequenceDataGenerator, SequenceDataGenerator, SequenceDataGenerator]:
    """
    Create sequence-based data generators for train/val/test.
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        sequence_length: Number of frames per sequence
        batch_size: Batch size
    
    Returns:
        Tuple of (train_gen, val_gen, test_gen)
    """
    train_gen = SequenceDataGenerator(
        train_df,
        sequence_length=sequence_length,
        batch_size=batch_size,
        shuffle=True,
        augment=True,
        augment_multiplier=augment_multiplier
    )
    
    val_gen = SequenceDataGenerator(
        val_df,
        sequence_length=sequence_length,
        batch_size=batch_size,
        shuffle=False,
        augment=False,
        augment_multiplier=1
    )
    
    test_gen = SequenceDataGenerator(
        test_df,
        sequence_length=sequence_length,
        batch_size=batch_size,
        shuffle=False,
        augment=False,
        augment_multiplier=1
    )
    
    logger.info(f"Created sequence generators:")
    logger.info(f"  Train: {len(train_gen)} batches ({len(train_gen.sequences)} sequences)")
    logger.info(f"  Val:   {len(val_gen)} batches ({len(val_gen.sequences)} sequences)")
    logger.info(f"  Test:  {len(test_gen)} batches ({len(test_gen.sequences)} sequences)")
    
    return train_gen, val_gen, test_gen
