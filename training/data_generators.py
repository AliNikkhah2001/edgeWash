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
tf.random.set_seed(RANDOM_SEED)


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
        augment: bool = False
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
        self.indices = np.arange(len(self.df))
        self.on_epoch_end()
    
    def __len__(self):
        """Return number of batches per epoch."""
        return int(np.floor(len(self.df) / self.batch_size))
    
    def __getitem__(self, index):
        """
        Generate one batch of data.
        
        Args:
            index: Batch index
        
        Returns:
            Tuple of (X, y) where X is images and y is labels
        """
        # Generate batch indices
        start_idx = index * self.batch_size
        end_idx = start_idx + self.batch_size
        batch_indices = self.indices[start_idx:end_idx]
        
        # Generate batch data
        X, y = self._generate_batch(batch_indices)
        return X, y
    
    def on_epoch_end(self):
        """Shuffle indices after each epoch."""
        if self.shuffle:
            np.random.shuffle(self.indices)
    
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
                # Use black image as fallback
                img = np.zeros((*self.img_size[::-1], 3), dtype=np.uint8)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Apply augmentation if enabled
            if self.augment:
                img = self._augment_image(img)
            
            # Normalize to [0, 1]
            img = img.astype(np.float32) / 255.0
            
            # Store
            X[i] = img
            y[i] = keras.utils.to_categorical(row['class_id'], self.num_classes)
        
        return X, y
    
    def _augment_image(self, img: np.ndarray) -> np.ndarray:
        """
        Apply random augmentations to image.
        
        Args:
            img: Input image (H, W, 3)
        
        Returns:
            Augmented image
        """
        # Horizontal flip
        if AUGMENTATION_CONFIG['horizontal_flip'] and np.random.rand() > 0.5:
            img = cv2.flip(img, 1)
        
        # Rotation
        if AUGMENTATION_CONFIG['rotation_range'] > 0:
            angle = np.random.uniform(
                -AUGMENTATION_CONFIG['rotation_range'],
                AUGMENTATION_CONFIG['rotation_range']
            )
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        
        # Zoom
        if AUGMENTATION_CONFIG['zoom_range'] > 0:
            zoom = np.random.uniform(
                1 - AUGMENTATION_CONFIG['zoom_range'],
                1 + AUGMENTATION_CONFIG['zoom_range']
            )
            h, w = img.shape[:2]
            new_h, new_w = int(h * zoom), int(w * zoom)
            img_resized = cv2.resize(img, (new_w, new_h))
            
            # Center crop or pad
            if zoom > 1:
                # Crop
                start_y = (new_h - h) // 2
                start_x = (new_w - w) // 2
                img = img_resized[start_y:start_y+h, start_x:start_x+w]
            else:
                # Pad
                pad_h = (h - new_h) // 2
                pad_w = (w - new_w) // 2
                img = cv2.copyMakeBorder(
                    img_resized,
                    pad_h, h - new_h - pad_h,
                    pad_w, w - new_w - pad_w,
                    cv2.BORDER_REFLECT
                )
        
        # Shift
        if AUGMENTATION_CONFIG['width_shift_range'] > 0 or AUGMENTATION_CONFIG['height_shift_range'] > 0:
            h, w = img.shape[:2]
            tx = int(np.random.uniform(-AUGMENTATION_CONFIG['width_shift_range'], 
                                       AUGMENTATION_CONFIG['width_shift_range']) * w)
            ty = int(np.random.uniform(-AUGMENTATION_CONFIG['height_shift_range'], 
                                       AUGMENTATION_CONFIG['height_shift_range']) * h)
            M = np.float32([[1, 0, tx], [0, 1, ty]])
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        
        # Brightness
        if 'brightness_range' in AUGMENTATION_CONFIG:
            brightness = np.random.uniform(*AUGMENTATION_CONFIG['brightness_range'])
            img = np.clip(img.astype(np.float32) * brightness, 0, 255).astype(np.uint8)
        
        return img


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
        augment: bool = False
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
                    sequences.append((frame_indices, class_id))
        
        return sequences
    
    def __len__(self):
        """Return number of batches per epoch."""
        return int(np.floor(len(self.sequences) / self.batch_size))
    
    def __getitem__(self, index):
        """
        Generate one batch of sequences.
        
        Args:
            index: Batch index
        
        Returns:
            Tuple of (X, y) where X is sequences and y is labels
        """
        start_idx = index * self.batch_size
        end_idx = start_idx + self.batch_size
        batch_indices = self.indices[start_idx:end_idx]
        
        X, y = self._generate_batch(batch_indices)
        return X, y
    
    def on_epoch_end(self):
        """Shuffle indices after each epoch."""
        if self.shuffle:
            np.random.shuffle(self.indices)
    
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
            frame_indices, class_id = self.sequences[idx]
            
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
                if self.augment:
                    img = self._augment_image(img)
                
                # Normalize
                img = img.astype(np.float32) / 255.0
                
                X[i, j] = img
            
            y[i] = keras.utils.to_categorical(class_id, self.num_classes)
        
        return X, y
    
    def _augment_image(self, img: np.ndarray) -> np.ndarray:
        """Apply augmentations (same as FrameDataGenerator)."""
        # Same augmentation logic as FrameDataGenerator
        if AUGMENTATION_CONFIG['horizontal_flip'] and np.random.rand() > 0.5:
            img = cv2.flip(img, 1)
        
        if AUGMENTATION_CONFIG['rotation_range'] > 0:
            angle = np.random.uniform(
                -AUGMENTATION_CONFIG['rotation_range'],
                AUGMENTATION_CONFIG['rotation_range']
            )
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        
        return img


def create_frame_generators(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    batch_size: int = BATCH_SIZE
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
        augment=True
    )
    
    val_gen = FrameDataGenerator(
        val_df,
        batch_size=batch_size,
        shuffle=False,
        augment=False
    )
    
    test_gen = FrameDataGenerator(
        test_df,
        batch_size=batch_size,
        shuffle=False,
        augment=False
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
    batch_size: int = BATCH_SIZE
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
        augment=True
    )
    
    val_gen = SequenceDataGenerator(
        val_df,
        sequence_length=sequence_length,
        batch_size=batch_size,
        shuffle=False,
        augment=False
    )
    
    test_gen = SequenceDataGenerator(
        test_df,
        sequence_length=sequence_length,
        batch_size=batch_size,
        shuffle=False,
        augment=False
    )
    
    logger.info(f"Created sequence generators:")
    logger.info(f"  Train: {len(train_gen)} batches ({len(train_gen.sequences)} sequences)")
    logger.info(f"  Val:   {len(val_gen)} batches ({len(val_gen.sequences)} sequences)")
    logger.info(f"  Test:  {len(test_gen)} batches ({len(test_gen.sequences)} sequences)")
    
    return train_gen, val_gen, test_gen
