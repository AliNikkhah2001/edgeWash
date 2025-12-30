"""
Training module for handwashing detection models.

Handles training loop, callbacks, checkpoints, and TensorBoard logging.
"""

import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

# Import project modules
from config import (
    PROCESSED_DIR,
    MODELS_DIR,
    CHECKPOINTS_DIR,
    LOGS_DIR,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    SEQUENCE_LENGTH,
    CHECKPOINT_CONFIG,
    EARLY_STOPPING_CONFIG,
    REDUCE_LR_CONFIG,
    RANDOM_SEED,
    LOG_FORMAT,
    LOG_DATE_FORMAT
)
from data_generators import create_frame_generators, create_sequence_generators
from models import get_model

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


def get_callbacks(
    model_name: str,
    checkpoint_dir: Path,
    log_dir: Path
) -> list:
    """
    Create training callbacks.
    
    Args:
        model_name: Name of model for checkpoint filenames
        checkpoint_dir: Directory to save checkpoints
        log_dir: Directory for TensorBoard logs
    
    Returns:
        List of Keras callbacks
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create directories
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Checkpoint path
    checkpoint_path = checkpoint_dir / f"{model_name}_{timestamp}"
    checkpoint_path.mkdir(exist_ok=True)
    checkpoint_file = checkpoint_path / "best_model.keras"
    
    # TensorBoard log directory
    tb_log_dir = log_dir / f"{model_name}_{timestamp}"
    
    callbacks_list = [
        # Model checkpoint
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_file),
            save_best_only=CHECKPOINT_CONFIG['save_best_only'],
            save_weights_only=CHECKPOINT_CONFIG['save_weights_only'],
            monitor=CHECKPOINT_CONFIG['monitor'],
            mode=CHECKPOINT_CONFIG['mode'],
            verbose=CHECKPOINT_CONFIG['verbose']
        ),
        
        # Early stopping
        keras.callbacks.EarlyStopping(
            monitor=EARLY_STOPPING_CONFIG['monitor'],
            patience=EARLY_STOPPING_CONFIG['patience'],
            restore_best_weights=EARLY_STOPPING_CONFIG['restore_best_weights'],
            verbose=EARLY_STOPPING_CONFIG['verbose']
        ),
        
        # Reduce learning rate on plateau
        keras.callbacks.ReduceLROnPlateau(
            monitor=REDUCE_LR_CONFIG['monitor'],
            factor=REDUCE_LR_CONFIG['factor'],
            patience=REDUCE_LR_CONFIG['patience'],
            min_lr=REDUCE_LR_CONFIG['min_lr'],
            verbose=REDUCE_LR_CONFIG['verbose']
        ),
        
        # TensorBoard
        keras.callbacks.TensorBoard(
            log_dir=str(tb_log_dir),
            histogram_freq=1,
            write_graph=True,
            write_images=False,
            update_freq='epoch'
        ),
        
        # CSV logger
        keras.callbacks.CSVLogger(
            filename=str(checkpoint_path / 'training_log.csv'),
            separator=',',
            append=False
        )
    ]
    
    logger.info(f"Callbacks configured:")
    logger.info(f"  Checkpoint: {checkpoint_file}")
    logger.info(f"  TensorBoard: {tb_log_dir}")
    logger.info(f"  CSV log: {checkpoint_path / 'training_log.csv'}")
    
    return callbacks_list


def train_model(
    model_type: str,
    train_csv: Path,
    val_csv: Path,
    batch_size: int = BATCH_SIZE,
    epochs: int = EPOCHS,
    learning_rate: float = LEARNING_RATE,
    resume_from: Optional[str] = None
) -> Dict[str, Any]:
    """
    Train a model.
    
    Args:
        model_type: Model type ('mobilenetv2', 'lstm', 'gru')
        train_csv: Path to training CSV
        val_csv: Path to validation CSV
        batch_size: Batch size
        epochs: Number of epochs
        learning_rate: Learning rate
        resume_from: Optional path to checkpoint to resume from
    
    Returns:
        Dictionary with training results
    """
    logger.info("=" * 80)
    logger.info(f"TRAINING: {model_type.upper()}")
    logger.info("=" * 80)
    
    # Load data
    logger.info("\nLoading data...")
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    
    logger.info(f"Train: {len(train_df)} samples")
    logger.info(f"Val:   {len(val_df)} samples")
    
    # Create data generators
    logger.info("\nCreating data generators...")
    if model_type == 'mobilenetv2':
        train_gen, val_gen, _ = create_frame_generators(
            train_df, val_df, val_df,  # Use val_df for test_gen (placeholder)
            batch_size=batch_size
        )
    else:  # LSTM or GRU
        train_gen, val_gen, _ = create_sequence_generators(
            train_df, val_df, val_df,
            sequence_length=SEQUENCE_LENGTH,
            batch_size=batch_size
        )
    
    # Create or load model
    if resume_from:
        logger.info(f"\nLoading model from checkpoint: {resume_from}")
        model = keras.models.load_model(resume_from)
    else:
        logger.info(f"\nCreating {model_type} model...")
        model = get_model(model_type, learning_rate=learning_rate)
    
    # Create callbacks
    callbacks_list = get_callbacks(
        model_name=model_type,
        checkpoint_dir=CHECKPOINTS_DIR,
        log_dir=LOGS_DIR
    )
    
    # Train
    logger.info("\n" + "=" * 80)
    logger.info("STARTING TRAINING")
    logger.info("=" * 80)
    logger.info(f"Epochs: {epochs}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"Train batches: {len(train_gen)}")
    logger.info(f"Val batches: {len(val_gen)}")
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks_list,
        verbose=1
    )
    
    # Save final model
    final_model_path = MODELS_DIR / f"{model_type}_final.keras"
    model.save(final_model_path)
    logger.info(f"\nFinal model saved: {final_model_path}")
    
    # Training summary
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 80)
    
    best_epoch = np.argmax(history.history['val_accuracy'])
    logger.info(f"Best epoch: {best_epoch + 1}")
    logger.info(f"Best val_accuracy: {history.history['val_accuracy'][best_epoch]:.4f}")
    logger.info(f"Best val_loss: {history.history['val_loss'][best_epoch]:.4f}")
    
    return {
        'model': model,
        'history': history.history,
        'best_epoch': best_epoch,
        'final_model_path': str(final_model_path)
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train handwashing detection model"
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=['mobilenetv2', 'lstm', 'gru'],
        help='Model architecture to train'
    )
    parser.add_argument(
        '--train-csv',
        type=str,
        default=str(PROCESSED_DIR / 'train.csv'),
        help='Path to training CSV'
    )
    parser.add_argument(
        '--val-csv',
        type=str,
        default=str(PROCESSED_DIR / 'val.csv'),
        help='Path to validation CSV'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=BATCH_SIZE,
        help='Batch size'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=EPOCHS,
        help='Number of epochs'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=LEARNING_RATE,
        help='Learning rate'
    )
    parser.add_argument(
        '--resume-from',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    
    args = parser.parse_args()
    
    # Train model
    result = train_model(
        model_type=args.model,
        train_csv=Path(args.train_csv),
        val_csv=Path(args.val_csv),
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        resume_from=args.resume_from
    )
    
    logger.info(f"\nTraining complete! Final model: {result['final_model_path']}")
