"""
Training module for handwashing detection models.

Handles training loop, callbacks, checkpoints, and TensorBoard logging.
"""

import os
import sys
import logging
import shutil
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, top_k_accuracy_score

# Import project modules
from config import (
    PROCESSED_DIR,
    MODELS_DIR,
    CHECKPOINTS_DIR,
    LOGS_DIR,
    RESULTS_DIR,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    SEQUENCE_LENGTH,
    CHECKPOINT_CONFIG,
    EARLY_STOPPING_CONFIG,
    REDUCE_LR_CONFIG,
    OPTIMIZER_NAME,
    WEIGHT_DECAY,
    LABEL_SMOOTHING,
    RESNET50_SCHEDULE,
    RESNET50_STAGE0_EPOCHS,
    RESNET50_STAGE1_EPOCHS,
    RESNET50_STAGE2_EPOCHS,
    RESNET50_STAGE0_LR,
    RESNET50_STAGE1_LR,
    RESNET50_STAGE2_LR,
    RESNET50_STAGE0_WD,
    RESNET50_STAGE1_WD,
    RESNET50_STAGE2_WD,
    RANDOM_SEED,
    LOG_FORMAT,
    LOG_DATE_FORMAT,
    AUGMENT_MULTIPLIER,
    NUM_CLASSES,
    CLASS_NAMES
)
from data_generators import (
    create_frame_generators,
    create_sequence_generators,
    FrameDataGenerator,
    SequenceDataGenerator
)
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
    log_dir: Path,
    enable_reduce_lr: bool = True
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
        
    ]

    if enable_reduce_lr:
        callbacks_list.append(
            keras.callbacks.ReduceLROnPlateau(
                monitor=REDUCE_LR_CONFIG['monitor'],
                factor=REDUCE_LR_CONFIG['factor'],
                patience=REDUCE_LR_CONFIG['patience'],
                min_lr=REDUCE_LR_CONFIG['min_lr'],
                verbose=REDUCE_LR_CONFIG['verbose']
            )
        )

    callbacks_list.extend([
        
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
    ])
    
    logger.info(f"Callbacks configured:")
    logger.info(f"  Checkpoint: {checkpoint_file}")
    logger.info(f"  TensorBoard: {tb_log_dir}")
    logger.info(f"  CSV log: {checkpoint_path / 'training_log.csv'}")
    
    return callbacks_list


def _env_float(key: str, default: float) -> float:
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_bool(key: str, default: bool) -> bool:
    value = os.getenv(key)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "y")


def _build_optimizer(learning_rate: float, weight_decay: float) -> keras.optimizers.Optimizer:
    name = os.getenv("OPTIMIZER_NAME", OPTIMIZER_NAME).lower()
    weight_decay = _env_float("WEIGHT_DECAY", weight_decay)
    if name == "adamw":
        return keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    if name == "adam":
        return keras.optimizers.Adam(learning_rate=learning_rate)
    raise ValueError(f"Unknown optimizer: {OPTIMIZER_NAME}")


def _compile_model(model: keras.Model, learning_rate: float, weight_decay: float) -> None:
    label_smoothing = _env_float("LABEL_SMOOTHING", LABEL_SMOOTHING)
    loss = keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing)
    model.compile(
        optimizer=_build_optimizer(learning_rate, weight_decay),
        loss=loss,
        metrics=[
            'accuracy',
            keras.metrics.TopKCategoricalAccuracy(k=2, name='top2_accuracy'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall')
        ]
    )


def _find_backbone(model: keras.Model, name_prefix: str) -> Optional[keras.Model]:
    for layer in model.layers:
        if isinstance(layer, keras.Model) and layer.name == name_prefix:
            return layer
    return None


def _set_resnet_trainable(base_model: keras.Model, train_conv4: bool, train_conv5: bool) -> None:
    for layer in base_model.layers:
        if layer.name.startswith("conv5_"):
            layer.trainable = train_conv5
        elif layer.name.startswith("conv4_"):
            layer.trainable = train_conv4
        else:
            layer.trainable = False


def _evaluate_generator(model: keras.Model, gen) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true, y_pred, y_pred_proba = [], [], []
    for i in range(len(gen)):
        batch_x, batch_y = gen[i]
        preds = model.predict_on_batch(batch_x)
        y_pred_proba.append(preds)
        y_true.extend(np.argmax(batch_y, axis=1))
        y_pred.extend(np.argmax(preds, axis=1))
    if y_pred_proba:
        y_pred_proba = np.concatenate(y_pred_proba, axis=0)
    else:
        y_pred_proba = np.zeros((0, NUM_CLASSES), dtype=np.float32)
    return np.array(y_true), np.array(y_pred), y_pred_proba


def _save_confusion_matrix(cm: np.ndarray, title: str, save_path: Path, normalize: bool) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if normalize and cm.sum(axis=1, keepdims=True).any():
        cm = cm.astype("float") / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(CLASS_NAMES))
    plt.xticks(tick_marks, CLASS_NAMES, rotation=45, ha="right")
    plt.yticks(tick_marks, CLASS_NAMES)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


class TestEvalCallback(keras.callbacks.Callback):
    def __init__(
        self,
        test_gen,
        test_mix_gen=None,
        results_dir: Path | None = None,
        normalize_cm: bool = False
    ) -> None:
        super().__init__()
        self.test_gen = test_gen
        self.test_mix_gen = test_mix_gen
        self.results_dir = results_dir
        self.normalize_cm = normalize_cm

    def _run_eval(self, gen, label: str, epoch: int) -> None:
        y_true, y_pred, y_pred_proba = _evaluate_generator(self.model, gen)
        if y_true.size == 0:
            return
        acc = accuracy_score(y_true, y_pred)
        top2 = top_k_accuracy_score(y_true, y_pred_proba, k=2, labels=list(range(NUM_CLASSES)))
        logger.info(f"[Epoch {epoch + 1}] {label} acc={acc:.4f} top2={top2:.4f}")
        cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
        if self.results_dir:
            _save_confusion_matrix(
                cm,
                f"{label} confusion (epoch {epoch + 1})",
                self.results_dir / f"{label}_confusion_epoch_{epoch + 1}.png",
                self.normalize_cm
            )

    def on_epoch_end(self, epoch, logs=None):
        self._run_eval(self.test_gen, "test", epoch)
        if self.test_mix_gen is not None:
            self._run_eval(self.test_mix_gen, "test_mix", epoch)


def _train_resnet50_schedule(
    train_gen,
    val_gen,
    checkpoint_dir: Path,
    log_dir: Path,
    test_gen=None,
    test_mix_gen=None
) -> tuple[keras.Model, Optional[keras.callbacks.History]]:
    logger.info("Using ResNet50 fine-tuning schedule (head -> conv4/5 -> full).")
    stage0_lr = _env_float("RESNET50_STAGE0_LR", RESNET50_STAGE0_LR)
    stage1_lr = _env_float("RESNET50_STAGE1_LR", RESNET50_STAGE1_LR)
    stage2_lr = _env_float("RESNET50_STAGE2_LR", RESNET50_STAGE2_LR)
    stage0_wd = _env_float("RESNET50_STAGE0_WD", RESNET50_STAGE0_WD)
    stage1_wd = _env_float("RESNET50_STAGE1_WD", RESNET50_STAGE1_WD)
    stage2_wd = _env_float("RESNET50_STAGE2_WD", RESNET50_STAGE2_WD)
    stage0_epochs = int(_env_float("RESNET50_STAGE0_EPOCHS", RESNET50_STAGE0_EPOCHS))
    stage1_epochs = int(_env_float("RESNET50_STAGE1_EPOCHS", RESNET50_STAGE1_EPOCHS))
    stage2_epochs = int(_env_float("RESNET50_STAGE2_EPOCHS", RESNET50_STAGE2_EPOCHS))

    model = get_model("resnet50", learning_rate=stage0_lr, freeze_backbone=True)

    backbone = _find_backbone(model, "resnet50_backbone")
    if backbone is None:
        raise RuntimeError("ResNet50 backbone not found; cannot apply schedule.")

    # Stage 0: head only
    _set_resnet_trainable(backbone, train_conv4=False, train_conv5=False)
    _compile_model(model, stage0_lr, stage0_wd)
    callbacks_stage0 = get_callbacks("resnet50_stage0", checkpoint_dir, log_dir, enable_reduce_lr=True)
    if test_gen is not None and _env_bool("EVAL_TEST_EACH_EPOCH", True):
        callbacks_stage0.append(
            TestEvalCallback(test_gen, test_mix_gen, results_dir=RESULTS_DIR / "resnet50", normalize_cm=_env_bool("CONFUSION_NORMALIZE", False))
        )
    hist0 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=stage0_epochs,
        callbacks=callbacks_stage0,
        verbose=1
    )

    # Stage 1: unfreeze conv4_x + conv5_x, cosine decay
    _set_resnet_trainable(backbone, train_conv4=True, train_conv5=True)
    steps = max(1, len(train_gen) * stage1_epochs)
    schedule = keras.optimizers.schedules.CosineDecay(stage1_lr, decay_steps=steps)
    optimizer = keras.optimizers.AdamW(learning_rate=schedule, weight_decay=stage1_wd)
    loss = keras.losses.CategoricalCrossentropy(label_smoothing=_env_float("LABEL_SMOOTHING", LABEL_SMOOTHING))
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[
            'accuracy',
            keras.metrics.TopKCategoricalAccuracy(k=2, name='top2_accuracy'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall')
        ]
    )
    callbacks_stage1 = get_callbacks("resnet50_stage1", checkpoint_dir, log_dir, enable_reduce_lr=False)
    if test_gen is not None and _env_bool("EVAL_TEST_EACH_EPOCH", True):
        callbacks_stage1.append(
            TestEvalCallback(test_gen, test_mix_gen, results_dir=RESULTS_DIR / "resnet50", normalize_cm=_env_bool("CONFUSION_NORMALIZE", False))
        )
    hist1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=stage1_epochs,
        callbacks=callbacks_stage1,
        verbose=1
    )

    # Stage 2: full fine-tune
    for layer in backbone.layers:
        layer.trainable = True
    _compile_model(model, stage2_lr, stage2_wd)
    callbacks_stage2 = get_callbacks("resnet50_stage2", checkpoint_dir, log_dir, enable_reduce_lr=True)
    if test_gen is not None and _env_bool("EVAL_TEST_EACH_EPOCH", True):
        callbacks_stage2.append(
            TestEvalCallback(test_gen, test_mix_gen, results_dir=RESULTS_DIR / "resnet50", normalize_cm=_env_bool("CONFUSION_NORMALIZE", False))
        )
    hist2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=stage2_epochs,
        callbacks=callbacks_stage2,
        verbose=1
    )
    # Prefer the final stage history for summary
    return model, hist2


def cleanup_old_checkpoints(model_prefix: str, checkpoint_dir: Path, keep: int = 3) -> None:
    """
    Keep only the latest `keep` checkpoint folders for the given model prefix.
    """
    if not checkpoint_dir.exists():
        return
    candidates = sorted(
        [p for p in checkpoint_dir.iterdir() if p.is_dir() and p.name.startswith(model_prefix + "_")],
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    for old_path in candidates[keep:]:
        try:
            shutil.rmtree(old_path)
            logger.info(f"Removed old checkpoint: {old_path}")
        except Exception as exc:
            logger.warning(f"Failed to remove {old_path}: {exc}")


def train_model(
    model_type: str,
    train_csv: Path,
    val_csv: Path,
    test_csv: Optional[Path] = None,
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
    test_df = None
    if test_csv and test_csv.exists():
        test_df = pd.read_csv(test_csv)
    
    logger.info(f"Train: {len(train_df)} samples")
    logger.info(f"Val:   {len(val_df)} samples")
    if test_df is not None:
        logger.info(f"Test:  {len(test_df)} samples")
    
    # Create data generators
    logger.info("\nCreating data generators...")
    if model_type in [
        'mobilenetv2', 'resnet50', 'resnet101', 'resnet152',
        'efficientnetb0', 'efficientnetb3', 'efficientnetv2b0',
        'convnext_tiny', 'vit_b16'
    ]:
        train_gen, val_gen, _ = create_frame_generators(
            train_df, val_df, val_df,  # Use val_df for test_gen (placeholder)
            batch_size=batch_size,
            augment_multiplier=AUGMENT_MULTIPLIER
        )
        test_gen = None
        test_mix_gen = None
        if test_df is not None:
            test_gen = FrameDataGenerator(
                test_df,
                batch_size=batch_size,
                shuffle=False,
                augment=False,
                augment_multiplier=1,
                augment_prob=0.0
            )
            mix_prob = max(0.0, min(1.0, (AUGMENT_MULTIPLIER - 1) / max(1, AUGMENT_MULTIPLIER)))
            test_mix_gen = FrameDataGenerator(
                test_df,
                batch_size=batch_size,
                shuffle=False,
                augment=True,
                augment_multiplier=1,
                augment_prob=mix_prob
            )
    else:  # sequence-based models
        train_gen, val_gen, _ = create_sequence_generators(
            train_df, val_df, val_df,
            sequence_length=SEQUENCE_LENGTH,
            batch_size=batch_size,
            augment_multiplier=AUGMENT_MULTIPLIER
        )
        test_gen = None
        test_mix_gen = None
        if test_df is not None:
            test_gen = SequenceDataGenerator(
                test_df,
                sequence_length=SEQUENCE_LENGTH,
                batch_size=batch_size,
                shuffle=False,
                augment=False,
                augment_multiplier=1,
                augment_prob=0.0
            )
            mix_prob = max(0.0, min(1.0, (AUGMENT_MULTIPLIER - 1) / max(1, AUGMENT_MULTIPLIER)))
            test_mix_gen = SequenceDataGenerator(
                test_df,
                sequence_length=SEQUENCE_LENGTH,
                batch_size=batch_size,
                shuffle=False,
                augment=True,
                augment_multiplier=1,
                augment_prob=mix_prob
            )
    
    # Create or load model
    if resume_from:
        logger.info(f"\nLoading model from checkpoint: {resume_from}")
        model = keras.models.load_model(resume_from)
    elif model_type == "resnet50" and _env_bool("RESNET50_SCHEDULE", RESNET50_SCHEDULE):
        model, history = _train_resnet50_schedule(
            train_gen=train_gen,
            val_gen=val_gen,
            checkpoint_dir=CHECKPOINTS_DIR,
            log_dir=LOGS_DIR,
            test_gen=test_gen,
            test_mix_gen=test_mix_gen
        )
        # Skip the default training block below
    else:
        logger.info(f"\nCreating {model_type} model...")
        model = get_model(model_type, learning_rate=learning_rate)
    
    if model_type == "resnet50" and _env_bool("RESNET50_SCHEDULE", RESNET50_SCHEDULE) and not resume_from:
        callbacks_list = []
    else:
        callbacks_list = get_callbacks(
            model_name=model_type,
            checkpoint_dir=CHECKPOINTS_DIR,
            log_dir=LOGS_DIR
        )
        eval_each_epoch = _env_bool("EVAL_TEST_EACH_EPOCH", True)
        normalize_cm = _env_bool("CONFUSION_NORMALIZE", False)
        if eval_each_epoch and test_gen is not None:
            results_dir = RESULTS_DIR / model_type
            callbacks_list.append(
                TestEvalCallback(test_gen, test_mix_gen, results_dir=results_dir, normalize_cm=normalize_cm)
            )
    
    # Train
    logger.info("\n" + "=" * 80)
    logger.info("STARTING TRAINING")
    logger.info("=" * 80)
    logger.info(f"Epochs: {epochs}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"Train batches: {len(train_gen)} (with augment_multiplier={AUGMENT_MULTIPLIER})")
    logger.info(f"Val batches: {len(val_gen)}")
    
    if not (model_type == "resnet50" and _env_bool("RESNET50_SCHEDULE", RESNET50_SCHEDULE) and not resume_from):
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

    # Cleanup old checkpoints (keep latest 3 for this model type)
    cleanup_old_checkpoints(model_type, CHECKPOINTS_DIR, keep=3)
    
    # Training summary
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 80)
    
    if history is not None:
        best_epoch = np.argmax(history.history['val_accuracy'])
        logger.info(f"Best epoch: {best_epoch + 1}")
        logger.info(f"Best val_accuracy: {history.history['val_accuracy'][best_epoch]:.4f}")
        logger.info(f"Best val_loss: {history.history['val_loss'][best_epoch]:.4f}")
    else:
        best_epoch = 0
    
    return {
        'model': model,
        'history': history.history if history is not None else {},
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
        choices=[
            'mobilenetv2', 'resnet50', 'resnet101', 'resnet152',
            'efficientnetb0', 'efficientnetb3', 'efficientnetv2b0',
            'convnext_tiny', 'vit_b16',
            'lstm', 'gru', '3d_cnn'
        ],
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
        '--test-csv',
        type=str,
        default=str(PROCESSED_DIR / 'test.csv'),
        help='Path to test CSV'
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
        test_csv=Path(args.test_csv) if args.test_csv else None,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        resume_from=args.resume_from
    )
    
    logger.info(f"\nTraining complete! Final model: {result['final_model_path']}")
