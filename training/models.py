"""
Model architectures for handwashing detection training pipeline.

Provides MobileNetV2 (frame-based), LSTM, and GRU (sequence-based) models.
"""

import logging
from typing import Tuple, Optional
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2, ResNet50, EfficientNetB0

# Import configuration
from config import (
    IMG_SIZE,
    NUM_CLASSES,
    SEQUENCE_LENGTH,
    MODEL_CONFIGS,
    LEARNING_RATE,
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
tf.random.set_seed(RANDOM_SEED)


def create_mobilenetv2_model(
    input_shape: Tuple[int, int, int] = (*IMG_SIZE, 3),
    num_classes: int = NUM_CLASSES,
    learning_rate: float = LEARNING_RATE,
    freeze_backbone: bool = True
) -> keras.Model:
    """
    Create MobileNetV2-based frame classifier.
    
    Architecture:
    - MobileNetV2 backbone (ImageNet pretrained)
    - GlobalAveragePooling2D
    - Dense(256) + ReLU + Dropout(0.5)
    - Dense(num_classes) + Softmax
    
    Args:
        input_shape: Input image shape (H, W, C)
        num_classes: Number of output classes
        learning_rate: Learning rate for Adam optimizer
        freeze_backbone: Freeze MobileNetV2 weights
    
    Returns:
        Compiled Keras model
    """
    config = MODEL_CONFIGS['mobilenetv2']
    
    # Load pretrained MobileNetV2
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights=config['pretrained_weights']
    )
    
    # Freeze backbone if specified
    base_model.trainable = not freeze_backbone
    
    # Build classifier
    inputs = keras.Input(shape=input_shape, name='image_input')
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    
    # Classification head
    for units in config['classifier_units']:
        x = layers.Dense(units, activation='relu', name=f'fc_{units}')(x)
        x = layers.Dropout(config['dropout_rate'], name=f'dropout_{units}')(x)
    
    outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)
    
    model = keras.Model(inputs, outputs, name='mobilenetv2_classifier')
    
    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.TopKCategoricalAccuracy(k=2, name='top2_accuracy'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall')
        ]
    )
    
    logger.info(f"Created {config['name']}")
    logger.info(f"  Total params: {model.count_params():,}")
    logger.info(f"  Trainable params: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")
    logger.info(f"  Backbone frozen: {freeze_backbone}")
    
    return model


def create_resnet50_model(
    input_shape: Tuple[int, int, int] = (*IMG_SIZE, 3),
    num_classes: int = NUM_CLASSES,
    learning_rate: float = LEARNING_RATE,
    freeze_backbone: bool = True
) -> keras.Model:
    """
    Create ResNet50-based frame classifier.
    """
    config = MODEL_CONFIGS['resnet50']
    base_model = ResNet50(
        input_shape=input_shape,
        include_top=False,
        weights=config['pretrained_weights']
    )
    base_model.trainable = not freeze_backbone

    inputs = keras.Input(shape=input_shape, name='image_input')
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    for units in config['classifier_units']:
        x = layers.Dense(units, activation='relu', name=f'fc_{units}')(x)
        x = layers.Dropout(config['dropout_rate'], name=f'dropout_{units}')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)

    model = keras.Model(inputs, outputs, name='resnet50_classifier')
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.TopKCategoricalAccuracy(k=2, name='top2_accuracy'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall')
        ]
    )
    logger.info(f"Created {config['name']}")
    logger.info(f"  Total params: {model.count_params():,}")
    logger.info(f"  Trainable params: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")
    logger.info(f"  Backbone frozen: {freeze_backbone}")
    return model


def create_efficientnetb0_model(
    input_shape: Tuple[int, int, int] = (*IMG_SIZE, 3),
    num_classes: int = NUM_CLASSES,
    learning_rate: float = LEARNING_RATE,
    freeze_backbone: bool = True
) -> keras.Model:
    """
    Create EfficientNetB0-based frame classifier.
    """
    config = MODEL_CONFIGS['efficientnetb0']
    base_model = EfficientNetB0(
        input_shape=input_shape,
        include_top=False,
        weights=config['pretrained_weights']
    )
    base_model.trainable = not freeze_backbone

    inputs = keras.Input(shape=input_shape, name='image_input')
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    for units in config['classifier_units']:
        x = layers.Dense(units, activation='relu', name=f'fc_{units}')(x)
        x = layers.Dropout(config['dropout_rate'], name=f'dropout_{units}')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)

    model = keras.Model(inputs, outputs, name='efficientnetb0_classifier')
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.TopKCategoricalAccuracy(k=2, name='top2_accuracy'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall')
        ]
    )
    logger.info(f"Created {config['name']}")
    logger.info(f"  Total params: {model.count_params():,}")
    logger.info(f"  Trainable params: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")
    logger.info(f"  Backbone frozen: {freeze_backbone}")
    return model


def create_lstm_model(
    sequence_length: int = SEQUENCE_LENGTH,
    img_shape: Tuple[int, int, int] = (*IMG_SIZE, 3),
    num_classes: int = NUM_CLASSES,
    learning_rate: float = LEARNING_RATE,
    freeze_backbone: bool = True
) -> keras.Model:
    """
    Create LSTM-based temporal classifier.
    
    Architecture:
    - MobileNetV2 feature extractor (per frame, shared weights)
    - TimeDistributed(GlobalAveragePooling2D)
    - LSTM(128)
    - Dense(64) + ReLU + Dropout(0.5)
    - Dense(num_classes) + Softmax
    
    Args:
        sequence_length: Number of frames per sequence
        img_shape: Single frame shape (H, W, C)
        num_classes: Number of output classes
        learning_rate: Learning rate for Adam optimizer
        freeze_backbone: Freeze MobileNetV2 weights
    
    Returns:
        Compiled Keras model
    """
    config = MODEL_CONFIGS['lstm']
    
    # Feature extractor (shared across frames)
    base_model = MobileNetV2(
        input_shape=img_shape,
        include_top=False,
        weights=config['pretrained_weights']
    )
    base_model.trainable = not freeze_backbone
    
    # Frame encoder
    frame_input = keras.Input(shape=img_shape, name='frame_input')
    x = base_model(frame_input, training=False)
    x = layers.GlobalAveragePooling2D(name='frame_features')(x)
    frame_encoder = keras.Model(frame_input, x, name='frame_encoder')
    
    # Sequence model
    sequence_input = keras.Input(shape=(sequence_length, *img_shape), name='sequence_input')
    x = layers.TimeDistributed(frame_encoder, name='time_distributed_encoder')(sequence_input)
    x = layers.LSTM(config['lstm_units'], return_sequences=False, name='lstm')(x)
    
    # Classification head
    for units in config['dense_units']:
        x = layers.Dense(units, activation='relu', name=f'fc_{units}')(x)
        x = layers.Dropout(config['dropout_rate'], name=f'dropout_{units}')(x)
    
    outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)
    
    model = keras.Model(sequence_input, outputs, name='lstm_classifier')
    
    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.TopKCategoricalAccuracy(k=2, name='top2_accuracy'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall')
        ]
    )
    
    logger.info(f"Created {config['name']}")
    logger.info(f"  Total params: {model.count_params():,}")
    logger.info(f"  Trainable params: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")
    logger.info(f"  Sequence length: {sequence_length}")
    
    return model


def create_gru_model(
    sequence_length: int = SEQUENCE_LENGTH,
    img_shape: Tuple[int, int, int] = (*IMG_SIZE, 3),
    num_classes: int = NUM_CLASSES,
    learning_rate: float = LEARNING_RATE,
    freeze_backbone: bool = True
) -> keras.Model:
    """
    Create GRU-based temporal classifier.
    
    Architecture:
    - MobileNetV2 feature extractor (per frame, shared weights)
    - TimeDistributed(GlobalAveragePooling2D)
    - GRU(128)
    - Dense(64) + ReLU + Dropout(0.5)
    - Dense(num_classes) + Softmax
    
    Args:
        sequence_length: Number of frames per sequence
        img_shape: Single frame shape (H, W, C)
        num_classes: Number of output classes
        learning_rate: Learning rate for Adam optimizer
        freeze_backbone: Freeze MobileNetV2 weights
    
    Returns:
        Compiled Keras model
    """
    config = MODEL_CONFIGS['gru']
    
    # Feature extractor (shared across frames)
    base_model = MobileNetV2(
        input_shape=img_shape,
        include_top=False,
        weights=config['pretrained_weights']
    )
    base_model.trainable = not freeze_backbone
    
    # Frame encoder
    frame_input = keras.Input(shape=img_shape, name='frame_input')
    x = base_model(frame_input, training=False)
    x = layers.GlobalAveragePooling2D(name='frame_features')(x)
    frame_encoder = keras.Model(frame_input, x, name='frame_encoder')
    
    # Sequence model
    sequence_input = keras.Input(shape=(sequence_length, *img_shape), name='sequence_input')
    x = layers.TimeDistributed(frame_encoder, name='time_distributed_encoder')(sequence_input)
    x = layers.GRU(config['gru_units'], return_sequences=False, name='gru')(x)
    
    # Classification head
    for units in config['dense_units']:
        x = layers.Dense(units, activation='relu', name=f'fc_{units}')(x)
        x = layers.Dropout(config['dropout_rate'], name=f'dropout_{units}')(x)
    
    outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)
    
    model = keras.Model(sequence_input, outputs, name='gru_classifier')
    
    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.TopKCategoricalAccuracy(k=2, name='top2_accuracy'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall')
        ]
    )
    
    logger.info(f"Created {config['name']}")
    logger.info(f"  Total params: {model.count_params():,}")
    logger.info(f"  Trainable params: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")
    logger.info(f"  Sequence length: {sequence_length}")
    
    return model


def create_3d_cnn_model(
    sequence_length: int = SEQUENCE_LENGTH,
    img_shape: Tuple[int, int, int] = (*IMG_SIZE, 3),
    num_classes: int = NUM_CLASSES,
    learning_rate: float = LEARNING_RATE,
    conv_filters=None,
    kernel_size=(3, 3, 3),
    pool_size=(1, 2, 2),
    dense_units=None,
    dropout_rate: float = 0.5
) -> keras.Model:
    """
    Create a lightweight 3D CNN for video sequences.
    Input shape: (T, H, W, C) where T=sequence_length.
    """
    config = MODEL_CONFIGS['3d_cnn']
    conv_filters = conv_filters or config['conv_filters']
    dense_units = dense_units or config['dense_units']

    sequence_input = keras.Input(shape=(sequence_length, *img_shape), name='sequence_input')
    x = sequence_input
    # 3D convolutional trunk
    for idx, filters in enumerate(conv_filters):
        x = layers.Conv3D(filters, kernel_size, padding='same', activation='relu', name=f'conv3d_{idx+1}')(x)
        x = layers.BatchNormalization(name=f'bn3d_{idx+1}')(x)
        x = layers.MaxPool3D(pool_size=pool_size, name=f'pool3d_{idx+1}')(x)
        x = layers.Dropout(dropout_rate * 0.5, name=f'dropout3d_{idx+1}')(x)

    x = layers.GlobalAveragePooling3D(name='global_avg_pool3d')(x)
    for units in dense_units:
        x = layers.Dense(units, activation='relu', name=f'fc_{units}')(x)
        x = layers.Dropout(dropout_rate, name=f'dropout_{units}')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)

    model = keras.Model(sequence_input, outputs, name='cnn3d_classifier')
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.TopKCategoricalAccuracy(k=2, name='top2_accuracy'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall')
        ]
    )
    logger.info(f"Created {config['name']}")
    logger.info(f"  Total params: {model.count_params():,}")
    logger.info(f"  Trainable params: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")
    logger.info(f"  Sequence length: {sequence_length}")
    return model


def get_model(
    model_type: str,
    **kwargs
) -> keras.Model:
    """
    Factory function to create models by name.
    
    Args:
        model_type: Model type ('mobilenetv2', 'lstm', 'gru')
        **kwargs: Additional arguments for model creation
    
    Returns:
        Compiled Keras model
    
    Raises:
        ValueError: If model_type is not recognized
    """
    model_type = model_type.lower()
    
    if model_type == 'mobilenetv2':
        return create_mobilenetv2_model(**kwargs)
    if model_type == 'resnet50':
        return create_resnet50_model(**kwargs)
    if model_type == 'efficientnetb0':
        return create_efficientnetb0_model(**kwargs)
    if model_type == 'lstm':
        return create_lstm_model(**kwargs)
    if model_type == 'gru':
        return create_gru_model(**kwargs)
    if model_type == '3d_cnn':
        return create_3d_cnn_model(**kwargs)
    raise ValueError("Unknown model type: mobilenetv2, resnet50, efficientnetb0, lstm, gru, 3d_cnn")


def print_model_summary(model: keras.Model, save_path: Optional[str] = None) -> None:
    """
    Print detailed model summary.
    
    Args:
        model: Keras model
        save_path: Optional path to save summary as text file
    """
    logger.info("=" * 80)
    logger.info(f"MODEL SUMMARY: {model.name}")
    logger.info("=" * 80)
    
    # Print to console
    model.summary()
    
    # Save to file if specified
    if save_path:
        with open(save_path, 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))
        logger.info(f"Summary saved to: {save_path}")


if __name__ == "__main__":
    # Test model creation
    logger.info("Testing model creation...")
    
    # Test MobileNetV2
    logger.info("\n" + "=" * 80)
    model_mobilenet = create_mobilenetv2_model()
    print_model_summary(model_mobilenet)
    
    # Test LSTM
    logger.info("\n" + "=" * 80)
    model_lstm = create_lstm_model()
    print_model_summary(model_lstm)
    
    # Test GRU
    logger.info("\n" + "=" * 80)
    model_gru = create_gru_model()
    print_model_summary(model_gru)
    
    logger.info("\nAll models created successfully!")
