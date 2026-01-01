"""
Model architectures for handwashing detection training pipeline.

Provides MobileNetV2 (frame-based), LSTM, and GRU (sequence-based) models.
"""

import logging
import os
from typing import Tuple, Optional
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import (
    MobileNetV2,
    ResNet50,
    ResNet101,
    ResNet152,
    EfficientNetB0,
    EfficientNetB3,
    EfficientNetV2B0,
    ConvNeXtTiny,
)
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.resnet import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as efficientnetv2_preprocess
from tensorflow.keras.applications.convnext import preprocess_input as convnext_preprocess

# Import configuration
from config import (
    IMG_SIZE,
    NUM_CLASSES,
    SEQUENCE_LENGTH,
    MODEL_CONFIGS,
    LEARNING_RATE,
    OPTIMIZER_NAME,
    WEIGHT_DECAY,
    LABEL_SMOOTHING,
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


def _build_optimizer(
    learning_rate: float = LEARNING_RATE,
    optimizer_name: str = OPTIMIZER_NAME,
    weight_decay: float = WEIGHT_DECAY
) -> keras.optimizers.Optimizer:
    name = os.getenv("OPTIMIZER_NAME", optimizer_name).lower()
    weight_decay = float(os.getenv("WEIGHT_DECAY", weight_decay))
    if name == "adamw":
        return keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    if name == "adam":
        return keras.optimizers.Adam(learning_rate=learning_rate)
    raise ValueError(f"Unknown optimizer: {optimizer_name}")


def _build_loss(label_smoothing: float = LABEL_SMOOTHING) -> keras.losses.Loss:
    label_smoothing = float(os.getenv("LABEL_SMOOTHING", label_smoothing))
    return keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing)


def _apply_preprocess(inputs: tf.Tensor, preprocess_fn, name: str) -> tf.Tensor:
    x = layers.Lambda(lambda t: t * 255.0, name=f"{name}_rescale")(inputs)
    return layers.Lambda(preprocess_fn, name=f"{name}_preprocess")(x)


def create_mobilenetv2_model(
    input_shape: Tuple[int, int, int] = (*IMG_SIZE, 3),
    num_classes: int = NUM_CLASSES,
    learning_rate: float = LEARNING_RATE,
    freeze_backbone: bool = True,
    optimizer_name: str = OPTIMIZER_NAME,
    weight_decay: float = WEIGHT_DECAY,
    label_smoothing: float = LABEL_SMOOTHING
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
    x = _apply_preprocess(inputs, mobilenet_preprocess, "mobilenetv2")
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    
    # Classification head
    for units in config['classifier_units']:
        x = layers.Dense(units, activation='relu', name=f'fc_{units}')(x)
        x = layers.Dropout(config['dropout_rate'], name=f'dropout_{units}')(x)
    
    outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)
    
    model = keras.Model(inputs, outputs, name='mobilenetv2_classifier')
    
    # Compile
    model.compile(
        optimizer=_build_optimizer(learning_rate, optimizer_name, weight_decay),
        loss=_build_loss(label_smoothing),
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
    freeze_backbone: bool = True,
    optimizer_name: str = OPTIMIZER_NAME,
    weight_decay: float = WEIGHT_DECAY,
    label_smoothing: float = LABEL_SMOOTHING
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
    base_model._name = "resnet50_backbone"
    base_model.trainable = not freeze_backbone

    inputs = keras.Input(shape=input_shape, name='image_input')
    x = _apply_preprocess(inputs, resnet_preprocess, "resnet50")
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    for units in config['classifier_units']:
        x = layers.Dense(units, activation='relu', name=f'fc_{units}')(x)
        x = layers.Dropout(config['dropout_rate'], name=f'dropout_{units}')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)

    model = keras.Model(inputs, outputs, name='resnet50_classifier')
    model.compile(
        optimizer=_build_optimizer(learning_rate, optimizer_name, weight_decay),
        loss=_build_loss(label_smoothing),
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
    freeze_backbone: bool = True,
    optimizer_name: str = OPTIMIZER_NAME,
    weight_decay: float = WEIGHT_DECAY,
    label_smoothing: float = LABEL_SMOOTHING
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
    base_model._name = "efficientnetb0_backbone"
    base_model.trainable = not freeze_backbone

    inputs = keras.Input(shape=input_shape, name='image_input')
    x = _apply_preprocess(inputs, efficientnet_preprocess, "efficientnetb0")
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    for units in config['classifier_units']:
        x = layers.Dense(units, activation='relu', name=f'fc_{units}')(x)
        x = layers.Dropout(config['dropout_rate'], name=f'dropout_{units}')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)

    model = keras.Model(inputs, outputs, name='efficientnetb0_classifier')
    model.compile(
        optimizer=_build_optimizer(learning_rate, optimizer_name, weight_decay),
        loss=_build_loss(label_smoothing),
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


def create_efficientnetb3_model(
    input_shape: Tuple[int, int, int] = (*IMG_SIZE, 3),
    num_classes: int = NUM_CLASSES,
    learning_rate: float = LEARNING_RATE,
    freeze_backbone: bool = True,
    optimizer_name: str = OPTIMIZER_NAME,
    weight_decay: float = WEIGHT_DECAY,
    label_smoothing: float = LABEL_SMOOTHING
) -> keras.Model:
    """
    Create EfficientNetB3-based frame classifier.
    """
    config = MODEL_CONFIGS['efficientnetb3']
    base_model = EfficientNetB3(
        input_shape=input_shape,
        include_top=False,
        weights=config['pretrained_weights']
    )
    base_model._name = "efficientnetb3_backbone"
    base_model.trainable = not freeze_backbone

    inputs = keras.Input(shape=input_shape, name='image_input')
    x = _apply_preprocess(inputs, efficientnet_preprocess, "efficientnetb3")
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    for units in config['classifier_units']:
        x = layers.Dense(units, activation='relu', name=f'fc_{units}')(x)
        x = layers.Dropout(config['dropout_rate'], name=f'dropout_{units}')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)

    model = keras.Model(inputs, outputs, name='efficientnetb3_classifier')
    model.compile(
        optimizer=_build_optimizer(learning_rate, optimizer_name, weight_decay),
        loss=_build_loss(label_smoothing),
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


def create_efficientnetv2b0_model(
    input_shape: Tuple[int, int, int] = (*IMG_SIZE, 3),
    num_classes: int = NUM_CLASSES,
    learning_rate: float = LEARNING_RATE,
    freeze_backbone: bool = True,
    optimizer_name: str = OPTIMIZER_NAME,
    weight_decay: float = WEIGHT_DECAY,
    label_smoothing: float = LABEL_SMOOTHING
) -> keras.Model:
    """
    Create EfficientNetV2B0-based frame classifier.
    """
    config = MODEL_CONFIGS['efficientnetv2b0']
    base_model = EfficientNetV2B0(
        input_shape=input_shape,
        include_top=False,
        weights=config['pretrained_weights']
    )
    base_model._name = "efficientnetv2b0_backbone"
    base_model.trainable = not freeze_backbone

    inputs = keras.Input(shape=input_shape, name='image_input')
    x = _apply_preprocess(inputs, efficientnetv2_preprocess, "efficientnetv2b0")
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    for units in config['classifier_units']:
        x = layers.Dense(units, activation='relu', name=f'fc_{units}')(x)
        x = layers.Dropout(config['dropout_rate'], name=f'dropout_{units}')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)

    model = keras.Model(inputs, outputs, name='efficientnetv2b0_classifier')
    model.compile(
        optimizer=_build_optimizer(learning_rate, optimizer_name, weight_decay),
        loss=_build_loss(label_smoothing),
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


def create_resnet101_model(
    input_shape: Tuple[int, int, int] = (*IMG_SIZE, 3),
    num_classes: int = NUM_CLASSES,
    learning_rate: float = LEARNING_RATE,
    freeze_backbone: bool = True,
    optimizer_name: str = OPTIMIZER_NAME,
    weight_decay: float = WEIGHT_DECAY,
    label_smoothing: float = LABEL_SMOOTHING
) -> keras.Model:
    """
    Create ResNet101-based frame classifier.
    """
    config = MODEL_CONFIGS['resnet101']
    base_model = ResNet101(
        input_shape=input_shape,
        include_top=False,
        weights=config['pretrained_weights']
    )
    base_model._name = "resnet101_backbone"
    base_model.trainable = not freeze_backbone

    inputs = keras.Input(shape=input_shape, name='image_input')
    x = _apply_preprocess(inputs, resnet_preprocess, "resnet101")
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    for units in config['classifier_units']:
        x = layers.Dense(units, activation='relu', name=f'fc_{units}')(x)
        x = layers.Dropout(config['dropout_rate'], name=f'dropout_{units}')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)

    model = keras.Model(inputs, outputs, name='resnet101_classifier')
    model.compile(
        optimizer=_build_optimizer(learning_rate, optimizer_name, weight_decay),
        loss=_build_loss(label_smoothing),
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


def create_resnet152_model(
    input_shape: Tuple[int, int, int] = (*IMG_SIZE, 3),
    num_classes: int = NUM_CLASSES,
    learning_rate: float = LEARNING_RATE,
    freeze_backbone: bool = True,
    optimizer_name: str = OPTIMIZER_NAME,
    weight_decay: float = WEIGHT_DECAY,
    label_smoothing: float = LABEL_SMOOTHING
) -> keras.Model:
    """
    Create ResNet152-based frame classifier.
    """
    config = MODEL_CONFIGS['resnet152']
    base_model = ResNet152(
        input_shape=input_shape,
        include_top=False,
        weights=config['pretrained_weights']
    )
    base_model._name = "resnet152_backbone"
    base_model.trainable = not freeze_backbone

    inputs = keras.Input(shape=input_shape, name='image_input')
    x = _apply_preprocess(inputs, resnet_preprocess, "resnet152")
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    for units in config['classifier_units']:
        x = layers.Dense(units, activation='relu', name=f'fc_{units}')(x)
        x = layers.Dropout(config['dropout_rate'], name=f'dropout_{units}')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)

    model = keras.Model(inputs, outputs, name='resnet152_classifier')
    model.compile(
        optimizer=_build_optimizer(learning_rate, optimizer_name, weight_decay),
        loss=_build_loss(label_smoothing),
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


def create_convnext_tiny_model(
    input_shape: Tuple[int, int, int] = (*IMG_SIZE, 3),
    num_classes: int = NUM_CLASSES,
    learning_rate: float = LEARNING_RATE,
    freeze_backbone: bool = True,
    optimizer_name: str = OPTIMIZER_NAME,
    weight_decay: float = WEIGHT_DECAY,
    label_smoothing: float = LABEL_SMOOTHING
) -> keras.Model:
    """
    Create ConvNeXtTiny-based frame classifier.
    """
    config = MODEL_CONFIGS['convnext_tiny']
    base_model = ConvNeXtTiny(
        input_shape=input_shape,
        include_top=False,
        weights=config['pretrained_weights']
    )
    base_model._name = "convnext_tiny_backbone"
    base_model.trainable = not freeze_backbone

    inputs = keras.Input(shape=input_shape, name='image_input')
    x = _apply_preprocess(inputs, convnext_preprocess, "convnext_tiny")
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    for units in config['classifier_units']:
        x = layers.Dense(units, activation='relu', name=f'fc_{units}')(x)
        x = layers.Dropout(config['dropout_rate'], name=f'dropout_{units}')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)

    model = keras.Model(inputs, outputs, name='convnext_tiny_classifier')
    model.compile(
        optimizer=_build_optimizer(learning_rate, optimizer_name, weight_decay),
        loss=_build_loss(label_smoothing),
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


def create_vit_b16_model(
    input_shape: Tuple[int, int, int] = (*IMG_SIZE, 3),
    num_classes: int = NUM_CLASSES,
    learning_rate: float = LEARNING_RATE,
    optimizer_name: str = OPTIMIZER_NAME,
    weight_decay: float = WEIGHT_DECAY,
    label_smoothing: float = LABEL_SMOOTHING
) -> keras.Model:
    """
    Create ViT-B16 classifier via keras-cv (if available).
    """
    try:
        import keras_cv
    except Exception as exc:
        raise ValueError("vit_b16 requires keras-cv (pip install keras-cv)") from exc
    model = keras_cv.models.ViTClassifier(
        input_shape=input_shape,
        num_classes=num_classes,
        activation="softmax",
        include_rescaling=True,
        pretrained="imagenet21k+imagenet2012",
    )
    model.compile(
        optimizer=_build_optimizer(learning_rate, optimizer_name, weight_decay),
        loss=_build_loss(label_smoothing),
        metrics=[
            'accuracy',
            keras.metrics.TopKCategoricalAccuracy(k=2, name='top2_accuracy'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall')
        ]
    )
    logger.info("Created ViT-B16 classifier")
    return model


def create_lstm_model(
    sequence_length: int = SEQUENCE_LENGTH,
    img_shape: Tuple[int, int, int] = (*IMG_SIZE, 3),
    num_classes: int = NUM_CLASSES,
    learning_rate: float = LEARNING_RATE,
    freeze_backbone: bool = True,
    optimizer_name: str = OPTIMIZER_NAME,
    weight_decay: float = WEIGHT_DECAY,
    label_smoothing: float = LABEL_SMOOTHING
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
    frame_pre = _apply_preprocess(frame_input, mobilenet_preprocess, "mobilenetv2")
    x = base_model(frame_pre, training=False)
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
        optimizer=_build_optimizer(learning_rate, optimizer_name, weight_decay),
        loss=_build_loss(label_smoothing),
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
    freeze_backbone: bool = True,
    optimizer_name: str = OPTIMIZER_NAME,
    weight_decay: float = WEIGHT_DECAY,
    label_smoothing: float = LABEL_SMOOTHING
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
    frame_pre = _apply_preprocess(frame_input, mobilenet_preprocess, "mobilenetv2")
    x = base_model(frame_pre, training=False)
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
        optimizer=_build_optimizer(learning_rate, optimizer_name, weight_decay),
        loss=_build_loss(label_smoothing),
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
    optimizer_name: str = OPTIMIZER_NAME,
    weight_decay: float = WEIGHT_DECAY,
    label_smoothing: float = LABEL_SMOOTHING,
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
        optimizer=_build_optimizer(learning_rate, optimizer_name, weight_decay),
        loss=_build_loss(label_smoothing),
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
    if model_type == 'efficientnetb3':
        return create_efficientnetb3_model(**kwargs)
    if model_type == 'efficientnetv2b0':
        return create_efficientnetv2b0_model(**kwargs)
    if model_type == 'resnet101':
        return create_resnet101_model(**kwargs)
    if model_type == 'resnet152':
        return create_resnet152_model(**kwargs)
    if model_type == 'convnext_tiny':
        return create_convnext_tiny_model(**kwargs)
    if model_type == 'vit_b16':
        return create_vit_b16_model(**kwargs)
    if model_type == 'lstm':
        return create_lstm_model(**kwargs)
    if model_type == 'gru':
        return create_gru_model(**kwargs)
    if model_type == '3d_cnn':
        return create_3d_cnn_model(**kwargs)
    raise ValueError(
        "Unknown model type: mobilenetv2, resnet50, resnet101, resnet152, "
        "efficientnetb0, efficientnetb3, efficientnetv2b0, convnext_tiny, "
        "vit_b16, lstm, gru, 3d_cnn"
    )


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
