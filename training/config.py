"""
Configuration file for handwashing detection training pipeline.

Contains all hyperparameters, paths, and dataset configurations.
"""

from pathlib import Path

# ============================================================================
# Paths
# ============================================================================
# Base directories
WORK_DIR = Path.cwd()
DATA_DIR = WORK_DIR / 'datasets'
RAW_DIR = DATA_DIR / 'raw'
PROCESSED_DIR = DATA_DIR / 'processed'
MODELS_DIR = WORK_DIR / 'models'
CHECKPOINTS_DIR = WORK_DIR / 'checkpoints'
LOGS_DIR = WORK_DIR / 'logs'
RESULTS_DIR = WORK_DIR / 'results'

# Create directories
for dir_path in [DATA_DIR, RAW_DIR, PROCESSED_DIR, MODELS_DIR, CHECKPOINTS_DIR, LOGS_DIR, RESULTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Dataset Configuration
# ============================================================================
DATASETS = {
    'pskus': {
        'name': 'PSKUS Hospital',
        'zenodo_id': '4537209',
        'type': 'frame-level',
        'classes': 7,
        'size_gb': 18.4,
        'files': ['DataSet1.zip', 'DataSet2.zip', 'DataSet3.zip', 'DataSet4.zip', 
                  'DataSet5.zip', 'DataSet6.zip', 'DataSet7.zip', 'DataSet8.zip',
                  'DataSet9.zip', 'DataSet10.zip', 'DataSet11.zip'],
        'metadata': ['summary.csv', 'statistics.csv']
    },
    'metc': {
        'name': 'METC Lab',
        'zenodo_id': '4537342',
        'type': 'frame-level',
        'classes': 7,
        'size_gb': 2.12,
        'files': ['Interface_number_1.zip', 'Interface_number_2.zip', 'Interface_number_3.zip'],
        'metadata': ['summary.csv', 'statistics.csv']
    },
    'synthetic_blender_rozakar': {
        'name': 'Synthetic Hand-Washing Gesture (Özakar & Gedikli 2025)',
        'gdrive_links': [
            'https://drive.google.com/uc?id=1EW3JQvElcuXzawxEMRkA8YXwK_Ipiv-p&export=download',
            'https://drive.google.com/uc?id=163TsrDe4q5KTQGCv90JRYFkCs7AGxFip&export=download',
            'https://drive.google.com/uc?id=1GxyTYfSodumH78NbjWdmbjm8JP8AOkAY&export=download',
            'https://drive.google.com/uc?id=1IoRsgBBr8qoC3HO-vEr6E7K4UZ6ku6-1&export=download',
            'https://drive.google.com/uc?id=1svCYnwDazy5FN1DYSgqbGscvDKL_YnID&export=download'
        ],
        'type': 'frame-level',
        'classes': 8,
        'size_gb': 6.0,
        'modalities': ['rgb', 'depth', 'depth_isolated', 'masks'],
        'structure': 'character/environment/gesture/modality/*.png'
    },
    'kaggle': {
        'name': 'Kaggle WHO6',
        'url': 'https://github.com/atiselsts/data/raw/master/kaggle-dataset-6classes.tar',
        'type': 'clip-level',
        'classes': 7,
        'size_gb': 1.21
    }
}

# ============================================================================
# Model Configuration
# ============================================================================
# Image and sequence settings
IMG_SIZE = (224, 224)  # MobileNetV2 standard input size
SEQUENCE_LENGTH = 16   # Number of frames for temporal models
FRAME_SKIP = 2         # Extract every Nth frame (1=all, 2=half)

# Classes
NUM_CLASSES = 7
CLASS_NAMES = [
    'Other',                    # Class 0
    'Step1_PalmToPalm',        # Class 1
    'Step2_PalmOverDorsum',    # Class 2
    'Step3_InterlacedFingers', # Class 3
    'Step4_BackOfFingers',     # Class 4
    'Step5_ThumbRub',          # Class 5
    'Step6_Fingertips'         # Class 6
]

# Class mapping for Kaggle dataset
KAGGLE_CLASS_MAPPING = {
    'step1': 1,
    'step2': 2,
    'step3': 3,
    'step4': 4,
    'step5': 5,
    'step6': 6,
    'other': 0
}

# ============================================================================
# Training Hyperparameters
# ============================================================================
# Training settings
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4
PATIENCE = 10           # Early stopping patience
REDUCE_LR_PATIENCE = 5  # ReduceLROnPlateau patience
MIN_LR = 1e-7           # Minimum learning rate

# Data split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ============================================================================
# Data Augmentation
# ============================================================================
AUGMENTATION_CONFIG = {
    'rotation_range': 15,
    'width_shift_range': 0.1,
    'height_shift_range': 0.1,
    'shear_range': 0.1,
    'zoom_range': 0.1,
    'horizontal_flip': True,
    'brightness_range': (0.9, 1.1),
    'fill_mode': 'nearest'
}

# ============================================================================
# Model Architectures
# ============================================================================
MODEL_CONFIGS = {
    'mobilenetv2': {
        'name': 'MobileNetV2 Frame Classifier',
        'type': 'frame-based',
        'backbone': 'MobileNetV2',
        'pretrained_weights': 'imagenet',
        'freeze_backbone': True,
        'classifier_units': [256],
        'dropout_rate': 0.5,
        'input_shape': (*IMG_SIZE, 3)
    },
    'resnet50': {
        'name': 'ResNet50 Frame Classifier',
        'type': 'frame-based',
        'backbone': 'ResNet50',
        'pretrained_weights': 'imagenet',
        'freeze_backbone': True,
        'classifier_units': [512, 256],
        'dropout_rate': 0.5,
        'input_shape': (*IMG_SIZE, 3)
    },
    'efficientnetb0': {
        'name': 'EfficientNetB0 Frame Classifier',
        'type': 'frame-based',
        'backbone': 'EfficientNetB0',
        'pretrained_weights': 'imagenet',
        'freeze_backbone': True,
        'classifier_units': [256],
        'dropout_rate': 0.4,
        'input_shape': (*IMG_SIZE, 3)
    },
    'lstm': {
        'name': 'LSTM Temporal Classifier',
        'type': 'sequence-based',
        'backbone': 'MobileNetV2',
        'pretrained_weights': 'imagenet',
        'freeze_backbone': True,
        'lstm_units': 128,
        'dense_units': [64],
        'dropout_rate': 0.5,
        'sequence_length': SEQUENCE_LENGTH,
        'input_shape': (SEQUENCE_LENGTH, *IMG_SIZE, 3)
    },
    'gru': {
        'name': 'GRU Temporal Classifier',
        'type': 'sequence-based',
        'backbone': 'MobileNetV2',
        'pretrained_weights': 'imagenet',
        'freeze_backbone': True,
        'gru_units': 128,
        'dense_units': [64],
        'dropout_rate': 0.5,
        'sequence_length': SEQUENCE_LENGTH,
        'input_shape': (SEQUENCE_LENGTH, *IMG_SIZE, 3)
    },
    '3d_cnn': {
        'name': '3D CNN Temporal Classifier',
        'type': 'sequence-based',
        'backbone': 'custom_3dcnn',
        'pretrained_weights': None,
        'freeze_backbone': False,
        'conv_filters': [32, 64, 128],
        'kernel_size': (3, 3, 3),
        'pool_size': (1, 2, 2),
        'dense_units': [256],
        'dropout_rate': 0.5,
        'sequence_length': SEQUENCE_LENGTH,
        'input_shape': (SEQUENCE_LENGTH, *IMG_SIZE, 3)
    }
}

# ============================================================================
# Callbacks Configuration
# ============================================================================
CHECKPOINT_CONFIG = {
    'save_best_only': True,
    'save_weights_only': False,
    'monitor': 'val_accuracy',
    'mode': 'max',
    'verbose': 1
}

EARLY_STOPPING_CONFIG = {
    'monitor': 'val_loss',
    'patience': PATIENCE,
    'restore_best_weights': True,
    'verbose': 1
}

REDUCE_LR_CONFIG = {
    'monitor': 'val_loss',
    'factor': 0.5,
    'patience': REDUCE_LR_PATIENCE,
    'min_lr': MIN_LR,
    'verbose': 1
}

# ============================================================================
# Evaluation Configuration
# ============================================================================
EVALUATION_METRICS = [
    'accuracy',
    'precision',
    'recall',
    'f1-score',
    'confusion_matrix',
    'classification_report'
]

# ============================================================================
# Reproducibility
# ============================================================================
RANDOM_SEED = 42

# ============================================================================
# Logging
# ============================================================================
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# Sampling options (used in notebook to explore frame skipping)
FRAME_SKIP_OPTIONS = [1, 2, 4]
