"""
Evaluation module for handwashing detection models.

Provides comprehensive evaluation metrics and model comparison.
"""

import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
    top_k_accuracy_score
)
import tensorflow as tf
from tensorflow import keras

# Import project modules
from config import (
    PROCESSED_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    BATCH_SIZE,
    NUM_CLASSES,
    CLASS_NAMES,
    SEQUENCE_LENGTH,
    RANDOM_SEED,
    LOG_FORMAT,
    LOG_DATE_FORMAT
)
from data_generators import create_frame_generators, create_sequence_generators

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt=LOG_DATE_FORMAT
)
logger = logging.getLogger(__name__)

# Set random seed
np.random.seed(RANDOM_SEED)


def evaluate_model(
    model_path: str,
    test_csv: Path,
    model_type: str,
    batch_size: int = BATCH_SIZE,
    save_results: bool = True
) -> Dict[str, Any]:
    """
    Evaluate a trained model on test set.
    
    Args:
        model_path: Path to saved model
        test_csv: Path to test CSV
        model_type: Model type ('mobilenetv2', 'lstm', 'gru')
        batch_size: Batch size
        save_results: Save results to disk
    
    Returns:
        Dictionary with evaluation results
    """
    logger.info("=" * 80)
    logger.info(f"EVALUATION: {model_type.upper()}")
    logger.info("=" * 80)
    
    # Load model
    logger.info(f"\nLoading model: {model_path}")
    model = keras.models.load_model(model_path)
    
    # Load test data
    logger.info("\nLoading test data...")
    test_df = pd.read_csv(test_csv)
    logger.info(f"Test samples: {len(test_df)}")
    
    # Create data generator
    logger.info("\nCreating data generator...")
    if model_type in [
        'mobilenetv2', 'resnet50', 'resnet101', 'resnet152',
        'efficientnetb0', 'efficientnetb3', 'efficientnetv2b0',
        'convnext_tiny', 'vit_b16'
    ]:
        _, _, test_gen = create_frame_generators(
            test_df, test_df, test_df,
            batch_size=batch_size
        )
    else:  # sequence-based
        _, _, test_gen = create_sequence_generators(
            test_df, test_df, test_df,
            sequence_length=SEQUENCE_LENGTH,
            batch_size=batch_size
        )
    
    # Predict
    logger.info("\nGenerating predictions...")
    y_pred_proba = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Get true labels
    y_true = []
    for i in range(len(test_gen)):
        _, batch_y = test_gen[i]
        y_true.extend(np.argmax(batch_y, axis=1))
    y_true = np.array(y_true[:len(y_pred)])
    
    # Calculate metrics
    logger.info("\nCalculating metrics...")
    
    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Precision, Recall, F1
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average='weighted'
    )
    
    # Top-2 accuracy
    # Provide full label set so sklearn doesn't error when a class is missing in y_true
    top2_acc = top_k_accuracy_score(y_true, y_pred_proba, k=2, labels=list(range(NUM_CLASSES)))
    
    # Per-class metrics
    per_class_metrics = classification_report(
        y_true, y_pred,
        target_names=CLASS_NAMES,
        output_dict=True
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Log results
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 80)
    logger.info(f"Accuracy:      {accuracy:.4f}")
    logger.info(f"Top-2 Accuracy: {top2_acc:.4f}")
    logger.info(f"Precision:     {precision:.4f}")
    logger.info(f"Recall:        {recall:.4f}")
    logger.info(f"F1-Score:      {f1:.4f}")
    
    logger.info("\nPer-Class Metrics:")
    for i, class_name in enumerate(CLASS_NAMES):
        metrics = per_class_metrics[class_name]
        logger.info(f"  {class_name}:")
        logger.info(f"    Precision: {metrics['precision']:.4f}")
        logger.info(f"    Recall:    {metrics['recall']:.4f}")
        logger.info(f"    F1-Score:  {metrics['f1-score']:.4f}")
        logger.info(f"    Support:   {int(metrics['support'])}")
    
    # Prepare results dictionary
    results = {
        'model_type': model_type,
        'model_path': model_path,
        'accuracy': accuracy,
        'top2_accuracy': top2_acc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'per_class_metrics': per_class_metrics,
        'confusion_matrix': cm,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    
    # Save results
    if save_results:
        results_dir = RESULTS_DIR / model_type
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save confusion matrix plot
        plot_confusion_matrix(
            cm,
            CLASS_NAMES,
            save_path=results_dir / 'confusion_matrix.png'
        )
        
        # Save classification report
        report_path = results_dir / 'classification_report.txt'
        with open(report_path, 'w') as f:
            f.write(classification_report(y_true, y_pred, target_names=CLASS_NAMES))
        
        # Save metrics CSV
        metrics_df = pd.DataFrame({
            'model': [model_type],
            'accuracy': [accuracy],
            'top2_accuracy': [top2_acc],
            'precision': [precision],
            'recall': [recall],
            'f1_score': [f1]
        })
        metrics_df.to_csv(results_dir / 'metrics.csv', index=False)
        
        logger.info(f"\nResults saved to: {results_dir}")
    
    return results


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Optional[Path] = None,
    normalize: bool = True
):
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        save_path: Path to save plot
        normalize: Normalize confusion matrix
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    sns.heatmap(
        cm,
        annot=True,
        fmt='.2f' if normalize else 'd',
        cmap='Blues',
        xticklabels=[cn.split('_')[-1] for cn in class_names],
        yticklabels=[cn.split('_')[-1] for cn in class_names],
        cbar_kws={'label': 'Proportion' if normalize else 'Count'},
        ax=ax
    )
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix' + (' (Normalized)' if normalize else ''), fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Confusion matrix saved: {save_path}")
    
    plt.close()


def compare_models(
    model_results: Dict[str, Dict[str, Any]],
    save_path: Optional[Path] = None
):
    """
    Compare multiple models.
    
    Args:
        model_results: Dictionary of {model_name: results_dict}
        save_path: Path to save comparison plot
    """
    # Extract metrics
    models = list(model_results.keys())
    metrics = ['accuracy', 'top2_accuracy', 'precision', 'recall', 'f1_score']
    
    data = {metric: [] for metric in metrics}
    for model_name in models:
        for metric in metrics:
            data[metric].append(model_results[model_name][metric])
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Bar plot
    x = np.arange(len(models))
    width = 0.15
    
    for i, metric in enumerate(metrics):
        axes[0].bar(x + i * width, data[metric], width, label=metric.replace('_', ' ').title())
    
    axes[0].set_xlabel('Model', fontsize=12)
    axes[0].set_ylabel('Score', fontsize=12)
    axes[0].set_title('Model Comparison - All Metrics', fontsize=14)
    axes[0].set_xticks(x + width * 2)
    axes[0].set_xticklabels(models)
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    
    # Radar plot
    from math import pi
    
    angles = [n / len(metrics) * 2 * pi for n in range(len(metrics))]
    angles += angles[:1]
    
    ax = plt.subplot(122, polar=True)
    
    for model_name in models:
        values = [model_results[model_name][m] for m in metrics]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=model_name)
        ax.fill(angles, values, alpha=0.15)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
    ax.set_ylim(0, 1)
    ax.set_title('Model Comparison - Radar Chart', fontsize=14, y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Comparison plot saved: {save_path}")
    
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate handwashing detection model"
    )
    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path to saved model'
    )
    parser.add_argument(
        '--model-type',
        type=str,
        required=True,
        choices=[
            'mobilenetv2', 'resnet50', 'resnet101', 'resnet152',
            'efficientnetb0', 'efficientnetb3', 'efficientnetv2b0',
            'convnext_tiny', 'vit_b16',
            'lstm', 'gru', '3d_cnn'
        ],
        help='Model architecture'
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
        '--no-save',
        action='store_true',
        help='Do not save results to disk'
    )
    
    args = parser.parse_args()
    
    # Evaluate model
    results = evaluate_model(
        model_path=args.model_path,
        test_csv=Path(args.test_csv),
        model_type=args.model_type,
        batch_size=args.batch_size,
        save_results=not args.no_save
    )
    
    logger.info("\nEvaluation complete!")
