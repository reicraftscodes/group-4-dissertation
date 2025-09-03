"""
Utility functions for YOLO11 Multimodal Emotion Detection
========================================================

This module contains utility functions for visualization, metrics, file handling, etc.
"""

import os
import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Optional, Union

import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

from config import MultimodalEmotionConfig


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------
def calculate_metrics(y_true: List, y_pred: List, class_names: List[str]) -> Dict:
    """Calculate comprehensive evaluation metrics with explicit class labels."""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        classification_report, confusion_matrix
    )

    # Ensure ints
    y_true = [int(x) for x in y_true]
    y_pred = [int(x) for x in y_pred]

    num_classes = len(class_names)
    labels = list(range(num_classes))

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro',
                                           labels=labels, zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro',
                                     labels=labels, zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro',
                             labels=labels, zero_division=0),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted',
                                              labels=labels, zero_division=0),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted',
                                        labels=labels, zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted',
                                labels=labels, zero_division=0),
    }

    # Per-class metrics aligned to labels
    precision_per_class = precision_score(y_true, y_pred, average=None,
                                          labels=labels, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None,
                                    labels=labels, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None,
                            labels=labels, zero_division=0)

    for i, name in enumerate(class_names):
        metrics[f'precision_{name}'] = float(precision_per_class[i])
        metrics[f'recall_{name}']    = float(recall_per_class[i])
        metrics[f'f1_{name}']        = float(f1_per_class[i])

    # Confusion matrix & classification report with explicit labels
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    metrics['classification_report'] = classification_report(
        y_true, y_pred, labels=labels, target_names=class_names,
        zero_division=0, output_dict=True
    )
    return metrics


# -----------------------------------------------------------------------------
# I/O helpers
# -----------------------------------------------------------------------------
def create_output_dirs(config: MultimodalEmotionConfig):
    """Create necessary output directories."""
    dirs = [
        config.output_path,
        config.model_save_path,
        config.logs_path,
        f"{config.output_path}/visualizations",
        f"{config.output_path}/results",
        f"{config.output_path}/checkpoints"
    ]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    print(f"Created output directories under: {config.output_path}")


def save_config_to_file(config: MultimodalEmotionConfig, filepath: str):
    """Save configuration to a YAML file."""
    config_dict = {
        'model_parameters': {
            'model_size': config.model_size,
            'num_classes': config.num_classes,
            'input_size': config.input_size,
            'modality': config.modality.value,
            'fusion_type': config.fusion_type.value
        },
        'training_parameters': {
            'batch_size': config.batch_size,
            'epochs': config.epochs,
            'learning_rate': config.learning_rate,
            'weight_decay': config.weight_decay
        },
        'dataset_parameters': {
            'dataset_root': config.dataset_root,
            'class_names': config.class_names,
            'train_split': config.train_split,
            'val_split': config.val_split
        },
        'detection_parameters': {
            'conf_threshold': config.conf_threshold,
            'iou_threshold': config.iou_threshold,
            'max_detections': config.max_detections
        }
    }
    with open(filepath, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)


def load_config_from_file(filepath: str) -> Dict:
    """Load configuration from YAML file."""
    with open(filepath, 'r') as f:
        return yaml.safe_load(f)


# -----------------------------------------------------------------------------
# Visualization helpers
# -----------------------------------------------------------------------------
def visualize_sample(sample: Dict, config: MultimodalEmotionConfig, save_path: Optional[str] = None):
    """Visualize a single sample from the dataset."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # RGB image
    if 'rgb_image' in sample and sample['rgb_image'] is not None:
        rgb_img = sample['rgb_image'].numpy().transpose(1, 2, 0)
        rgb_img = np.clip(rgb_img, 0, 1)
        axes[0].imshow(rgb_img)
        axes[0].set_title(f"RGB - {sample['emotion']}")
        axes[0].axis('off')

    # Thermal image
    if 'thermal_image' in sample and sample['thermal_image'] is not None:
        thermal_img = sample['thermal_image'].numpy().transpose(1, 2, 0)
        thermal_img = np.clip(thermal_img, 0, 1)
        axes[1].imshow(thermal_img)
        axes[1].set_title(f"Thermal - {sample['emotion']}")
        axes[1].axis('off')

    # Combined image (if available)
    if 'combined_image' in sample and sample['combined_image'] is not None:
        combined_img = sample['combined_image'].numpy().transpose(1, 2, 0)
        combined_img = np.clip(combined_img, 0, 1)
        axes[2].imshow(combined_img)
        axes[2].set_title(f"Combined - {sample['emotion']}")
        axes[2].axis('off')
    else:
        if 'image' in sample:
            img = sample['image'].numpy().transpose(1, 2, 0)
            img = np.clip(img, 0, 1)
            axes[2].imshow(img)
            axes[2].set_title(f"Image - {sample['emotion']}")
            axes[2].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Sample visualization saved to: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_class_distribution(dataset_files: Dict, config: MultimodalEmotionConfig, save_path: Optional[str] = None):
    """Plot class distribution for train/test splits."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    modes = ['train', 'test']
    colors = ['skyblue', 'lightcoral']

    for i, mode in enumerate(modes):
        if mode in dataset_files and len(dataset_files[mode]) > 0:
            # Count emotions
            emotion_counts = {}
            for file_data in dataset_files[mode]:
                emotion = file_data['emotion']
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

            emotions = list(emotion_counts.keys())
            counts = list(emotion_counts.values())

            axes[i].bar(emotions, counts, color=colors[i], alpha=0.7)
            axes[i].set_title(f'{mode.capitalize()} Set Distribution (Total: {sum(counts)})')
            axes[i].set_xlabel('Emotion')
            axes[i].set_ylabel('Count')
            axes[i].tick_params(axis='x', rotation=45)

            for j, count in enumerate(counts):
                axes[i].text(j, count + 0.5, str(count), ha='center', va='bottom')
        else:
            axes[i].text(0.5, 0.5, f'No {mode} data', ha='center', va='center',
                         transform=axes[i].transAxes, fontsize=16)
            axes[i].set_title(f'{mode.capitalize()} Set')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Class distribution plot saved to: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_training_history(history: Dict, save_path: Optional[str] = None):
    """Plot comprehensive training history similar to VT results."""
    # Create a 2x3 subplot layout to match VT style
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Colors for consistency
    train_color = '#1f77b4'
    
    # 1. Loss Over Epochs
    if 'train_loss' in history:
        epochs = range(1, len(history['train_loss']) + 1)
        axes[0, 0].plot(epochs, history['train_loss'], color=train_color, linewidth=2, marker='o', markersize=3)
        axes[0, 0].set_title('Loss Over Epochs', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_xlim(1, len(history['train_loss']))

    # 2. Accuracy Over Epochs
    if 'train_acc' in history:
        epochs = range(1, len(history['train_acc']) + 1)
        axes[0, 1].plot(epochs, history['train_acc'], color=train_color, linewidth=2, marker='o', markersize=3)
        axes[0, 1].set_title('Accuracy Over Epochs', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_xlim(1, len(history['train_acc']))
        axes[0, 1].set_ylim(0, 1)

    # 3. F1 Over Epochs
    if 'train_f1' in history:
        epochs = range(1, len(history['train_f1']) + 1)
        axes[0, 2].plot(epochs, history['train_f1'], color=train_color, linewidth=2, marker='o', markersize=3)
        axes[0, 2].set_title('F1 Over Epochs', fontsize=12, fontweight='bold')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('F1')
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].set_xlim(1, len(history['train_f1']))
        axes[0, 2].set_ylim(0, 1)

    # 4. Precision Over Epochs
    if 'train_precision' in history:
        epochs = range(1, len(history['train_precision']) + 1)
        axes[1, 0].plot(epochs, history['train_precision'], color=train_color, linewidth=2, marker='o', markersize=3)
        axes[1, 0].set_title('Precision Over Epochs', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_xlim(1, len(history['train_precision']))
        axes[1, 0].set_ylim(0, 1)

    # 5. Recall Over Epochs
    if 'train_recall' in history:
        epochs = range(1, len(history['train_recall']) + 1)
        axes[1, 1].plot(epochs, history['train_recall'], color=train_color, linewidth=2, marker='o', markersize=3)
        axes[1, 1].set_title('Recall Over Epochs', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_xlim(1, len(history['train_recall']))
        axes[1, 1].set_ylim(0, 1)

    # 6. Learning Rate Schedule
    if 'learning_rate' in history:
        epochs = range(1, len(history['learning_rate']) + 1)
        axes[1, 2].plot(epochs, history['learning_rate'], color='#ff7f0e', linewidth=2, marker='o', markersize=3)
        axes[1, 2].set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Learning Rate')
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].set_xlim(1, len(history['learning_rate']))

    plt.suptitle('Training History', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training history plot saved to: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_confusion_matrix(y_true: List, y_pred: List, class_names: List[str],
                          save_path: Optional[str] = None):
    """Plot confusion matrix with explicit labels to align with class_names."""
    from sklearn.metrics import confusion_matrix as sk_cm

    labels = list(range(len(class_names)))
    cm = sk_cm(y_true, y_pred, labels=labels)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    else:
        plt.show()
    plt.close()


# -----------------------------------------------------------------------------
# JSON save/load (robust to NumPy types)
# -----------------------------------------------------------------------------
def _to_serializable(obj):
    # numpy scalars
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if obj is np.nan:
        return None
    # numpy arrays
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    # containers
    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    # everything else
    return obj


def plot_per_class_metrics(metrics: Dict, class_names: List[str], save_path: Optional[str] = None):
    """Plot per-class performance metrics in VT style."""
    # Extract per-class metrics
    precision_scores = []
    recall_scores = []
    f1_scores = []
    
    for class_name in class_names:
        precision_scores.append(metrics.get(f'precision_{class_name}', 0.0))
        recall_scores.append(metrics.get(f'recall_{class_name}', 0.0))
        f1_scores.append(metrics.get(f'f1_{class_name}', 0.0))
    
    # Create the plot
    x = np.arange(len(class_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create bars
    bars1 = ax.bar(x - width, precision_scores, width, label='Precision', color='#1f77b4', alpha=0.8)
    bars2 = ax.bar(x, recall_scores, width, label='Recall', color='#ff7f0e', alpha=0.8)
    bars3 = ax.bar(x + width, f1_scores, width, label='F1-Score', color='#2ca02c', alpha=0.8)
    
    # Add value labels on bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    add_value_labels(bars3)
    
    # Customize the plot
    ax.set_xlabel('Emotion Classes', fontweight='bold')
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([name.capitalize() for name in class_names])
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Per-class metrics plot saved to: {save_path}")
    else:
        plt.show()
    plt.close()


def create_formatted_classification_report(y_true: List[int], y_pred: List[int], 
                                         class_names: List[str], save_path: Optional[str] = None) -> Dict:
    """Create a formatted classification report similar to VT results."""
    from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
    
    # Calculate overall metrics
    accuracy = accuracy_score(y_true, y_pred)
    macro_precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    weighted_precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    weighted_recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Get per-class metrics
    labels = list(range(len(class_names)))
    per_class_precision = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    
    # Count support (number of samples per class)
    support = [sum(1 for t in y_true if t == i) for i in range(len(class_names))]
    
    # Create formatted report
    report_dict = {}
    
    # Per-class metrics
    for i, class_name in enumerate(class_names):
        report_dict[class_name] = {
            'precision': float(per_class_precision[i]),
            'recall': float(per_class_recall[i]),
            'f1-score': float(per_class_f1[i]),
            'support': int(support[i])
        }
    
    # Overall metrics
    report_dict['accuracy'] = float(accuracy)
    report_dict['macro avg'] = {
        'precision': float(macro_precision),
        'recall': float(macro_recall),
        'f1-score': float(macro_f1),
        'support': len(y_true)
    }
    report_dict['weighted avg'] = {
        'precision': float(weighted_precision),
        'recall': float(weighted_recall),
        'f1-score': float(weighted_f1),
        'support': len(y_true)
    }
    
    if save_path:
        # Create a nicely formatted table image
        _create_classification_report_image(report_dict, class_names, save_path)
    
    return report_dict


def _create_classification_report_image(report_dict: Dict, class_names: List[str], save_path: str):
    """Create a formatted classification report image."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Prepare data for table
    rows = []
    headers = ['Class', 'Precision', 'Recall', 'F1-Score', 'Support']
    
    # Add per-class rows
    for class_name in class_names:
        if class_name in report_dict:
            row = [
                class_name.capitalize(),
                f"{report_dict[class_name]['precision']:.4f}",
                f"{report_dict[class_name]['recall']:.4f}",
                f"{report_dict[class_name]['f1-score']:.4f}",
                str(report_dict[class_name]['support'])
            ]
            rows.append(row)
    
    # Add overall metrics
    rows.append(['', '', '', '', ''])  # Empty row
    rows.append(['Accuracy', '', '', f"{report_dict['accuracy']:.4f}", str(report_dict['macro avg']['support'])])
    rows.append(['Macro Avg', 
                f"{report_dict['macro avg']['precision']:.4f}",
                f"{report_dict['macro avg']['recall']:.4f}",
                f"{report_dict['macro avg']['f1-score']:.4f}",
                str(report_dict['macro avg']['support'])])
    rows.append(['Weighted Avg', 
                f"{report_dict['weighted avg']['precision']:.4f}",
                f"{report_dict['weighted avg']['recall']:.4f}",
                f"{report_dict['weighted avg']['f1-score']:.4f}",
                str(report_dict['weighted avg']['support'])])
    
    # Create table
    table = ax.table(cellText=rows, colLabels=headers, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Style the table
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style accuracy row
    accuracy_row = len(class_names) + 1
    for i in range(len(headers)):
        table[(accuracy_row, i)].set_facecolor('#E3F2FD')
        table[(accuracy_row, i)].set_text_props(weight='bold')
    
    ax.axis('off')
    ax.set_title('Classification Report', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_results(results: dict, filepath: str):
    """Save results dict to JSON, converting numpy types to native Python."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(_to_serializable(results), f, indent=2)


def load_results(filepath: str) -> Dict:
    """Load results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def print_metrics_summary(metrics: Dict):
    """Print a summary of evaluation metrics (dynamically uses reported class names)."""
    print("\n" + "="*60)
    print("EVALUATION METRICS SUMMARY")
    print("="*60)

    print(f"Accuracy: {metrics.get('accuracy', 0.0):.4f}")
    print(f"Precision (Macro): {metrics.get('precision_macro', 0.0):.4f}")
    print(f"Recall (Macro): {metrics.get('recall_macro', 0.0):.4f}")
    print(f"F1-Score (Macro): {metrics.get('f1_macro', 0.0):.4f}")

    print(f"\nWeighted Averages:")
    print(f"Precision (Weighted): {metrics.get('precision_weighted', 0.0):.4f}")
    print(f"Recall (Weighted): {metrics.get('recall_weighted', 0.0):.4f}")
    print(f"F1-Score (Weighted): {metrics.get('f1_weighted', 0.0):.4f}")

    # Use classification_report keys to list per-class metrics (skip summary rows)
    report = metrics.get('classification_report', {})
    skip = {'accuracy', 'macro avg', 'weighted avg'}
    class_rows = [k for k in report.keys() if k not in skip]
    if class_rows:
        print(f"\nPer-Class Metrics:")
        for cname in class_rows:
            row = report[cname]
            try:
                p = float(row.get('precision', 0.0))
                r = float(row.get('recall', 0.0))
                f1 = float(row.get('f1-score', 0.0))
                print(f"{cname.capitalize():>10}: P={p:.3f}, R={r:.3f}, F1={f1:.3f}")
            except Exception:
                # Fallback to keys we set earlier
                p = metrics.get(f'precision_{cname}', 0.0)
                r = metrics.get(f'recall_{cname}', 0.0)
                f1 = metrics.get(f'f1_{cname}', 0.0)
                print(f"{cname.capitalize():>10}: P={p:.3f}, R={r:.3f}, F1={f1:.3f}")

    print("="*60)


def ensure_dir_exists(filepath: str):
    """Ensure directory exists for a given filepath."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Additional helper functions for evaluation visualizations
# -----------------------------------------------------------------------------
def save_confusion_matrix(y_true: List[int], y_pred: List[int], class_names: List[str],
                          outdir: Union[str, Path], normalize: bool = True,
                          fname: str = None):
    """
    Generate and save a confusion matrix plot (with explicit labels).

    Args:
        y_true: List of ground truth class indices.
        y_pred: List of predicted class indices.
        class_names: Names of the classes.
        outdir: Directory where the plot will be saved.
        normalize: Whether to normalize the confusion matrix by true label counts.
        fname: Optional file name. If None, auto-select based on normalize.
    """
    from sklearn.metrics import confusion_matrix as sk_cm

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    labels = list(range(len(class_names)))
    cm = sk_cm(y_true, y_pred, labels=labels)

    if normalize:
        cm_plot = cm.astype(float)
        with np.errstate(all='ignore'):
            cm_plot = cm_plot / cm_plot.sum(axis=1, keepdims=True)
            cm_plot = np.nan_to_num(cm_plot)
    else:
        cm_plot = cm

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_plot, annot=True, fmt='.2f' if normalize else 'd', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    if fname is None:
        fname = 'confusion_matrix_norm.png' if normalize else 'confusion_matrix.png'
    out_path = outdir / fname
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to: {out_path}")


def save_classification_report(y_true: List[int], y_pred: List[int], class_names: List[str],
                               outdir: Union[str, Path]):
    """
    Compute and save a classification report (precision, recall, f1-score per class).
    Uses explicit labels so it stays aligned with class_names even if some classes are absent.
    """
    from sklearn.metrics import classification_report as sk_report

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    labels = list(range(len(class_names)))

    # JSON (structured)
    report = sk_report(
        y_true, y_pred,
        labels=labels,
        target_names=class_names,
        zero_division=0,
        output_dict=True
    )
    with open(outdir / 'classification_report.json', 'w') as f:
        json.dump(report, f, indent=2)

    # TXT (pretty table)
    report_txt = sk_report(
        y_true, y_pred,
        labels=labels,
        target_names=class_names,
        zero_division=0,
        output_dict=False
    )
    with open(outdir / 'classification_report.txt', 'w') as f:
        f.write(report_txt)

    print(f"Classification report saved to: {outdir/'classification_report.json'}")


def save_misclassified_grid(records: List[Dict[str, str]], outdir: Union[str, Path], k: int = 16):
    """
    Save a grid of misclassified images with labels.

    Each record in `records` should be a dict with keys:
        'image_path' (path to the image file),
        'true' (true class name),
        'pred' (predicted class name),
        (optional) 'confidence' (float).
    """
    from PIL import Image
    import math
    import random

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Filter misclassified
    mis = [r for r in records if r.get('true') != r.get('pred')]
    if not mis:
        print("No misclassified samples to display.")
        return

    random.shuffle(mis)
    mis = mis[:k]
    num_images = len(mis)

    cols = min(4, num_images)
    rows = math.ceil(num_images / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))

    if isinstance(axes, np.ndarray):
        axes_list = axes.reshape(-1)
    else:
        axes_list = np.array([axes])

    for idx, record in enumerate(mis):
        ax = axes_list[idx]
        try:
            img = Image.open(record['image_path']).convert('RGB')
        except Exception:
            img = Image.new('RGB', (224, 224), (255, 255, 255))
        ax.imshow(img)
        title = f"T: {record.get('true','?')} | P: {record.get('pred','?')}"
        if 'confidence' in record:
            title += f" | Conf: {record['confidence']:.2f}"
        ax.set_title(title, fontsize=9)
        ax.axis('off')

    for ax in axes_list[num_images:]:
        ax.axis('off')

    plt.tight_layout()
    save_path = outdir / 'misclassified_grid.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Misclassified samples grid saved to: {save_path}")


# -----------------------------------------------------------------------------
# Experiment logging & comparisons
# -----------------------------------------------------------------------------
def get_model_size_info(model_size: str) -> Dict:
    """Get model size information for YOLO11."""
    size_info = {
        'n': {'params': '2.6M', 'gflops': '6.5', 'description': 'Nano - Fastest'},
        's': {'params': '9.4M', 'gflops': '21.5', 'description': 'Small - Fast'},
        'm': {'params': '20.1M', 'gflops': '51.4', 'description': 'Medium - Balanced'},
        'l': {'params': '25.3M', 'gflops': '68.9', 'description': 'Large - Accurate'},
        'x': {'params': '56.9M', 'gflops': '194.9', 'description': 'Extra Large - Most Accurate'}
    }
    return size_info.get(model_size, size_info['n'])


def log_experiment(config: MultimodalEmotionConfig, results: Dict, log_path: str):
    """Log experiment details and results."""
    experiment_log = {
        'experiment_id': f"{config.modality.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        'config': config.to_dict(),
        'results': results,
        'model_info': get_model_size_info(config.model_size),
        'timestamp': datetime.now().isoformat()
    }

    ensure_dir_exists(log_path)
    with open(log_path, 'w') as f:
        json.dump(experiment_log, f, indent=2)
    print(f"Experiment logged to: {log_path}")


def compare_modalities(results_dict: Dict[str, Dict], save_path: Optional[str] = None):
    """Compare results across different modalities."""
    modalities = list(results_dict.keys())
    metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        values = [results_dict[mod].get(metric, 0) for mod in modalities]
        bars = axes[i].bar(modalities, values,
                           color=['skyblue', 'lightgreen', 'lightcoral', 'lightyellow'])
        axes[i].set_title(f'{metric.replace("_", " ").title()}')
        axes[i].set_ylabel('Score')
        axes[i].set_ylim(0, 1)

        for bar, value in zip(bars, values):
            axes[i].text(bar.get_x() + bar.get_width()/2,
                         bar.get_height() + 0.01,
                         f'{value:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Modality comparison plot saved to: {save_path}")
    else:
        plt.show()
    plt.close()
