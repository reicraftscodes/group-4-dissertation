"""
YOLO11 Emotion Detection Evaluation Module
=========================================

Comprehensive evaluation system for YOLO11 facial emotion detection models.
Provides detailed metrics, visualizations, and performance analysis.
"""

import time
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    precision_recall_fscore_support, accuracy_score
)

import torch
import cv2
from ultralytics import YOLO

from config import MultimodalEmotionConfig
from yolo11_emotion_model import EmotionYOLO11, create_model
from utils import save_results


class YOLO11EmotionEvaluator:
    """Comprehensive evaluation system for YOLO11 emotion detection models."""
    
    def __init__(self, config: MultimodalEmotionConfig):
        """Initialize evaluator with configuration."""
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = config.class_names
        self.num_classes = len(self.class_names)
        
        # Create output directories
        self.output_dir = Path("outputs") / "yolo11_evaluation"
        self.vis_dir = self.output_dir / "visualizations"
        self.results_dir = self.output_dir / "results"
        
        for dir_path in [self.output_dir, self.vis_dir, self.results_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        print(f"YOLO11 Evaluator initialized for {config.modality.value}")
        print(f"Device: {self.device}")
        print(f"Output directory: {self.output_dir}")
    
    def evaluate_model(
        self, 
        model_path: str, 
        test_dataset_path: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.5
    ) -> Dict:
        """
        Comprehensive evaluation of YOLO11 emotion detection model.
        
        Args:
            model_path: Path to trained YOLO11 model (.pt file)
            test_dataset_path: Path to test dataset images
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
        
        Returns:
            Comprehensive evaluation results
        """
        print("=" * 70)
        print("YOLO11 EMOTION DETECTION EVALUATION")
        print("=" * 70)
        print(f"Model: {model_path}")
        print(f"Test dataset: {test_dataset_path}")
        print(f"Confidence threshold: {conf_threshold}")
        print(f"IoU threshold: {iou_threshold}")
        
        # Load model
        model = self._load_model(model_path)
        
        # Load test dataset
        test_images, ground_truth = self._load_test_dataset(test_dataset_path)
        
        # Run inference and collect results
        predictions, inference_times = self._run_inference(
            model, test_images, conf_threshold, iou_threshold
        )
        
        # Calculate metrics
        metrics = self._calculate_metrics(predictions, ground_truth)
        
        # Analyze inference times
        timing_analysis = self._analyze_inference_times(inference_times)
        
        # Generate visualizations
        self._generate_visualizations(predictions, ground_truth, metrics)
        
        # Compile final results
        evaluation_results = {
            'model_info': {
                'model_path': model_path,
                'modality': self.config.modality.value,
                'conf_threshold': conf_threshold,
                'iou_threshold': iou_threshold,
                'evaluation_timestamp': datetime.now().isoformat()
            },
            'dataset_info': {
                'test_dataset_path': test_dataset_path,
                'total_images': len(test_images),
                'total_faces': len(ground_truth)
            },
            'performance_metrics': metrics,
            'timing_analysis': timing_analysis,
            'class_names': self.class_names
        }
        
        # Save results
        self._save_evaluation_results(evaluation_results)
        
        # Print summary
        self._print_evaluation_summary(evaluation_results)
        
        return evaluation_results
    
    def _load_model(self, model_path: str) -> YOLO:
        """Load YOLO11 model."""
        print(f"\n📥 Loading YOLO11 model from: {model_path}")
        
        try:
            # Load the trained YOLO model
            model = YOLO(model_path)
            print(f"✅ Model loaded successfully")
            print(f"   Model type: {type(model)}")
            
            return model
            
        except Exception as e:
            print(f"❌ Failed to load model: {str(e)}")
            raise e
    
    def _load_test_dataset(self, test_dataset_path: str) -> Tuple[List[str], List[Dict]]:
        """
        Load test dataset images and ground truth labels.
        
        Returns:
            Tuple of (image_paths, ground_truth_labels)
        """
        print(f"\n📂 Loading test dataset from: {test_dataset_path}")
        
        test_path = Path(test_dataset_path)
        
        # Find test images
        if (test_path / "images").exists():
            # YOLO format: test/images/ and test/labels/
            images_dir = test_path / "images"
            labels_dir = test_path / "labels"
        else:
            # Direct image folder
            images_dir = test_path
            labels_dir = None
        
        # Get image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_paths = []
        
        for ext in image_extensions:
            image_paths.extend(list(images_dir.glob(f"*{ext}")))
            image_paths.extend(list(images_dir.glob(f"*{ext.upper()}")))
        
        image_paths = sorted([str(p) for p in image_paths])
        
        # Load ground truth labels
        ground_truth = []
        
        if labels_dir and labels_dir.exists():
            # YOLO format labels
            for img_path in image_paths:
                img_name = Path(img_path).stem
                label_file = labels_dir / f"{img_name}.txt"
                
                if label_file.exists():
                    with open(label_file, 'r') as f:
                        lines = f.readlines()
                    
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            x_center, y_center, width, height = map(float, parts[1:5])
                            
                            ground_truth.append({
                                'image_path': img_path,
                                'class_id': class_id,
                                'emotion': self.class_names[class_id],
                                'bbox': [x_center, y_center, width, height],  # YOLO format
                                'bbox_format': 'yolo'
                            })
        else:
            # Parse from filename (fallback)
            for img_path in image_paths:
                img_name = Path(img_path).name
                # Try to extract emotion from filename
                emotion_class_id = self._parse_emotion_from_filename(img_name)
                
                if emotion_class_id is not None:
                    ground_truth.append({
                        'image_path': img_path,
                        'class_id': emotion_class_id,
                        'emotion': self.class_names[emotion_class_id],
                        'bbox': [0.5, 0.5, 0.8, 0.8],  # Default full-image bbox
                        'bbox_format': 'yolo'
                    })
        
        print(f"✅ Loaded {len(image_paths)} test images")
        print(f"✅ Loaded {len(ground_truth)} ground truth labels")
        
        # Print class distribution
        class_counts = {}
        for gt in ground_truth:
            emotion = gt['emotion']
            class_counts[emotion] = class_counts.get(emotion, 0) + 1
        
        print(f"\n📊 Test set class distribution:")
        for emotion, count in sorted(class_counts.items()):
            print(f"   {emotion}: {count}")
        
        return image_paths, ground_truth
    
    def _parse_emotion_from_filename(self, filename: str) -> Optional[int]:
        """Parse emotion class from filename."""
        filename_lower = filename.lower()
        
        for i, emotion in enumerate(self.class_names):
            if emotion.lower() in filename_lower:
                return i
        
        # Try alternative emotion names
        emotion_mapping = {
            'fear': 'fearful',
            'disgust': 'disgusted'
        }
        
        for alt_name, standard_name in emotion_mapping.items():
            if alt_name in filename_lower:
                try:
                    return self.class_names.index(standard_name)
                except ValueError:
                    continue
        
        return None
    
    def _run_inference(
        self, 
        model: YOLO, 
        image_paths: List[str],
        conf_threshold: float,
        iou_threshold: float
    ) -> Tuple[List[Dict], List[float]]:
        """
        Run inference on test images.
        
        Returns:
            Tuple of (predictions, inference_times)
        """
        print(f"\n🔍 Running inference on {len(image_paths)} images...")
        
        predictions = []
        inference_times = []
        
        for i, img_path in enumerate(image_paths):
            if (i + 1) % 100 == 0:
                print(f"   Processed {i + 1}/{len(image_paths)} images")
            
            # Load image
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            # Run inference with timing
            start_time = time.time()
            results = model(img, conf=conf_threshold, iou=iou_threshold, verbose=False)
            inference_time = time.time() - start_time
            
            inference_times.append(inference_time)
            
            # Process results
            img_predictions = []
            
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                
                for j in range(len(boxes)):
                    # Get detection data
                    box = boxes.xyxy[j].cpu().numpy()  # [x1, y1, x2, y2]
                    conf = float(boxes.conf[j].cpu())
                    class_id = int(boxes.cls[j].cpu())
                    
                    # Convert to YOLO format for consistency
                    img_h, img_w = img.shape[:2]
                    x_center = (box[0] + box[2]) / 2 / img_w
                    y_center = (box[1] + box[3]) / 2 / img_h
                    width = (box[2] - box[0]) / img_w
                    height = (box[3] - box[1]) / img_h
                    
                    img_predictions.append({
                        'class_id': class_id,
                        'emotion': self.class_names[class_id],
                        'confidence': conf,
                        'bbox': [x_center, y_center, width, height],
                        'bbox_format': 'yolo'
                    })
            
            predictions.append({
                'image_path': img_path,
                'detections': img_predictions
            })
        
        print(f"✅ Inference completed")
        print(f"   Average inference time: {np.mean(inference_times)*1000:.2f}ms per image")
        
        return predictions, inference_times
    
    def _calculate_metrics(self, predictions: List[Dict], ground_truth: List[Dict]) -> Dict:
        """Calculate comprehensive evaluation metrics."""
        print(f"\n📊 Calculating evaluation metrics...")
        
        # Align predictions with ground truth
        y_true = []
        y_pred = []
        detection_metrics = {'tp': 0, 'fp': 0, 'fn': 0}
        
        # Create mapping from image path to ground truth
        gt_by_image = {}
        for gt in ground_truth:
            img_path = gt['image_path']
            if img_path not in gt_by_image:
                gt_by_image[img_path] = []
            gt_by_image[img_path].append(gt)
        
        for pred_data in predictions:
            img_path = pred_data['image_path']
            detections = pred_data['detections']
            
            # Get ground truth for this image
            img_gt = gt_by_image.get(img_path, [])
            
            if len(img_gt) == 0:
                # No ground truth, all predictions are false positives
                detection_metrics['fp'] += len(detections)
                continue
            
            if len(detections) == 0:
                # No detections, all ground truth are false negatives
                detection_metrics['fn'] += len(img_gt)
                continue
            
            # For simplicity, match highest confidence detection with ground truth
            # In a full implementation, you'd use IoU-based matching
            if len(detections) > 0:
                best_detection = max(detections, key=lambda x: x['confidence'])
                
                # Assume one face per image for emotion classification
                if len(img_gt) > 0:
                    gt_emotion = img_gt[0]['class_id']
                    pred_emotion = best_detection['class_id']
                    
                    y_true.append(gt_emotion)
                    y_pred.append(pred_emotion)
                    
                    if gt_emotion == pred_emotion:
                        detection_metrics['tp'] += 1
                    else:
                        detection_metrics['fp'] += 1
                        detection_metrics['fn'] += 1
        
        # Calculate classification metrics
        if len(y_true) > 0:
            accuracy = accuracy_score(y_true, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average='weighted', zero_division=0
            )
            
            # Per-class metrics
            per_class_precision, per_class_recall, per_class_f1, support = \
                precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            # Detection metrics
            total_detections = detection_metrics['tp'] + detection_metrics['fp']
            detection_precision = detection_metrics['tp'] / max(total_detections, 1)
            detection_recall = detection_metrics['tp'] / max(
                detection_metrics['tp'] + detection_metrics['fn'], 1
            )
            detection_f1 = 2 * detection_precision * detection_recall / max(
                detection_precision + detection_recall, 1e-8
            )
            
            metrics = {
                'classification': {
                    'accuracy': float(accuracy),
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1_score': float(f1),
                    'num_samples': len(y_true)
                },
                'per_class': {
                    'precision': per_class_precision.tolist(),
                    'recall': per_class_recall.tolist(),
                    'f1_score': per_class_f1.tolist(),
                    'support': support.tolist()
                },
                'detection': {
                    'precision': float(detection_precision),
                    'recall': float(detection_recall),
                    'f1_score': float(detection_f1),
                    'true_positives': detection_metrics['tp'],
                    'false_positives': detection_metrics['fp'],
                    'false_negatives': detection_metrics['fn']
                },
                'confusion_matrix': cm.tolist(),
                'predictions': {
                    'y_true': y_true,
                    'y_pred': y_pred
                }
            }
        else:
            # No valid predictions
            metrics = {
                'classification': {
                    'accuracy': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1_score': 0.0,
                    'num_samples': 0
                },
                'detection': {
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1_score': 0.0,
                    'true_positives': 0,
                    'false_positives': detection_metrics['fp'],
                    'false_negatives': detection_metrics['fn']
                }
            }
        
        print(f"✅ Metrics calculated")
        
        return metrics
    
    def _analyze_inference_times(self, inference_times: List[float]) -> Dict:
        """Analyze inference timing performance."""
        if not inference_times:
            return {}
        
        times_ms = [t * 1000 for t in inference_times]  # Convert to milliseconds
        
        timing_analysis = {
            'mean_ms': float(np.mean(times_ms)),
            'median_ms': float(np.median(times_ms)),
            'std_ms': float(np.std(times_ms)),
            'min_ms': float(np.min(times_ms)),
            'max_ms': float(np.max(times_ms)),
            'p95_ms': float(np.percentile(times_ms, 95)),
            'p99_ms': float(np.percentile(times_ms, 99)),
            'fps': 1.0 / np.mean(inference_times),
            'total_samples': len(inference_times)
        }
        
        return timing_analysis
    
    def _generate_visualizations(self, predictions: List[Dict], ground_truth: List[Dict], metrics: Dict):
        """Generate comprehensive evaluation visualizations."""
        print(f"\n🎨 Generating evaluation visualizations...")
        
        if 'predictions' not in metrics:
            print("⚠️ No predictions available for visualization")
            return
        
        y_true = metrics['predictions']['y_true']
        y_pred = metrics['predictions']['y_pred']
        
        if not y_true:
            print("⚠️ No valid predictions for visualization")
            return
        
        # 1. Confusion Matrix
        self._plot_confusion_matrix(y_true, y_pred)
        
        # 2. Per-class Performance
        if 'per_class' in metrics:
            self._plot_per_class_metrics(metrics['per_class'])
        
        # 3. Classification Report
        self._create_classification_report_visualization(y_true, y_pred)
        
        print(f"✅ Visualizations saved to: {self.vis_dir}")
    
    def _plot_confusion_matrix(self, y_true: List[int], y_pred: List[int]):
        """Plot normalized confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm_normalized, 
            annot=True, 
            fmt='.2f', 
            cmap='Blues',
            xticklabels=self.class_names, 
            yticklabels=self.class_names,
            cbar_kws={'label': 'Normalized Count'}
        )
        plt.title('YOLO11 Emotion Detection - Normalized Confusion Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Emotion', fontsize=12)
        plt.ylabel('True Emotion', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        save_path = self.vis_dir / 'confusion_matrix_yolo11.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   📊 Confusion matrix saved: {save_path}")
    
    def _plot_per_class_metrics(self, per_class_metrics: Dict):
        """Plot per-class precision, recall, and F1-score."""
        metrics_data = {
            'Precision': per_class_metrics['precision'],
            'Recall': per_class_metrics['recall'],
            'F1-Score': per_class_metrics['f1_score']
        }
        
        x = np.arange(len(self.class_names))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        colors = ['#2E8B57', '#FF6B6B', '#4ECDC4']
        
        for i, (metric_name, values) in enumerate(metrics_data.items()):
            ax.bar(x + i * width, values, width, label=metric_name, color=colors[i], alpha=0.8)
        
        ax.set_xlabel('Emotion Classes', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('YOLO11 Per-Class Performance Metrics', fontsize=16, fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels(self.class_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.1)
        
        # Add value labels on bars
        for i, (metric_name, values) in enumerate(metrics_data.items()):
            for j, value in enumerate(values):
                ax.text(j + i * width, value + 0.01, f'{value:.2f}', 
                       ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        save_path = self.vis_dir / 'per_class_metrics_yolo11.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   📊 Per-class metrics saved: {save_path}")
    
    def _create_classification_report_visualization(self, y_true: List[int], y_pred: List[int]):
        """Create and save classification report as image."""
        # Generate classification report
        report = classification_report(
            y_true, y_pred, 
            target_names=self.class_names,
            digits=3,
            zero_division=0
        )
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.text(0.1, 0.5, report, transform=ax.transAxes, fontsize=11, 
                verticalalignment='center', fontfamily='monospace')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('YOLO11 Emotion Detection - Classification Report', 
                    fontsize=16, fontweight='bold', pad=20)
        
        save_path = self.vis_dir / 'classification_report_yolo11.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"   📊 Classification report saved: {save_path}")
    
    def _save_evaluation_results(self, results: Dict):
        """Save evaluation results to JSON file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = self.results_dir / f'yolo11_evaluation_{timestamp}.json'
        
        # Convert numpy types to native Python types for JSON serialization
        results_json = json.loads(json.dumps(results, default=str))
        
        save_results(results_json, str(results_file))
        print(f"📁 Evaluation results saved: {results_file}")
    
    def _print_evaluation_summary(self, results: Dict):
        """Print comprehensive evaluation summary."""
        print("\n" + "=" * 70)
        print("YOLO11 EMOTION DETECTION - EVALUATION SUMMARY")
        print("=" * 70)
        
        # Model info
        model_info = results['model_info']
        print(f"🤖 Model: {Path(model_info['model_path']).name}")
        print(f"📊 Modality: {model_info['modality']}")
        print(f"🎯 Confidence threshold: {model_info['conf_threshold']}")
        
        # Dataset info
        dataset_info = results['dataset_info']
        print(f"\n📂 Dataset:")
        print(f"   Total images: {dataset_info['total_images']}")
        print(f"   Total faces: {dataset_info['total_faces']}")
        
        # Performance metrics
        if 'performance_metrics' in results:
            metrics = results['performance_metrics']
            
            if 'classification' in metrics:
                cls_metrics = metrics['classification']
                print(f"\n🎯 Classification Performance:")
                print(f"   Accuracy:  {cls_metrics['accuracy']:.3f}")
                print(f"   Precision: {cls_metrics['precision']:.3f}")
                print(f"   Recall:    {cls_metrics['recall']:.3f}")
                print(f"   F1-Score:  {cls_metrics['f1_score']:.3f}")
            
            if 'detection' in metrics:
                det_metrics = metrics['detection']
                print(f"\n🔍 Detection Performance:")
                print(f"   Precision: {det_metrics['precision']:.3f}")
                print(f"   Recall:    {det_metrics['recall']:.3f}")
                print(f"   F1-Score:  {det_metrics['f1_score']:.3f}")
                print(f"   True Positives:  {det_metrics['true_positives']}")
                print(f"   False Positives: {det_metrics['false_positives']}")
                print(f"   False Negatives: {det_metrics['false_negatives']}")
        
        # Timing analysis
        if 'timing_analysis' in results:
            timing = results['timing_analysis']
            print(f"\n⏱️ Inference Timing:")
            print(f"   Mean time:   {timing['mean_ms']:.2f}ms")
            print(f"   Median time: {timing['median_ms']:.2f}ms")
            print(f"   95th percentile: {timing['p95_ms']:.2f}ms")
            print(f"   FPS: {timing['fps']:.1f}")
        
        print(f"\n📁 Visualizations: {self.vis_dir}")
        print(f"📁 Results: {self.results_dir}")
        print("=" * 70)


def evaluate_yolo11_model(
    model_path: str,
    test_dataset_path: str,
    config: Optional[MultimodalEmotionConfig] = None,
    **kwargs
) -> Dict:
    """
    Convenience function to evaluate YOLO11 emotion detection model.
    
    Args:
        model_path: Path to trained YOLO11 model
        test_dataset_path: Path to test dataset
        config: Configuration object
        **kwargs: Additional evaluation parameters
    
    Returns:
        Evaluation results
    """
    if config is None:
        config = MultimodalEmotionConfig()
    
    evaluator = YOLO11EmotionEvaluator(config)
    return evaluator.evaluate_model(model_path, test_dataset_path, **kwargs)


if __name__ == "__main__":
    # Example usage
    from config import MultimodalEmotionConfig, ModalityType
    
    # Create configuration
    config = MultimodalEmotionConfig()
    config.modality = ModalityType.RGB_ONLY
    
    # Example evaluation
    # results = evaluate_yolo11_model(
    #     model_path="models/yolo11_emotion_best.pt",
    #     test_dataset_path="yolo_emotion_dataset/test",
    #     config=config,
    #     conf_threshold=0.25,
    #     iou_threshold=0.5
    # )
    
    print("YOLO11 evaluation module ready!")
