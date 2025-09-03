"""
Main entry point for REAL YOLO11 Facial Emotion Detection
========================================================

This script provides the interface for the actual YOLO11 implementation:
- Dataset conversion to YOLO format
- YOLO11 model training for emotion detection
- Model evaluation and inference
- Face detection + emotion classification
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional

from config import MultimodalEmotionConfig, ModalityType
from yolo11_trainer import YOLO11EmotionTrainer, train_yolo11_emotion_model
from yolo_dataset_converter import convert_emotion_dataset_to_yolo
from yolo11_emotion_model import EmotionYOLO11, create_model
from yolo11_evaluate import YOLO11EmotionEvaluator, evaluate_yolo11_model


def convert_dataset_command(config: MultimodalEmotionConfig, args):
    """Convert emotion dataset to YOLO format."""
    print("=" * 60)
    print("CONVERTING EMOTION DATASET TO YOLO FORMAT")
    print("=" * 60)
    
    modalities = []
    if args.modality in ['rgb_only', 'early_fusion', 'late_fusion']:
        modalities.append('rgb')
    if args.modality in ['thermal_only', 'early_fusion', 'late_fusion']:
        modalities.append('thermal')
    
    
    output_dir = args.output_dir or "yolo_emotion_dataset"
    
    data_yaml_path = convert_emotion_dataset_to_yolo(
        config,
        output_dir=output_dir,
        modalities=modalities
    )
    
    print(f"\n✅ Dataset conversion completed!")
    print(f"📁 Output directory: {output_dir}")
    print(f"📄 Data config: {data_yaml_path}")
    print(f"\n🚀 To train YOLO11:")
    print(f"python main_yolo11.py --train --modality {args.modality} --data {data_yaml_path}")


def fuse_datasets_command(config: MultimodalEmotionConfig, args):
    """Fuse existing RGB and Thermal datasets for early/late fusion."""
    print("=" * 60)
    print("FUSING EXISTING RGB AND THERMAL DATASETS")
    print("=" * 60)
    
    # Check if both datasets exist
    rgb_dataset = Path("yolo_emotion_dataset")
    thermal_dataset = Path("yolo_thermal_dataset")
    
    if not rgb_dataset.exists():
        print(f"❌ RGB dataset not found: {rgb_dataset}")
        print("Please run: python main_yolo11.py --convert --modality rgb_only")
        sys.exit(1)
    
    if not thermal_dataset.exists():
        print(f"❌ Thermal dataset not found: {thermal_dataset}")
        print("Please run: python main_yolo11.py --convert --modality thermal_only")
        sys.exit(1)
    
    print(f"✅ Found RGB dataset: {rgb_dataset}")
    print(f"✅ Found Thermal dataset: {thermal_dataset}")
    
    # Determine output directory
    output_dir = args.output_dir or f"yolo_{args.modality}_dataset"
    
    # Import the fusion function
    from yolo_dataset_converter import fuse_existing_datasets
    
    try:
        data_yaml_path = fuse_existing_datasets(
            rgb_dataset_path=rgb_dataset,
            thermal_dataset_path=thermal_dataset,
            output_dir=output_dir,
            fusion_type=args.modality
        )
        
        print(f"\n✅ Dataset fusion completed!")
        print(f"📁 Output directory: {output_dir}")
        print(f"📄 Data config: {data_yaml_path}")
        print(f"\n🚀 To train YOLO11:")
        print(f"python main_yolo11.py --train --modality {args.modality} --data {data_yaml_path}")
        
    except Exception as e:
        print(f"\n❌ Dataset fusion failed: {str(e)}")
        sys.exit(1)


def train_command(config: MultimodalEmotionConfig, args):
    """Train YOLO11 emotion detection model."""
    print("=" * 60)
    print(f"TRAINING YOLO11 FOR EMOTION DETECTION ({config.modality.value.upper()})")
    print("=" * 60)
    
    # Set training parameters
    if args.epochs:
        config.epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    
    # Train model
    try:
        results = train_yolo11_emotion_model(
            config,
            data_yaml_path=args.data,
            epochs=config.epochs,
            batch_size=config.batch_size,
            img_size=config.input_size
        )
        
        print(f"\n✅ Training completed successfully!")
        print(f"📊 Training time: {results.get('training_time_minutes', 0):.1f} minutes")
        print(f"💾 Model saved to: {results.get('model_path', 'Unknown')}")
        
    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        sys.exit(1)


def evaluate_command(config: MultimodalEmotionConfig, args):
    """Evaluate trained YOLO11 model."""
    if not args.model:
        print("❌ Model path required for evaluation. Use --model PATH_TO_MODEL")
        sys.exit(1)
    
    if not args.test_data:
        print("❌ Test dataset path required. Use --test-data PATH_TO_TEST_DATASET")
        sys.exit(1)
    
    print("=" * 60)
    print("YOLO11 EMOTION DETECTION EVALUATION")
    print("=" * 60)
    
    # Run evaluation
    try:
        results = evaluate_yolo11_model(
            model_path=args.model,
            test_dataset_path=args.test_data,
            config=config,
            conf_threshold=args.conf or config.conf_threshold,
            iou_threshold=args.iou or 0.5
        )
        
        print(f"\n✅ Evaluation completed successfully!")
        print(f"📊 Results saved to: outputs/yolo11_evaluation/")
        
    except Exception as e:
        print(f"\n❌ Evaluation failed: {str(e)}")
        sys.exit(1)


def inference_command(config: MultimodalEmotionConfig, args):
    """Run inference on images."""
    if not args.model:
        print("❌ Model path required for inference. Use --model PATH_TO_MODEL")
        sys.exit(1)
    
    if not args.input:
        print("❌ Input image path required. Use --input PATH_TO_IMAGE")
        sys.exit(1)
    
    print("=" * 60)
    print("YOLO11 EMOTION DETECTION INFERENCE")
    print("=" * 60)
    
    # Load model
    model = create_model(config)
    model.load_model(args.model)
    
    # Run inference
    results = model.detect_faces_and_emotions(
        args.input, 
        conf_threshold=args.conf or config.conf_threshold
    )
    
    print(f"\n🔍 Detection results for: {args.input}")
    for i, result in enumerate(results):
        print(f"\nImage {i+1}:")
        if result['faces']:
            for j, face in enumerate(result['faces']):
                bbox = face['bbox']
                print(f"  Face {j+1}:")
                print(f"    Emotion: {face['emotion']}")
                print(f"    Confidence: {face['emotion_confidence']:.3f}")
                print(f"    BBox: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")
        else:
            print("  No faces detected")


def test_dataset_command(config: MultimodalEmotionConfig):
    """Test dataset loading and show statistics."""
    print("=" * 60)
    print("TESTING DATASET FOR YOLO11 CONVERSION")
    print("=" * 60)
    
    dataset_root = Path(config.dataset_root)
    if not dataset_root.exists():
        print(f"❌ Dataset root not found: {dataset_root}")
        return
    
    # Check folder structure
    folders = ['RGB', 'Thermal', 'RgbAug', 'ThermalAug']
    found_folders = []
    
    for folder in folders:
        folder_path = dataset_root / folder
        if folder_path.exists():
            img_files = list(folder_path.glob('*.jpg')) + list(folder_path.glob('*.JPG'))
            print(f"✅ {folder}: {len(img_files)} images")
            found_folders.append(folder)
        else:
            print(f"❌ {folder}: Not found")
    
    if not found_folders:
        print("\n❌ No valid image folders found!")
        print("Expected folder structure:")
        print("  dataset_root/")
        print("    RGB/")
        print("    Thermal/")
        print("    RgbAug/")
        print("    ThermalAug/")
        return
    
    print(f"\n✅ Dataset looks good for YOLO11 conversion!")
    print(f"📁 Found {len(found_folders)} valid folders")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="YOLO11 Facial Emotion Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test dataset
  python main_yolo11.py --test-dataset
  
  # Convert dataset to YOLO format
  python main_yolo11.py --convert --modality rgb_only --output-dir yolo_data
  
  # Fuse existing RGB and Thermal datasets
  python main_yolo11.py --fuse --modality early_fusion --output-dir yolo_early_fusion_dataset
  
  # Train YOLO11 model
  python main_yolo11.py --train --modality rgb_only --epochs 100 --batch-size 16
  
  # Evaluate trained model
  python main_yolo11.py --evaluate --model best.pt --test-data yolo_emotion_dataset/test --conf 0.25
  
  # Run inference
  python main_yolo11.py --inference --model best.pt --input image.jpg --conf 0.5
        """
    )
    
    # Commands
    parser.add_argument('--test-dataset', action='store_true', help='Test dataset structure')
    parser.add_argument('--convert', action='store_true', help='Convert dataset to YOLO format')
    parser.add_argument('--fuse', action='store_true', help='Fuse existing RGB and Thermal datasets')
    parser.add_argument('--train', action='store_true', help='Train YOLO11 model')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate trained YOLO11 model')
    parser.add_argument('--inference', action='store_true', help='Run inference on images')
    
    # Configuration
    parser.add_argument('--modality', choices=['rgb_only', 'thermal_only', 'early_fusion', 'late_fusion'], 
                       default='rgb_only', help='Modality type')
    parser.add_argument('--dataset-root', help='Dataset root directory')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, help='Training batch size')
    parser.add_argument('--data', help='Path to YOLO data.yaml file')
    
    # Model and evaluation parameters
    parser.add_argument('--model', help='Path to trained model')
    parser.add_argument('--test-data', help='Path to test dataset for evaluation')
    parser.add_argument('--input', help='Input image path for inference')
    parser.add_argument('--conf', type=float, help='Confidence threshold')
    parser.add_argument('--iou', type=float, help='IoU threshold for NMS')
    
    # Output
    parser.add_argument('--output-dir', help='Output directory')
    
    args = parser.parse_args()
    
    # Create configuration
    config = MultimodalEmotionConfig()
    config.modality = ModalityType(args.modality)
    
    if args.dataset_root:
        config.dataset_root = args.dataset_root
        # Re-detect thermal folders with the new dataset root
        config.thermal_folder = config._detect_thermal_folder(silent=True)
        config.thermal_aug_folder = config._detect_thermal_aug_folder(silent=True)
    
    # Execute commands
    if args.test_dataset:
        test_dataset_command(config)
    elif args.convert:
        convert_dataset_command(config, args)
    elif args.fuse:
        fuse_datasets_command(config, args)
    elif args.train:
        train_command(config, args)
    elif args.evaluate:
        evaluate_command(config, args)
    elif args.inference:
        inference_command(config, args)
    else:
        print("❌ No command specified. Use --help for available commands.")
        print("\nQuick start:")
        print("1. python main_yolo11.py --test-dataset")
        print("2. python main_yolo11.py --convert --modality rgb_only")
        print("3. python main_yolo11.py --train --modality rgb_only --epochs 50")
        print("4. python main_yolo11.py --evaluate --model best.pt --test-data yolo_emotion_dataset/test")


if __name__ == "__main__":
    main()
