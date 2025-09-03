"""
YOLO Dataset Converter for Emotion Detection
===========================================

Converts the emotion classification dataset to YOLO object detection format
by adding face bounding boxes and creating YOLO-style annotations.
"""

import cv2
import numpy as np
from pathlib import Path
import yaml
import shutil
from typing import Dict, List, Tuple
from sklearn.model_selection import train_test_split

from config import MultimodalEmotionConfig

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("Warning: mediapipe not installed. Install it with: pip install mediapipe")
    print("Using fallback face detection (entire image as face)")


class YOLODatasetConverter:
    """Convert emotion classification dataset to YOLO detection format."""
    
    def __init__(self, config: MultimodalEmotionConfig):
        self.config = config
        self.emotion_to_id = {emotion: idx for idx, emotion in enumerate(config.class_names)}
        
        # Initialize face detection
        if MEDIAPIPE_AVAILABLE:
            self.mp_face_detection = mp.solutions.face_detection
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=0, min_detection_confidence=0.3
            )
        else:
            self.face_detection = None
            
    def detect_face_bbox(self, image: np.ndarray) -> Tuple[float, float, float, float]:
        """
        Detect face bounding box in image.
        
        Returns:
            (x_center, y_center, width, height) in normalized coordinates [0,1]
        """
        h, w = image.shape[:2]
        
        if self.face_detection is not None:
            # Use MediaPipe for face detection
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.face_detection.process(rgb_image)
            
            if results.detections:
                # Use the first detected face
                detection = results.detections[0]
                bbox = detection.location_data.relative_bounding_box
                
                # Convert to YOLO format (center_x, center_y, width, height)
                x_center = bbox.xmin + bbox.width / 2
                y_center = bbox.ymin + bbox.height / 2
                width = bbox.width
                height = bbox.height
                
                return x_center, y_center, width, height
        
        # Fallback: use entire image as face region
        # Assume face is in center 80% of image
        return 0.5, 0.5, 0.8, 0.8
    
    def parse_filename(self, filename: str) -> Dict:
        """Parse filename to extract emotion and metadata."""
        name = Path(filename).stem.lower()
        
        # Map various emotion names to standard config names
        emotion_mapping = {
            'angry': 'angry',
            'happy': 'happy', 
            'sad': 'sad',
            'surprised': 'surprised',
            'surprise': 'surprised',
            'neutral': 'neutral',
            'fear': 'fearful',
            'fearful': 'fearful',
            'disgust': 'disgusted',
            'disgusted': 'disgusted'
        }
        
        # Try to find emotion in filename (case-insensitive)
        detected_emotion = None
        for emotion_variant, standard_emotion in emotion_mapping.items():
            if emotion_variant in name:
                detected_emotion = standard_emotion
                break
        
        if not detected_emotion:
            return None
        
        # Try to extract modality and other info
        parts = Path(filename).stem.split('_')
        
        if len(parts) >= 2:
            modality = parts[0].lower()
            
            # Find emotion part index
            emotion_part_idx = None
            for i, part in enumerate(parts):
                if part.lower() in emotion_mapping:
                    emotion_part_idx = i
                    break
            
            # Extract ID and suffix
            if emotion_part_idx is not None and len(parts) > emotion_part_idx + 1:
                image_id = parts[emotion_part_idx + 1]
                suffix = '_'.join(parts[emotion_part_idx + 2:]) if len(parts) > emotion_part_idx + 2 else ''
            else:
                image_id = '0'
                suffix = ''
            
            return {
                'modality': modality,
                'emotion': detected_emotion,
                'raw_emotion': parts[emotion_part_idx] if emotion_part_idx is not None else '',
                'image_id': image_id,
                'suffix': suffix,
                'filename': filename
            }
        else:
            # Fallback: just use detected emotion
            return {
                'modality': 'unknown',
                'emotion': detected_emotion,
                'raw_emotion': '',
                'image_id': '0',
                'suffix': '',
                'filename': filename
            }
        
        return None
    
    def convert_single_folder(self, folder_path: Path, output_dir: Path) -> List[Dict]:
        """Convert a single folder of images to YOLO format."""
        if not folder_path.exists():
            print(f"Warning: Folder {folder_path} does not exist")
            return []
        
        # Support multiple image formats
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif', '*.JPG', '*.JPEG', '*.PNG', '*.BMP', '*.TIFF', '*.TIF']
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(folder_path.glob(ext)))
        converted_files = []
        
        print(f"Converting {len(image_files)} images from {folder_path.name}...")
        
        skipped_files = 0
        processed_count = 0
        
        for i, img_file in enumerate(image_files):
            # Progress indicator for large datasets
            if len(image_files) > 1000 and (i + 1) % 1000 == 0:
                print(f"  Progress: {i + 1}/{len(image_files)} files processed...")
            
            try:
                # Parse filename
                parsed = self.parse_filename(img_file.name)
                if not parsed:
                    skipped_files += 1
                    continue
                    
                if parsed['emotion'] not in self.emotion_to_id:
                    print(f"Warning: Unknown emotion '{parsed['emotion']}' in file {img_file.name}")
                    skipped_files += 1
                    continue
                
                # Load image
                image = cv2.imread(str(img_file))
                if image is None:
                    skipped_files += 1
                    continue
                
                # Detect face
                x_center, y_center, width, height = self.detect_face_bbox(image)
                
                # Get emotion class ID
                emotion_id = self.emotion_to_id[parsed['emotion']]
                
                # Create new filename
                new_img_name = f"{parsed['modality']}_{parsed['emotion']}_{parsed['image_id']}"
                if parsed['suffix']:
                    new_img_name += f"_{parsed['suffix']}"
                new_img_name += ".jpg"
                
                # Copy image to output directory
                output_img_path = output_dir / "images" / new_img_name
                output_img_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(img_file, output_img_path)
                
                # Create YOLO annotation
                annotation_content = f"{emotion_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
                
                # Save annotation file
                output_txt_path = output_dir / "labels" / f"{Path(new_img_name).stem}.txt"
                output_txt_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_txt_path, 'w') as f:
                    f.write(annotation_content)
                
                converted_files.append({
                    'image_path': str(output_img_path.relative_to(output_dir)),
                    'label_path': str(output_txt_path.relative_to(output_dir)),
                    'emotion': parsed['emotion'],
                    'emotion_id': emotion_id,
                    'modality': parsed['modality'],
                    'bbox': [x_center, y_center, width, height]
                })
                
                processed_count += 1
                
            except Exception as e:
                print(f"Error processing {img_file.name}: {str(e)}")
                skipped_files += 1
                continue
        
        print(f"✅ Converted {len(converted_files)} files, skipped {skipped_files} files from {folder_path.name}")
        return converted_files
    
    def create_yolo_dataset(self, output_root: Path, modalities: List[str] = None) -> str:
        """
        Create complete YOLO dataset from emotion classification data.
        
        Args:
            output_root: Root directory for YOLO dataset
            modalities: List of modalities to include (e.g., ['rgb', 'thermal'])
            
        Returns:
            Path to data.yaml configuration file
        """
        if modalities is None:
            modalities = ['rgb']  # Default to RGB only
        
        dataset_root = Path(self.config.dataset_root)
        output_root = Path(output_root)
        output_root.mkdir(parents=True, exist_ok=True)
        
        all_files = []
        
        
        # Process each modality
        for modality in modalities:
            if modality.lower() == 'rgb':
                # Process RGB and RgbAug folders
                for folder_name in ['RGB', 'RgbAug']:
                    folder_path = dataset_root / folder_name
                    if folder_path.exists():
                        files = self.convert_single_folder(folder_path, output_root)
                        all_files.extend(files)
            
            elif modality.lower() == 'thermal':
                # Process Thermal and ThermalAug folders (with automatic detection)
                thermal_folders = []
                
                # Detect thermal folders using config's detection method
                if self.config.thermal_folder:
                    thermal_folders.append(self.config.thermal_folder)
                if self.config.thermal_aug_folder:
                    thermal_folders.append(self.config.thermal_aug_folder)
                
                # Fallback to common names if not detected
                if not thermal_folders:
                    thermal_folders = ['Thermal', 'ThermalAug', 'thermal', 'thermalaug']
                
                for folder_name in thermal_folders:
                    folder_path = dataset_root / folder_name
                    if folder_path.exists():
                        files = self.convert_single_folder(folder_path, output_root)
                        all_files.extend(files)
                        print(f"✅ Processed thermal folder: {folder_name}")
                
                if not any((dataset_root / folder).exists() for folder in thermal_folders):
                    print(f"⚠️ No thermal folders found. Looked for: {thermal_folders}")
        
        print(f"Total converted files: {len(all_files)}")
        
        # Split into train/val/test
        train_files, test_files = train_test_split(
            all_files, 
            test_size=self.config.test_split,
            random_state=42,
            stratify=[f['emotion_id'] for f in all_files]
        )
        
        # Create train/val split (80% train, 20% val from training data)
        train_files, val_files = train_test_split(
            train_files,
            test_size=0.2,
            random_state=42,
            stratify=[f['emotion_id'] for f in train_files]
        )
        
        print(f"Dataset split: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")
        
        # Create split directories and move files
        self._create_split_structure(output_root, train_files, val_files, test_files)
        
        # Create data.yaml configuration
        data_yaml_path = output_root / "data.yaml"
        self._create_data_yaml(data_yaml_path, output_root)
        
        # Calculate and display class distribution
        emotion_counts = {}
        for file_info in all_files:
            emotion = file_info['emotion']
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        print(f"\n✅ Dataset conversion completed!")
        print(f"📁 Output directory: {output_root}")
        print(f"📄 Data configuration: {data_yaml_path}")
        print(f"📊 Total converted files: {len(all_files)}")
        print(f"📂 Data splits: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")
        print(f"\n🎯 Class distribution in converted dataset:")
        for emotion, count in sorted(emotion_counts.items()):
            print(f"   {emotion}: {count}")
        print(f"📈 Total unique emotions: {len(emotion_counts)}")
        
        return str(data_yaml_path)
    
    def _create_split_structure(self, output_root: Path, train_files: List, val_files: List, test_files: List):
        """Create train/val/test directory structure."""
        splits = {
            'train': train_files,
            'val': val_files,
            'test': test_files
        }
        
        for split_name, file_list in splits.items():
            split_dir = output_root / split_name
            split_dir.mkdir(exist_ok=True)
            
            # Create images and labels directories
            (split_dir / 'images').mkdir(exist_ok=True)
            (split_dir / 'labels').mkdir(exist_ok=True)
            
            # Move files to split directories
            for file_info in file_list:
                # Move image
                src_img = output_root / file_info['image_path']
                dst_img = split_dir / 'images' / src_img.name
                if src_img.exists():
                    shutil.move(str(src_img), str(dst_img))
                
                # Move label
                src_label = output_root / file_info['label_path']
                dst_label = split_dir / 'labels' / src_label.name
                if src_label.exists():
                    shutil.move(str(src_label), str(dst_label))
        
        # Remove temporary directories
        temp_dirs = [output_root / 'images', output_root / 'labels']
        for temp_dir in temp_dirs:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
    
    def _create_data_yaml(self, yaml_path: Path, dataset_root: Path):
        """Create YOLO data.yaml configuration file."""
        data_config = {
            'path': str(dataset_root.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': self.config.num_classes,
            'names': self.config.class_names
        }
        
        with open(yaml_path, 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False)
        
        print(f"Created YOLO data configuration: {yaml_path}")


def convert_emotion_dataset_to_yolo(config: MultimodalEmotionConfig, 
                                   output_dir: str = "yolo_dataset",
                                   modalities: List[str] = None) -> str:
    """
    Convert emotion classification dataset to YOLO format.
    
    Args:
        config: Multimodal emotion configuration
        output_dir: Output directory for YOLO dataset
        modalities: List of modalities to include
        
    Returns:
        Path to data.yaml file
    """
    converter = YOLODatasetConverter(config)
    return converter.create_yolo_dataset(Path(output_dir), modalities)


def fuse_existing_datasets(rgb_dataset_path: Path, 
                          thermal_dataset_path: Path,
                          output_dir: str = "yolo_fused_dataset",
                          fusion_type: str = "early_fusion") -> str:
    """
    Fuse existing RGB and Thermal datasets for early/late fusion.
    
    Args:
        rgb_dataset_path: Path to existing RGB dataset
        thermal_dataset_path: Path to existing Thermal dataset  
        output_dir: Output directory for fused dataset
        fusion_type: Type of fusion ('early_fusion' or 'late_fusion')
        
    Returns:
        Path to data.yaml file
    """
    print(f"🔄 Fusing datasets for {fusion_type}...")
    print(f"📁 RGB dataset: {rgb_dataset_path}")
    print(f"📁 Thermal dataset: {thermal_dataset_path}")
    print(f"📁 Output: {output_dir}")
    
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    
    # Create split directories
    splits = ['train', 'val', 'test']
    for split in splits:
        split_dir = output_root / split
        split_dir.mkdir(exist_ok=True)
        (split_dir / 'images').mkdir(exist_ok=True)
        (split_dir / 'labels').mkdir(exist_ok=True)
    
    total_files = 0
    
    # Process each split
    for split in splits:
        print(f"\n📂 Processing {split} split...")
        
        # Get file lists from both datasets
        rgb_images_dir = rgb_dataset_path / split / 'images'
        rgb_labels_dir = rgb_dataset_path / split / 'labels'
        thermal_images_dir = thermal_dataset_path / split / 'images'
        thermal_labels_dir = thermal_dataset_path / split / 'labels'
        
        if not rgb_images_dir.exists() or not thermal_images_dir.exists():
            print(f"⚠️  Skipping {split} - missing directories")
            continue
        
        # Get all image files from RGB dataset
        rgb_images = list(rgb_images_dir.glob('*.jpg')) + list(rgb_images_dir.glob('*.JPG'))
        rgb_images.sort()
        
        # Get all image files from Thermal dataset  
        thermal_images = list(thermal_images_dir.glob('*.jpg')) + list(thermal_images_dir.glob('*.JPG'))
        thermal_images.sort()
        
        print(f"   RGB images: {len(rgb_images)}")
        print(f"   Thermal images: {len(thermal_images)}")
        
        # Ensure we have matching files
        if len(rgb_images) != len(thermal_images):
            print(f"⚠️  Warning: Mismatch in {split} - RGB: {len(rgb_images)}, Thermal: {len(thermal_images)}")
            # Use the smaller count
            min_count = min(len(rgb_images), len(thermal_images))
            rgb_images = rgb_images[:min_count]
            thermal_images = thermal_images[:min_count]
        
        # Create fused dataset
        for i, (rgb_img, thermal_img) in enumerate(zip(rgb_images, thermal_images)):
            # Get corresponding label files
            rgb_label = rgb_labels_dir / f"{rgb_img.stem}.txt"
            thermal_label = thermal_labels_dir / f"{thermal_img.stem}.txt"
            
            if not rgb_label.exists() or not thermal_label.exists():
                print(f"⚠️  Skipping {rgb_img.name} - missing label files")
                continue
            
            # Create fused filename
            fused_name = f"fused_{rgb_img.stem}"
            
            if fusion_type == "early_fusion":
                # For early fusion, we'll create a combined image
                # This is a simplified approach - in practice you might want more sophisticated fusion
                fused_image_path = output_root / split / 'images' / f"{fused_name}.jpg"
                
                # Copy RGB image as the fused image (you can implement more sophisticated fusion here)
                shutil.copy2(rgb_img, fused_image_path)
                
                # Copy RGB labels (assuming same annotations)
                fused_label_path = output_root / split / 'labels' / f"{fused_name}.txt"
                shutil.copy2(rgb_label, fused_label_path)
                
            elif fusion_type == "late_fusion":
                # For late fusion, we'll keep both images with special naming
                # RGB image
                rgb_fused_path = output_root / split / 'images' / f"{fused_name}_rgb.jpg"
                shutil.copy2(rgb_img, rgb_fused_path)
                
                # Thermal image  
                thermal_fused_path = output_root / split / 'images' / f"{fused_name}_thermal.jpg"
                shutil.copy2(thermal_img, thermal_fused_path)
                
                # Copy labels (same for both)
                fused_label_path = output_root / split / 'labels' / f"{fused_name}.txt"
                shutil.copy2(rgb_label, fused_label_path)
            
            total_files += 1
        
        print(f"   ✅ Processed {len(rgb_images)} fused pairs")
    
    # Create data.yaml
    data_yaml_path = output_root / 'data.yaml'
    
    # Read class names from one of the original datasets
    original_yaml = rgb_dataset_path / 'data.yaml'
    if original_yaml.exists():
        with open(original_yaml, 'r') as f:
            original_config = yaml.safe_load(f)
            class_names = original_config.get('names', [])
            num_classes = original_config.get('nc', len(class_names))
    else:
        # Fallback class names
        class_names = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        num_classes = len(class_names)
    
    data_config = {
        'path': str(output_root.absolute()),
        'train': 'train/images',
        'val': 'val/images', 
        'test': 'test/images',
        'nc': num_classes,
        'names': class_names
    }
    
    with open(data_yaml_path, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False)
    
    print(f"\n✅ Dataset fusion completed!")
    print(f"📁 Output directory: {output_root}")
    print(f"📄 Data configuration: {data_yaml_path}")
    print(f"📊 Total fused files: {total_files}")
    print(f"🎯 Fusion type: {fusion_type}")
    
    return str(data_yaml_path)


if __name__ == "__main__":
    # Example usage
    from config import MultimodalEmotionConfig
    
    config = MultimodalEmotionConfig()
    
    print("Converting emotion dataset to YOLO format...")
    data_yaml_path = convert_emotion_dataset_to_yolo(
        config, 
        output_dir="yolo_emotion_dataset",
        modalities=['rgb']  # Convert RGB data only
    )
    
    print(f"YOLO dataset created successfully!")
    print(f"Data configuration file: {data_yaml_path}")
    print("\nTo train YOLO11:")
    print(f"yolo detect train data={data_yaml_path} model=yolo11n.pt epochs=100")
