"""
Dataset parsing and loading for YOLO11 Multimodal Emotion Detection
==================================================================

This module handles dataset parsing, loading, and preprocessing for the emotion detection system.
"""

import torch
import torch.utils.data
import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
from sklearn.model_selection import train_test_split

from config import MultimodalEmotionConfig, ModalityType, FusionType


class DatasetParser:
    """Parse the custom dataset structure and extract emotion labels."""
    
    def __init__(self, config: MultimodalEmotionConfig):
        self.config = config
        self.dataset_root = Path(config.dataset_root)
        
        # Emotion mapping from filename to class index
        # Support both 'fear'/'fearful' and 'disgust'/'disgusted'
        self.emotion_mapping = {
            'happy': 0,
            'sad': 1, 
            'angry': 2,
            'surprised': 3,
            'fearful': 4,
            'fear': 4,  # Alternative name
            'disgusted': 5,
            'disgust': 5,  # Alternative name  
            'neutral': 6
        }
        
        # Reverse mapping for class names
        self.class_mapping = {v: k for k, v in self.emotion_mapping.items()}
    
    def parse_filename(self, filename: str) -> Dict:
        """Parse filename to extract emotion and metadata."""
        # Remove extension
        name = Path(filename).stem
        
        # Split by underscore
        parts = name.split('_')
        
        if len(parts) >= 3:
            modality = parts[0]  # T for thermal, R for RGB
            emotion = parts[1]   # emotion name
            image_id = parts[2]  # image ID
            suffix = '_'.join(parts[3:]) if len(parts) > 3 else ''  # remaining parts
            
            return {
                'modality': modality,
                'emotion': emotion.lower(),
                'image_id': image_id,
                'suffix': suffix,
                'filename': filename
            }
        
        return None
    
    def get_emotion_class(self, emotion: str) -> int:
        """Get class index for emotion."""
        return self.emotion_mapping.get(emotion.lower(), 0)
    
    def load_dataset_files(self, modality_type: ModalityType) -> Dict:
        """Load dataset files based on modality type."""
        files = {
            'train': [],
            'test': []
        }
        
        if modality_type == ModalityType.RGB_ONLY:
            files = self._load_rgb_only_files()
        elif modality_type == ModalityType.THERMAL_ONLY:
            files = self._load_thermal_only_files()
        elif modality_type in [ModalityType.EARLY_FUSION, ModalityType.LATE_FUSION]:
            files = self._load_multimodal_files()
        
        return files
    
    def _load_rgb_only_files(self) -> Dict:
        """Load RGB-only dataset files from both original and augmented folders."""
        files = {'train': [], 'test': []}
        all_files = []
        
        print(f"Loading RGB-only data from:")
        
        # Load original RGB files
        rgb_path = self.dataset_root / self.config.rgb_folder
        if rgb_path.exists():
            rgb_files = list(rgb_path.glob('*.jpg')) + list(rgb_path.glob('*.JPG'))
            print(f"  - {self.config.rgb_folder}: {len(rgb_files)} images")
            for file_path in rgb_files:
                parsed = self.parse_filename(file_path.name)
                if parsed and parsed['emotion'] in self.emotion_mapping:
                    all_files.append({
                        'file_path': file_path,
                        'emotion': parsed['emotion'],
                        'class_id': self.get_emotion_class(parsed['emotion']),
                        'source': 'original',
                        'modality': 'rgb'
                    })
        
        # Load augmented RGB files
        rgb_aug_path = self.dataset_root / self.config.rgb_aug_folder
        if rgb_aug_path.exists():
            rgb_aug_files = list(rgb_aug_path.glob('*.jpg')) + list(rgb_aug_path.glob('*.JPG'))
            print(f"  - {self.config.rgb_aug_folder}: {len(rgb_aug_files)} images")
            for file_path in rgb_aug_files:
                parsed = self.parse_filename(file_path.name)
                if parsed and parsed['emotion'] in self.emotion_mapping:
                    all_files.append({
                        'file_path': file_path,
                        'emotion': parsed['emotion'],
                        'class_id': self.get_emotion_class(parsed['emotion']),
                        'source': 'augmented',
                        'modality': 'rgb'
                    })
        
        # Split data into train/test (80/20) - NO validation split
        if all_files:
            train_files, test_files = train_test_split(
                all_files,
                test_size=self.config.test_split,  # 0.2 = 20% for test, 80% for train
                random_state=42,
                stratify=[f['class_id'] for f in all_files]
            )
            
            files['train'] = train_files
            files['test'] = test_files
        
        return files
    
    def _load_thermal_only_files(self) -> Dict:
        """Load thermal-only dataset files from both original and augmented folders."""
        files = {'train': [], 'test': []}
        all_files = []
        
        # Check if thermal folder exists
        if self.config.thermal_folder is None:
            print("Warning: No thermal folder found. Skipping thermal-only loading.")
            return files
        
        print(f"Loading thermal-only data from:")
            
        # Load original thermal files
        thermal_path = self.dataset_root / self.config.thermal_folder
        if thermal_path.exists():
            thermal_files = list(thermal_path.glob('*.jpg')) + list(thermal_path.glob('*.JPG'))
            print(f"  - {self.config.thermal_folder}: {len(thermal_files)} images")
            for file_path in thermal_files:
                parsed = self.parse_filename(file_path.name)
                if parsed and parsed['emotion'] in self.emotion_mapping:
                    all_files.append({
                        'file_path': file_path,
                        'emotion': parsed['emotion'],
                        'class_id': self.get_emotion_class(parsed['emotion']),
                        'source': 'original',
                        'modality': 'thermal'
                    })
        
        # Load augmented thermal files
        if self.config.thermal_aug_folder is not None:
            thermal_aug_path = self.dataset_root / self.config.thermal_aug_folder
            if thermal_aug_path.exists():
                thermal_aug_files = list(thermal_aug_path.glob('*.jpg')) + list(thermal_aug_path.glob('*.JPG'))
                print(f"  - {self.config.thermal_aug_folder}: {len(thermal_aug_files)} images")
                for file_path in thermal_aug_files:
                    parsed = self.parse_filename(file_path.name)
                    if parsed and parsed['emotion'] in self.emotion_mapping:
                        all_files.append({
                            'file_path': file_path,
                            'emotion': parsed['emotion'],
                            'class_id': self.get_emotion_class(parsed['emotion']),
                            'source': 'augmented',
                            'modality': 'thermal'
                        })
        
        # Split data into train/test (80/20) - NO validation split
        if all_files:
            train_files, test_files = train_test_split(
                all_files,
                test_size=self.config.test_split,  # 0.2 = 20% for test, 80% for train
                random_state=42,
                stratify=[f['class_id'] for f in all_files]
            )
            
            files['train'] = train_files
            files['test'] = test_files
        
        return files
    
    def _load_multimodal_files(self) -> Dict:
        """Load multimodal (RGB + Thermal) dataset files."""
        files = {'train': [], 'test': []}
        
        # Check if thermal folder exists
        if self.config.thermal_folder is None:
            print("Warning: No thermal folder found. Multimodal fusion requires both RGB and thermal data.")
            return files
        
        print(f"Loading multimodal (RGB + Thermal) data from:")
        
        # For combined approaches, we need both RGB and thermal pairs from original and augmented folders
        rgb_files = {}
        thermal_files = {}
        
        # Load original RGB files
        rgb_path = self.dataset_root / self.config.rgb_folder
        if rgb_path.exists():
            rgb_files_list = list(rgb_path.glob('*.jpg'))
            print(f"  - {self.config.rgb_folder}: {len(rgb_files_list)} images")
            for file_path in rgb_files_list:
                parsed = self.parse_filename(file_path.name)
                if parsed and parsed['emotion'] in self.emotion_mapping:
                    key = f"{parsed['emotion']}_{parsed['image_id']}_original"
                    rgb_files[key] = {
                        'file_path': file_path,
                        'emotion': parsed['emotion'],
                        'class_id': self.get_emotion_class(parsed['emotion']),
                        'parsed': parsed,
                        'source': 'original'
                    }
        
        # Load augmented RGB files
        rgb_aug_path = self.dataset_root / self.config.rgb_aug_folder
        if rgb_aug_path.exists():
            rgb_aug_files_list = list(rgb_aug_path.glob('*.jpg'))
            print(f"  - {self.config.rgb_aug_folder}: {len(rgb_aug_files_list)} images")
            for file_path in rgb_aug_files_list:
                parsed = self.parse_filename(file_path.name)
                if parsed and parsed['emotion'] in self.emotion_mapping:
                    key = f"{parsed['emotion']}_{parsed['image_id']}_augmented"
                    rgb_files[key] = {
                        'file_path': file_path,
                        'emotion': parsed['emotion'],
                        'class_id': self.get_emotion_class(parsed['emotion']),
                        'parsed': parsed,
                        'source': 'augmented'
                    }
        
        # Load original thermal files
        thermal_path = self.dataset_root / self.config.thermal_folder
        if thermal_path.exists():
            thermal_files_list = list(thermal_path.glob('*.jpg'))
            print(f"  - {self.config.thermal_folder}: {len(thermal_files_list)} images")
            for file_path in thermal_files_list:
                parsed = self.parse_filename(file_path.name)
                if parsed and parsed['emotion'] in self.emotion_mapping:
                    key = f"{parsed['emotion']}_{parsed['image_id']}_original"
                    thermal_files[key] = {
                        'file_path': file_path,
                        'emotion': parsed['emotion'],
                        'class_id': self.get_emotion_class(parsed['emotion']),
                        'parsed': parsed,
                        'source': 'original'
                    }
        
        # Load augmented thermal files
        if self.config.thermal_aug_folder is not None:
            thermal_aug_path = self.dataset_root / self.config.thermal_aug_folder
            if thermal_aug_path.exists():
                thermal_aug_files_list = list(thermal_aug_path.glob('*.jpg'))
                print(f"  - {self.config.thermal_aug_folder}: {len(thermal_aug_files_list)} images")
                for file_path in thermal_aug_files_list:
                    parsed = self.parse_filename(file_path.name)
                    if parsed and parsed['emotion'] in self.emotion_mapping:
                        key = f"{parsed['emotion']}_{parsed['image_id']}_augmented"
                        thermal_files[key] = {
                            'file_path': file_path,
                            'emotion': parsed['emotion'],
                            'class_id': self.get_emotion_class(parsed['emotion']),
                            'parsed': parsed,
                            'source': 'augmented'
                        }
        
        # Find matching pairs
        common_keys = set(rgb_files.keys()) & set(thermal_files.keys())
        all_pairs = []
        
        for key in common_keys:
            rgb_data = rgb_files[key]
            thermal_data = thermal_files[key]
            
            if rgb_data['emotion'] == thermal_data['emotion']:
                all_pairs.append({
                    'rgb_file': rgb_data['file_path'],
                    'thermal_file': thermal_data['file_path'],
                    'emotion': rgb_data['emotion'],
                    'class_id': rgb_data['class_id'],
                    'image_id': key,
                    'source': rgb_data['source']  # Will be 'original' or 'augmented'
                })
        
        # Split data into train/test (80/20) - NO validation split
        if all_pairs:
            train_pairs, test_pairs = train_test_split(
                all_pairs,
                test_size=self.config.test_split,  # 0.2 = 20% for test, 80% for train
                random_state=42,
                stratify=[f['class_id'] for f in all_pairs]
            )
            
            files['train'] = train_pairs
            files['test'] = test_pairs
        
        return files


class AdaptedMultimodalEmotionDataset(torch.utils.data.Dataset):
    """Custom dataset for multimodal facial emotion detection - adapted for your structure."""
    
    def __init__(self, config: MultimodalEmotionConfig, mode='train'):
        self.config = config
        self.mode = mode
        self.modality = config.modality
        
        # Initialize parser
        self.parser = DatasetParser(config)
        
        # Load dataset files
        self.dataset_files = self.parser.load_dataset_files(self.modality)
        self.files = self.dataset_files[mode]
        
        print(f"Loaded {len(self.files)} {mode} samples for {self.modality.value}")
        print(f"Data split ratios: train={self.config.train_split:.1%}, test={self.config.test_split:.1%}")
        print("Note: Using combined original + augmented data, then splitting into train/test (NO validation split)")
        
        # Print class distribution and data source
        self._print_dataset_stats()
    
    def _print_dataset_stats(self):
        """Print dataset statistics."""
        class_counts = {}
        source_counts = {}
        for file_data in self.files:
            emotion = file_data['emotion']
            source = file_data.get('source', 'unknown')
            class_counts[emotion] = class_counts.get(emotion, 0) + 1
            source_counts[source] = source_counts.get(source, 0) + 1
        
        print(f"Class distribution for {self.mode}:")
        for emotion, count in class_counts.items():
            print(f"  {emotion}: {count}")
        
        print(f"Data source distribution for {self.mode}:")
        for source, count in source_counts.items():
            print(f"  {source}: {count}")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file_data = self.files[idx]
        
        if self.modality in [ModalityType.EARLY_FUSION, ModalityType.LATE_FUSION]:
            # Combined approach
            return self._load_combined_sample(file_data)
        else:
            # Single modality
            return self._load_single_sample(file_data)
    
    def _load_single_sample(self, file_data):
        """Load single modality sample."""
        file_path = file_data['file_path']
        
        # Load image
        img = cv2.imread(str(file_path))
        if img is None:
            raise ValueError(f"Could not load image: {file_path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Create dummy bounding box (full image) for emotion classification
        h, w = img.shape[:2]
        bbox = [0, 0, w, h]  # [x, y, width, height]
        
        # Create label format: [class_id, x_center, y_center, width, height]
        x_center = bbox[0] + bbox[2] / 2
        y_center = bbox[1] + bbox[3] / 2
        labels = np.array([[file_data['class_id'], x_center, y_center, bbox[2], bbox[3]]])
        
        # Preprocess
        img_processed, labels_processed = self.preprocess_image(img, labels)
        
        return {
            'image': torch.from_numpy(img_processed).float(),
            'labels': torch.from_numpy(labels_processed).float(),
            'emotion': file_data['emotion'],
            'class_id': file_data['class_id'],
            # Include the original file path for later analysis/visualization
            'file_path': str(file_path),
            # Unified key for downstream evaluation and misclassification grids
            'image_path': str(file_path)
        }
    
    def _load_combined_sample(self, file_data):
        """Load combined RGB + thermal sample."""
        rgb_path = file_data['rgb_file']
        thermal_path = file_data['thermal_file']
        
        # Load RGB image
        rgb_img = cv2.imread(str(rgb_path))
        if rgb_img is None:
            raise ValueError(f"Could not load RGB image: {rgb_path}")
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        
        # Load thermal image
        thermal_img = cv2.imread(str(thermal_path))
        if thermal_img is None:
            raise ValueError(f"Could not load thermal image: {thermal_path}")
        
        # Convert thermal to 3-channel if needed
        if len(thermal_img.shape) == 2:
            thermal_img = cv2.cvtColor(thermal_img, cv2.COLOR_GRAY2RGB)
        else:
            thermal_img = cv2.cvtColor(thermal_img, cv2.COLOR_BGR2RGB)
        
        # Create labels
        h, w = rgb_img.shape[:2]
        bbox = [0, 0, w, h]
        x_center = bbox[0] + bbox[2] / 2
        y_center = bbox[1] + bbox[3] / 2
        labels = np.array([[file_data['class_id'], x_center, y_center, bbox[2], bbox[3]]])
        
        if self.modality == ModalityType.EARLY_FUSION:
            # Early fusion: combine images before model input
            combined_img = self._early_fusion(rgb_img, thermal_img)
            img_processed, labels_processed = self.preprocess_image(combined_img, labels)
            
            return {
                'rgb_image': torch.from_numpy(img_processed).float(),
                'thermal_image': torch.from_numpy(img_processed).float(),
                'combined_image': torch.from_numpy(img_processed).float(),
                'labels': torch.from_numpy(labels_processed).float(),
                'emotion': file_data['emotion'],
                'class_id': file_data['class_id'],
                'rgb_path': str(rgb_path),
                'thermal_path': str(thermal_path),
                # For misclassification visualizations, use the RGB image path as the primary
                'image_path': str(rgb_path)
            }
        else:  # LATE_FUSION
            # Late fusion: process images separately
            rgb_processed, _ = self.preprocess_image(rgb_img, labels)
            thermal_processed, _ = self.preprocess_image(thermal_img, labels)
            
            return {
                'rgb_image': torch.from_numpy(rgb_processed).float(),
                'thermal_image': torch.from_numpy(thermal_processed).float(),
                'combined_image': None,  # Will be fused later
                'labels': torch.from_numpy(labels).float(),
                'emotion': file_data['emotion'],
                'class_id': file_data['class_id'],
                'rgb_path': str(rgb_path),
                'thermal_path': str(thermal_path),
                # Use RGB path as representative image for misclassification grids
                'image_path': str(rgb_path)
            }
    
    def _early_fusion(self, rgb_img, thermal_img):
        """Early fusion: combine RGB and thermal images."""
        # Method 1: Simple concatenation along channel dimension
        if self.config.fusion_type == FusionType.CONCATENATE:
            # Resize thermal to match RGB dimensions
            thermal_resized = cv2.resize(thermal_img, (rgb_img.shape[1], rgb_img.shape[0]))
            # Concatenate along channel dimension (RGB + Thermal = 6 channels)
            combined = np.concatenate([rgb_img, thermal_resized], axis=2)
            # Convert back to 3 channels by taking mean of each modality
            combined = np.mean(combined.reshape(-1, 2, 3), axis=1).reshape(rgb_img.shape)
            return combined
        
        # Method 2: Weighted addition
        elif self.config.fusion_type == FusionType.ADD:
            thermal_resized = cv2.resize(thermal_img, (rgb_img.shape[1], rgb_img.shape[0]))
            # Weighted combination (0.7 RGB + 0.3 Thermal)
            combined = 0.7 * rgb_img + 0.3 * thermal_resized
            return combined.astype(np.uint8)
        
        # Method 3: Element-wise multiplication
        elif self.config.fusion_type == FusionType.MULTIPLY:
            thermal_resized = cv2.resize(thermal_img, (rgb_img.shape[1], rgb_img.shape[0]))
            # Normalize and multiply
            rgb_norm = rgb_img.astype(np.float32) / 255.0
            thermal_norm = thermal_resized.astype(np.float32) / 255.0
            combined = (rgb_norm * thermal_norm * 255).astype(np.uint8)
            return combined
        
        else:
            return rgb_img  # Default fallback
    
    def preprocess_image(self, img: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess image and labels for YOLO11."""
        h, w = img.shape[:2]
        
        # Resize image
        scale = min(self.config.input_size / w, self.config.input_size / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        img_resized = cv2.resize(img, (new_w, new_h))
        
        # Create padded image
        img_padded = np.full((self.config.input_size, self.config.input_size, 3), 114, dtype=np.uint8)
        img_padded[:new_h, :new_w] = img_resized
        
        # Adjust labels
        if len(labels) > 0:
            labels[:, 1:] *= scale  # Scale coordinates
            labels[:, 1] += (self.config.input_size - new_w) / 2  # Add padding offset
            labels[:, 2] += (self.config.input_size - new_h) / 2
        
        # Normalize image
        img_padded = img_padded.astype(np.float32) / 255.0
        img_padded = img_padded.transpose(2, 0, 1)  # HWC to CHW
        
        return img_padded, labels