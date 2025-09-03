"""
Configuration classes for YOLO11 Multimodal Emotion Detection
============================================================

This module contains all configuration classes and enums for the emotion detection system.
"""

from enum import Enum
from pathlib import Path


class ModalityType(Enum):
    """Enum for different modality types."""
    RGB_ONLY = "rgb_only"
    THERMAL_ONLY = "thermal_only"
    EARLY_FUSION = "early_fusion"  # RGB + Thermal combined early
    LATE_FUSION = "late_fusion"    # RGB + Thermal combined late


class FusionType(Enum):
    """Enum for fusion strategies."""
    CONCATENATE = "concatenate"
    ADD = "add"
    MULTIPLY = "multiply"
    ATTENTION = "attention"


class MultimodalEmotionConfig:
    """Configuration class for multimodal emotion detection model."""
    
    def __init__(self):
        # Model parameters
        self.model_size = 'n'  # 'n', 's', 'm', 'l', 'x'
        #  # 7 emotion classes
        self.reg_max = 16      # DFL bins
        self.stride = [8, 16, 32]  # Multi-scale detection
        
        # Modality configuration
        self.modality = ModalityType.RGB_ONLY  # Default to RGB-only
        self.fusion_type = FusionType.CONCATENATE  # For combined approaches
        self.use_attention = True  # Use attention mechanism for fusion
        
        # Training parameters
        self.batch_size = 16
        self.epochs = 100
        self.learning_rate = 0.01
        self.weight_decay = 0.0005
        
        # Detection parameters
        self.conf_threshold = 0.25
        self.iou_threshold = 0.45
        self.max_detections = 300
        
        # Data parameters
        self.input_size = 640
        self.class_names = ['happy', 'sad', 'angry', 'surprised', 'fearful', 'disgusted', 'neutral']
        self.num_classes = len(self.class_names)

        
        # Dataset paths - update for your environment
        # Default for Google Colab: set to a typical Drive location. You can override this
        # via the command line or configuration file if your dataset lives elsewhere.
        self.dataset_root = "/content/drive/MyDrive/Data"
        self.rgb_folder = "RGB"
        self.rgb_aug_folder = "RgbAug"  # Augmented RGB folder (note: lowercase 'g')
        
        # Only detect thermal folders if dataset_root exists
        self.thermal_folder = None
        self.thermal_aug_folder = None
        if Path(self.dataset_root).exists():
            self.thermal_folder = self._detect_thermal_folder(silent=True)
            self.thermal_aug_folder = self._detect_thermal_aug_folder(silent=True)
        
        # Output paths
        self.output_path = "outputs/multimodal_emotion_detection"
        self.model_save_path = "models"
        self.logs_path = "logs"
        
        # Fusion parameters
        self.fusion_channels = 256  # Number of channels after fusion
        self.attention_heads = 8     # Number of attention heads
        
        # Augmentation
        self.augmentation = True
        self.mixup = 0.1
        self.mosaic = 0.5
        
        # Data split - Single dataset split into train/test (80/20)
        self.train_split = 0.8  # 80% for training
        self.test_split = 0.2   # 20% for testing (no validation split)
        
        # Create output directories
        self._create_directories()
    
    def _detect_thermal_folder(self, silent=False):
        """Detect thermal folder with multiple possible names."""
        if not hasattr(self, 'dataset_root'):
            return None
            
        dataset_root = Path(self.dataset_root)
        if not dataset_root.exists():
            return None
        
        # Possible thermal folder names
        thermal_names = [
            "Thermal",
            "thermal", 
            "Thermal_Images",
            "thermal_images",
            "IR",
            "infrared",
            "Infrared"
        ]
        
        for name in thermal_names:
            thermal_path = dataset_root / name
            if thermal_path.exists() and thermal_path.is_dir():
                # Check if it contains image files
                image_files = list(thermal_path.glob("*.jpg")) + list(thermal_path.glob("*.JPG"))
                if image_files:
                    if not silent:
                        print(f"Detected thermal folder: {name} ({len(image_files)} images)")
                    return name
        
        if not silent:
            print("No thermal folder detected - only RGB modality will be available")
        return None
    
    def _detect_thermal_aug_folder(self, silent=False):
        """Detect thermal augmented folder with multiple possible names."""
        if not hasattr(self, 'dataset_root'):
            return None
            
        dataset_root = Path(self.dataset_root)
        if not dataset_root.exists():
            return None
        
        # Possible thermal augmented folder names
        thermal_aug_names = [
            "ThermalAug",
            "thermalaug", 
            "Thermal_Aug",
            "thermal_aug",
            "ThermalAugmented",
            "thermal_augmented"
        ]
        
        for name in thermal_aug_names:
            thermal_aug_path = dataset_root / name
            if thermal_aug_path.exists() and thermal_aug_path.is_dir():
                # Check if it contains image files
                image_files = list(thermal_aug_path.glob("*.jpg")) + list(thermal_aug_path.glob("*.JPG"))
                if image_files:
                    if not silent:
                        print(f"Detected thermal augmented folder: {name} ({len(image_files)} images)")
                    return name
        
        if not silent:
            print("No thermal augmented folder detected")
        return None
    
    def _create_directories(self):
        """Create necessary output directories."""
        for path in [self.output_path, self.model_save_path, self.logs_path]:
            Path(path).mkdir(parents=True, exist_ok=True)
    
    def to_dict(self):
        """Convert config to dictionary for saving."""
        return {
            'model_size': self.model_size,
            'num_classes': self.num_classes,
            'modality': self.modality.value,
            'fusion_type': self.fusion_type.value,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'learning_rate': self.learning_rate,
            'input_size': self.input_size,
            'class_names': self.class_names,
            'dataset_root': self.dataset_root
        }
    
    def save_config(self, filepath: str):
        """Save configuration to file."""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_config(cls, filepath: str):
        """Load configuration from file."""
        import json
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        config = cls()
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        return config