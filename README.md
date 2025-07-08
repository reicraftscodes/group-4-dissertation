# Multimodal Vision Transformer for Facial Expression Recognition

A comprehensive implementation of Vision Transformer (ViT) for Facial Expression Recognition (FER) supporting RGB, Thermal, and Combined (RGB+Thermal) modalities with multiple fusion strategies.

## Features

- **Multi-modal Support**: RGB-only, Thermal-only, and Combined (RGB+Thermal) modes
- **Fusion Strategies**: Early fusion (input-level) and Late fusion (feature/prediction-level)
- **Fusion Types**: Concatenation, Addition, and Attention-based fusion
- **Pre-trained Models**: Built on HuggingFace ViT with fine-tuning capabilities
- **Comprehensive Evaluation**: Detailed metrics, confusion matrices, and visualizations
- **Early Stopping**: Prevents overfitting with configurable patience
- **Experiment Tracking**: TensorBoard and WandB support
- **Class Imbalance Handling**: Automatic class weight calculation

## Dataset

### Supported Emotions
- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprised

### Data Structure
Your dataset should be organized as follows:
```
Data/
    RGB/
        R_Angry_1_KTFE.jpg
        R_Angry_2_KTFE.jpg
        R_Disgust_1_KTFE.jpg
        ...
    Thermal/
        T_Angry_1_KTFE.jpg
        T_Angry_2_KTFE.jpg
        T_Disgust_1_KTFE.jpg
        ...
    augmented/ (optional)
        RGB/
            aug_R_Angry_1_KTFE.jpg
            ...
        Thermal/
            aug_T_Angry_1_KTFE.jpg
            ...
```

**File Naming Convention**: `{modality}_{emotion}_{id}_{suffix}.jpg`
- `modality`: R (RGB) or T (Thermal)
- `emotion`: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprised
- `id`: Unique identifier
- `suffix`: Additional identifier (e.g., KTFE)

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)

### Setup Environment
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Required Dependencies
```bash
pip install torch torchvision torchaudio
pip install transformers
pip install scikit-learn
pip install matplotlib seaborn
pip install tensorboard
pip install wandb  # Optional
pip install tqdm
pip install pillow
```

## Project Structure

```
train.py              # Main training script
model.py              # Model architectures (ViT, Early/Late Fusion)
dataset.py            # Dataset loading and preprocessing
utils.py              # Utilities (metrics, early stopping, etc.)
evaluate.py           # Model evaluation and visualization
quick_inference.py    # Single image inference
requirements.txt      # Dependencies
README.md 
experiments/          # all result training
Data/                 # download and unzip from https://drive.google.com/drive/folders/1hBpFWxtlbHpPX9DnQ5x9j9-gvyaLznuz?usp=sharing
```

## Training

### Configuration
Edit the `config` dictionary in `train.py` to customize training parameters:

```python
config = {
    'data_dir': '/path/to/your/Data',
    'mode': 'rgb',  # 'rgb', 'thermal', or 'combined'
    'fusion_strategy': 'early',  # 'early' or 'late' (for combined mode)
    'fusion_type': 'concat',  # 'concat', 'add', or 'attention'
    'fusion_layer': 'feature',  # 'feature' or 'prediction' (for late fusion)
    'batch_size': 32,
    'num_epochs': 30,
    'learning_rate': 5e-5,
    # ... other parameters
}
```

### Training Commands

#### RGB-only Mode

#### Thermal-only Mode



### Fusion Strategies

#### Early Fusion
- **Input-level fusion**: Concatenates RGB and Thermal images
- **Architecture**: Single ViT encoder processes fused input
- **Best for**: Learning joint representations from the start

#### Late Fusion
- **Feature/Prediction-level fusion**: Separate ViT encoders for each modality
- **Architecture**: Two ViT encoders + fusion layer
- **Best for**: Learning modality-specific features first

### Fusion Types
- **Concatenation**: `torch.cat([rgb_features, thermal_features], dim=1)`
- **Addition**: `rgb_features + thermal_features`
- **Attention**: Multi-head attention mechanism for feature weighting


## Configuration Options

### Model Parameters
- `model_name`: Pre-trained ViT model (`google/vit-base-patch16-224-in21k`)
- `num_classes`: Number of emotion classes (7)
- `dropout_rate`: Dropout rate for classifier (0.1)
- `freeze_backbone`: Whether to freeze ViT backbone (False)

### Training Parameters
- `batch_size`: Training batch size (32)
- `num_epochs`: Maximum training epochs (30)
- `learning_rate`: Learning rate (5e-5)
- `weight_decay`: Weight decay (0.01)
- `warmup_steps`: Warmup steps for scheduler (500)

### Data Parameters
- `image_size`: Input image size (224)
- `use_augmented`: Use augmented data (True)
- `use_class_weights`: Handle class imbalance (True)
- `val_split`: Validation split ratio (0.2)

### Early Stopping
- `early_stopping_patience`: Patience epochs (10)
- `early_stopping_min_delta`: Minimum improvement (0.001)

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `batch_size` in config
   - Enable `use_gradient_checkpointing`

2. **Poor Performance**
   - Check class balance with `use_class_weights: True`
   - Adjust learning rate
   - Try different fusion strategies

3. **Slow Training**
   - Increase `num_workers` in data loading
   - Use smaller image size (e.g., 224 � 192)

### Debug Commands
```bash
# Check dataset statistics
python -c "from dataset import analyze_dataset; analyze_dataset('Data')"

# Test model creation
python model.py

# Check GPU usage
nvidia-smi
```








