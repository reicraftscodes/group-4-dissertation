# ViT Architecture Documentation

ViT Architecture Overview
 
###  1. Single Modal ViT (ViTForFER)
 
- Uses pre-trained google/vit-base-patch16-224-in21k
- Architecture: 12 layers, 768 hidden size, 12 attention heads
- Custom classifier: 2-layer MLP with LayerNorm and dropout
- Input: 224x224 RGB or thermal images
 
### 2. Early Fusion ViT (EarlyFusionViT)
 
- Combines RGB + thermal at input level
- Concat mode: 6-channel input (RGB=3 + Thermal=3)
- Add mode: Element-wise addition of RGB and thermal
- Modifies patch embedding layer for multi-channel input
 
### 3. Late Fusion ViT (LateFusionViT)
 
- Separate ViT encoders for RGB and thermal
- Feature fusion: Combines features before classification
- Prediction fusion: Separate classifiers, then combine predictions

___ 

##  How Fine-tuning Works
 
Fine-tuning Strategies
 
1. Classifier-only fine-tuning (freeze backbone)
model = ViTForFER(freeze_backbone=True)
 
2. Full fine-tuning (train entire model)
model = ViTForFER(freeze_backbone=False)
 
3. Progressive unfreezing
model = ViTForFER(freeze_backbone=True)

4. Train classifier first, then:
model.unfreeze_backbone()
 
## Key Fine-tuning Features
Custom Classifier Head:

```
nn.Sequential(
nn.LayerNorm(768), # Normalize features
nn.Dropout(0.1), # Prevent overfitting
nn.Linear(768, 384), # Hidden layer
nn.GELU(), # Activation
nn.Dropout(0.05), # More dropout
nn.Linear(384, 7) # Output 7 emotions
)
```
 
Different Learning Rates:
- Backbone: learning_rate * 0.1 (lower for pre-trained weights)
- Classifier: learning_rate (higher for new weights)
 
Memory Optimization:
- Gradient checkpointing to reduce memory usage
- Supports freezing backbone to save compute
___ 

## Usage/Examples
 
1. Single RGB Model
 
   Create model
   ```
   model = create_multimodal_vit_model(mode='rgb')
    ```
   For fine-tuning
   ```
   model = ViTForFER(
   model_name="google/vit-base-patch16-224-in21k",
   num_classes=7,
   freeze_backbone=False, # Full fine-tuning
   dropout_rate=0.1
   )
   ```
    
2. Multi-modal Model
 
   Early fusion (concat RGB + thermal)
   ```
   model = create_multimodal_vit_model(
   mode='combined',
   fusion_strategy='early',
   fusion_type='concat'
   )
   ```
 
   Late fusion (separate encoders)
   ```
   model = create_multimodal_vit_model(
   mode='combined',
   fusion_strategy='late',
   fusion_type='attention',
   fusion_layer='feature'
   )
   ```
 
3. Smart Optimizer Setup
    ```
   optimizer, scheduler = get_optimizer_and_scheduler(
   model=model,
   learning_rate=5e-5,
   warmup_steps=1000,
   num_training_steps=10000
   )
   ```

##  Fine-tuning Process

1. Load pre-trained weights from HuggingFace
2. Replace classifier head with custom 7-class head
3. Optionally freeze backbone for initial training
4. Use different learning rates for backbone vs classifier
5. Apply cosine scheduler with warmup
6. Gradually unfreeze layers if needed
