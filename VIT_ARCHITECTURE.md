# ViT Architecture Documentation

### Table of Contents

- [Overview](#overview)
- [Single Modal ViT (ViTForFER)](#1-single-modal-vit-vitforfer)
- [Early Fusion ViT (EarlyFusionViT)](#2-early-fusion-vit-earlyfusionvit)
  - [Fusion Type](#early-fusion-types)
- [Late Fusion ViT (LateFusionViT)](#3-late-fusion-vit-latefusionvit)
- [How Fine-tuning Works](#how-fine-tuning-works-)
- [Usage / Examples](#usage-examples)

___ 



<a id="overview"></a>
### Overview
A comprehensive implementation of Vision Transformer (ViT) for Facial Expression Recognition (FER) supporting RGB, Thermal, and Combined (RGB+Thermal) modalities with multiple fusion strategies.


<a id="single-modal"></a>
# Single Modal ViT (ViTForFER)
 
- Uses pre-trained google/vit-base-patch16-224-in21k
- Architecture: 12 layers, 768 hidden size, 12 attention heads
- Custom classifier: 2-layer MLP with LayerNorm and dropout
- Input: 224x224 RGB or thermal images
 
<a id="early-fusion"></a>
# Early Fusion ViT (EarlyFusionViT)

- Combines RGB + thermal at input level
- Concat mode: 6-channel input (RGB=3 + Thermal=3)
- Add mode: Element-wise addition of RGB and thermal
- Modifies patch embedding layer for multi-channel input

In Early Fusion, RGB and thermal images are merged before entering the ViT encoder. There are three common fusion types:

<a id="early-fusion-types"></a>
### Fusion Types
   - **Concat**  
     Combines RGB and thermal channels by concatenating them, resulting in a 6-channel input (3 RGB + 3 thermal).  
     The patch embedding layer is modified to handle this multi-channel input, allowing the model to learn joint features from both modalities right from the start.
   

   - **Add**  
     Combines RGB and thermal images by adding their pixel values element-wise, producing a single 3-channel image.  
     The model processes this fused image as normal, blending information from both modalities at the input level.
   

   - **Attention**  
     Uses an attention mechanism to fuse RGB and thermal inputs dynamically at the input token level.  
     This lets the model learn how much weight to assign each modality for every patch, improving fusion by focusing on the most informative features.
   


<a id="late-fusion"></a>
# Late Fusion ViT (LateFusionViT)
 
- Separate ViT encoders for RGB and thermal
- Feature fusion: Combines features before classification
- Prediction fusion: Separate classifiers, then combine predictions

___ 

<a id="how-fine-tuning-works"></a>
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

<a id="usage-examples"></a>
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

<a id="fine-tuning-process"></a>
##  Fine-tuning Process

1. Load pre-trained weights from HuggingFace
2. Replace classifier head with custom 7-class head
3. Optionally freeze backbone for initial training
4. Use different learning rates for backbone vs classifier
5. Apply cosine scheduler with warmup
6. Gradually unfreeze layers if needed
