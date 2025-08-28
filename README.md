# Data Processing
The data processing code implementation is available in the Jupyter Notebook file ```Data_Processing.ipynb```

## Datasets Information
Two datasets were used in this project - IRIS and KTFEv.2

**- IRIS dataset** 
  - Contains 4,228 pairs of thermal and RGB images. 
  - Comprised of 30 individuals. 
  - Each individual shows 3 emotions: surprised, laughing, angry. 
  - Includes 5 illumination conditions with neutral facial expressions. 
  - Available from [OTCBVS](https://vcipl-okstate.org/pbvs/bench/).

**- KTFEv.2 dataset**
  - Contains 3,190 pairs of thermal and RGB images. 
  - Comprised of 30 individuals. 
  - Includes 7 emotions: surprise, happiness, anger, neutral, sadness, fear, disgust. 
  - Available from [Kaggle](https://www.kaggle.com/datasets/nafisaislamrifa/facial-emotions-thermal-visual/data).
  
**Dataset Size**
- RGB raw images: ```4,139```
- Thermal raw images: ```4,139```


## Data Structure
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
    augmented/ 
        RGB/
            aug_R_Angry_1_KTFE.jpg
            ...
        Thermal/
            aug_T_Angry_1_KTFE.jpg
            ...
```

File naming convention
```
`{modality}_{emotion}_{id}_{suffix}.jpg`
- `modality`: R (RGB) or T (Thermal)
- `emotion`: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprised
- `id`: Unique identifier
- `suffix`: Additional identifier (e.g., KTFE)
```
- Each RGB image has a corresponding thermal image with the same unique number. 
- Augmented images use the aug_ suffix.

## Data Processing
- Only images containing emotions are used : Angry, Disgust,Fear, Happy, Neutral, Sad, Surprised
- Resized to 224×224 pixels for model input.


## Data Augmentation
The dataset was enhanced using three core offline augmentation stages applied prior to training. 
1. ```Horizontal Flip``` – randomly flips images horizontally.
2. ```Rotation ±15°``` – rotates images within a ±15-degree range.
3. ```CLAHE``` (Contrast Limited Adaptive Histogram Equalization) – improves local contrast with a clip limit of 5.


The output of these augmentations resulted in the following dataset sizes:
- RGB Augmented Images: ```16,556```
- Thermal Augmented Images: ```16,556```

These augmented images were then used as the base dataset for all models, ensuring consistent training data across architectures.


# Authors
- Fiorella Scarpino (21010043), University of the West of England (UWE)
- May Sanejo (15006280), University of the West of England (UWE)
- Soumia Kouadri (24058628), University of the West of England (UWE)