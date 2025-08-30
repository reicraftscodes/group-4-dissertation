# Convolutional Neural Network (CNN) for Facial Expression Recognition (FER) Documentation

### Table of Contents
- [Overview](#overview)
- [Datasets](#dataset)
- [Run Google Collab](#run-to-google-collab)
  - [Single Modality](#single-modality)
  - [Multimodal Early Fusion](#multimodal-early-fusion)
- [Results](#results)

# Overview
A comprehensive implementation of Convolutional Neural Network (CNN) for Facial Expression Recognition (FER) supporting RGB, Thermal, and Multimodal modalities

## Dataset
### Supported Emotions
- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprised

## Data Structure
The dataset should be organised as follows:
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

# Run to Google Collab

### Single Modality

**To run CNN_Final.ipynb file**

1. If running on Google Colab: run all cells under ```1.1 Setup for Colab``` ensuring that code cell 2 is changed to personal path 
2. Ensure that Data file is in the same directory as the ```CNN Final``` notebook
3. Uncomment code cell 4 to run the requirements file
4. Change the ```modalDir``` variable under the ```loadDataSingleModality``` function to either ```rgbDir``` or ```thermalDir``` depending on which modality needs to be trained
5. Run notebook as normal

### Multimodal Early Fusion

**To run the CNN_EarlyFusion_Concat_Final.ipynb file**
1. If running on Google Colab: run all cells under ```1.1 Setup for Colab``` ensuring that code cell 2 is changed to personal path 
2. Ensure that Data file is in the same directory as the ```CNN Final``` notebook 
3. Uncomment code cell 4 to run the requirements file 
4. Run notebook as normal

### Results
To view results, 
- Results can be viewed in notebook after running
- Results are also saved as csv files and images, in a folder called ```trainingResults```, with each result file having the suffix ```RGB``` or ```Thermal``` depending on which modality tested


# Authors
- Fiorella Scarpino (21010043), University of the West of England (UWE)
- May Sanejo (15006280), University of the West of England (UWE)
- Soumia Kouadri (24058628), University of the West of England (UWE)