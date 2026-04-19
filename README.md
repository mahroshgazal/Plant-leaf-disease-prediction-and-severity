# Plant Disease Detection & Severity Estimation 🌿

## Overview
This project builds a deep learning system that not only detects plant diseases from leaf images but also estimates the severity of the infection. 

Unlike standard classification models, this system predicts:
- Whether a leaf is healthy or diseased
- The type of disease
- The percentage of the leaf affected (severity)

This makes the solution more practical for real-world agricultural use.

---

## Key Features
- Multi-class disease classification using CNNs
- Severity estimation (0–100% infection level)
- Image preprocessing with leaf segmentation
- Data augmentation for improved generalization
- Visual explanations using Grad-CAM heatmaps

---

## Dataset
- PlantVillage dataset (or similar plant leaf datasets)
- Contains labeled images of healthy and diseased leaves

---

## Methodology

### 1. Data Preprocessing
- Image resizing and normalization
- Data augmentation:
  - Rotation
  - Scaling
  - Brightness/contrast adjustments
- Leaf segmentation using OpenCV to remove background noise

### 2. Disease Classification
- Model: CNN (ResNet / EfficientNet)
- Task: Classify leaf into healthy or disease categories

### 3. Severity Estimation
- Regression model to predict infection percentage
- Alternatively, pixel-based estimation of affected regions

### 4. Visualization
- Grad-CAM / saliency maps
- Highlights regions responsible for predictions

---

## Tech Stack
- Python
- PyTorch
- OpenCV
- NumPy
- Matplotlib

---

## Results
- Accurate disease classification across multiple classes
- Meaningful severity estimation (0–100%)
- Visual heatmaps for model interpretability

---

## Future Improvements
- Deploy as a mobile application for farmers
- Integrate real-time camera-based detection
- Use attention mechanisms for better localization
- Expand dataset for more plant species

---

## Impact
This project goes beyond simple classification by quantifying plant damage. It can help:
- Farmers make better treatment decisions
- Reduce crop loss
- Enable real-time field diagnostics

---

## Author
Mahrosh Gazal
