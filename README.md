# Breast Cancer Classifier with Grad-CAM

This Streamlit-based web app allows users to classify histopathological breast cancer images as **Benign** or **Malignant** using deep learning models (ResNet50-based), with Grad-CAM explainability.

## Features

- Upload single or multiple images (or zip archives)
- Dual-model classification with confidence-based selection
- Grad-CAM visualizations for interpretability
- PDF report generation with predictions and heatmaps
- Feedback system for model accuracy tracking

## Tech Stack

- Python
- TensorFlow & Keras
- OpenCV
- Streamlit
- FPDF
- ResNet50 (pretrained backbone)

## Folder Structure

project/

├── app.py              # Streamlit application

├── utils.py            # Image preprocessing helper

├── best_model_fold3.keras

├── best_model_fold4.keras

├── requirement.txt

├── .gitignore

├── LICENSE

└── README.md

└── test images
