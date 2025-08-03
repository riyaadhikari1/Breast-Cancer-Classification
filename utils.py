import cv2
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input

IMG_SIZE = 256

def load_and_preprocess_image(uploaded_file):
    # Read bytes from uploaded file
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    # Decode image in color mode
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    # Resize to desired input size
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    # Convert to float32 and apply ResNet50 preprocessing
    img = preprocess_input(img.astype(np.float32))
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    return img
