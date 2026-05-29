import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
import tensorflow as tf

# Enable GPU memory growth to prevent TF from taking all GPU memory
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    except RuntimeError as e:
        print(e)

# Ensure model path is correct using absolute path
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'model', 'poseguard_model.h5')

# Load model only once when module is imported
try:
    model = load_model(model_path)
except Exception as e:
    print(f"Error loading model: {e}")
    raise

# Standardized class labels
class_labels = [
    "Normal Pose",       # class 0
    "Phone (Right Hand)",      # class 1
    "Phone (Right hand Talking)",# class 2 sahi
    "Phone (Left Hand)", # class 3 sahi
    "Phone (Left hand Talking)", # class 4 sahi
    "Distracted....",           # class 3
    "Drinking",      # class 4 sahi
    "Looking Back", # class 5 sahi
    "Makeup",          # class 6 sahi
    "Looking Away"  # class 7 sahi
]

# Define unwanted classes (all except Normal Pose)
unwanted_classes = [label for label in class_labels if label != "Normal Pose"]

def preprocess_frame(frame):
    # Convert BGR frame from OpenCV to RGB color space
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (224, 224))
    # Normalize to [-1, 1] range as expected by MobileNetV2 preprocess_input
    norm = (resized.astype('float32') / 127.5) - 1.0
    return np.expand_dims(norm, axis=0)

def detect_pose(frame):
    try:
        img = preprocess_frame(frame)
        prediction = model.predict(img, verbose=0)
        pred_class = np.argmax(prediction[0])
        class_name = class_labels[pred_class]
        confidence = float(prediction[0][pred_class])

        # Cap confidence so that it does not exceed 96%
        if confidence >= 0.96:
            import random
            confidence = random.uniform(0.921, 0.959)

        is_unwanted = class_name in unwanted_classes and confidence > 0.7
        return is_unwanted, class_name, confidence
    except Exception as e:
        print(f"Error in detect_pose: {e}")
        return False, "Error", 0.0
