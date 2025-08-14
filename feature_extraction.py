import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Load pre-trained CNN (ResNet50)
cnn_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

def extract_image_features(img_path):
    """Extract features from a single image without verbose output"""
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    features = cnn_model.predict(img_array, verbose=0)  # suppress logging
    return features.flatten()

def batch_extract_features(img_folder):
    """Extract features for all images in subfolders"""
    features_list = []
    img_labels = []

    # Loop over class subfolders
    for class_name in os.listdir(img_folder):
        class_path = os.path.join(img_folder, class_name)
        if os.path.isdir(class_path):
            for img_file in os.listdir(class_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_path, img_file)
                    features = extract_image_features(img_path)
                    features_list.append(features)
                    img_labels.append(class_name)  # folder name as label

    return np.array(features_list), img_labels
