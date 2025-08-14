import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

# Load pre-trained CNN once
cnn_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

def preprocess_image(img_path):
    """Load and preprocess a single image"""
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

def extract_image_features(img_path):
    """Extract CNN features from a single image"""
    img_array = preprocess_image(img_path)
    features = cnn_model.predict(img_array)
    return features.flatten()

def preprocess_metadata(new_meta_df, X_meta_columns):
    """One-hot encode and align metadata with training columns"""
    new_meta_df = pd.get_dummies(new_meta_df)
    return new_meta_df.reindex(columns=X_meta_columns, fill_value=0)
