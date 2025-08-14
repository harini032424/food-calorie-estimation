import pandas as pd
import numpy as np
import joblib
from utils import extract_image_features, preprocess_metadata

# Load trained model
model = joblib.load("calorie_predictor_rf.pkl")
X_meta_columns = joblib.load("X_meta_columns.pkl")

def predict_calories(image_paths, metadata_list):
    all_features = []

    for img_path, meta_dict in zip(image_paths, metadata_list):
        img_feat = extract_image_features(img_path)
        meta_df = pd.DataFrame([meta_dict])
        meta_feat = preprocess_metadata(meta_df, X_meta_columns).values.flatten()
        combined_feat = np.hstack((img_feat, meta_feat))
        all_features.append(combined_feat)

    X_new = np.vstack(all_features)
    predictions = model.predict(X_new)
    return predictions

# Example usage
if __name__ == "__main__":
    image_paths = ["new_food1.jpg", "new_food2.jpg"]
    metadata_list = [
        {'portion_size': 150, 'ingredient_type': 'crab_cakes', 'prep_style': 'fried'},
        {'portion_size': 200, 'ingredient_type': 'salmon', 'prep_style': 'grilled'}
    ]
    predicted_calories = predict_calories(image_paths, metadata_list)
    for path, cal in zip(image_paths, predicted_calories):
        print(f"Image: {path}, Predicted Calories: {cal:.2f}")
