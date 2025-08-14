import tensorflow as tf
import numpy as np
import os
import pandas as pd
from PIL import Image
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Paths
MODEL_PATH = "food_calorie_model.h5"
DATA_DIR = r"C:/Users/03har/food-calorie-estimation/data/food-101-tiny/valid"
CSV_PATH = r"C:/Users/03har/food-calorie-estimation/data/nutrition.csv"

# Load model without compiling to avoid H5 deserialization issues
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# Load nutrition CSV and create calorie mapping
nutrition_df = pd.read_csv(CSV_PATH)
nutrition_df['label'] = nutrition_df['label'].str.lower().str.replace(" ", "_").str.strip()
calorie_dict = dict(zip(nutrition_df['label'], nutrition_df['calories']))

# Preprocess image function
def preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Loop over all validation images
y_true = []
y_pred = []

for class_name in os.listdir(DATA_DIR):
    class_folder = os.path.join(DATA_DIR, class_name)
    if not os.path.isdir(class_folder) or class_name not in calorie_dict:
        continue
    for img_file in os.listdir(class_folder):
        img_path = os.path.join(class_folder, img_file)
        try:
            img_array = preprocess_image(img_path)
            pred = model.predict(img_array)[0][0]
            y_pred.append(pred)
            y_true.append(calorie_dict[class_name])
        except Exception as e:
            print(f"Skipped {img_path}: {e}")

# Convert to numpy
y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Compute metrics
mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print(f"Validation MAE: {mae:.2f} kcal")
print(f"Validation MSE: {mse:.2f}")
print(f"Validation R2 Score: {r2:.2f}")
