import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
from utils import cnn_model, preprocess_image

# Load metadata
METADATA_CSV = "data/nutrition.csv"
metadata = pd.read_csv(METADATA_CSV)

num_samples = 1000  # adjust as needed

# --- Feature extraction (for demo, take only first N samples) ---
train_features = np.load("train_features.npy")  # Optional: save CNN features if large

# Metadata preprocessing
X_meta_df = metadata.iloc[:num_samples].drop(columns=['calories'])
cat_cols = X_meta_df.select_dtypes(include='object').columns
X_meta_df = pd.get_dummies(X_meta_df, columns=cat_cols)
X_meta = X_meta_df.values

# Combine image + metadata
X = np.hstack((train_features[:num_samples], X_meta))
y = metadata.iloc[:num_samples]['calories'].values

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}, R2: {r2_score(y_test, y_pred):.2f}")

# Save model and metadata columns
joblib.dump(model, "calorie_predictor_rf.pkl")
joblib.dump(X_meta_df.columns, "X_meta_columns.pkl")
