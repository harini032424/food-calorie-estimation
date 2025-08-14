🎯 Food Calorie Estimation from Images + Metadata

Project Description  
This project estimates the calorie content of food items by combining image features extracted using a pre-trained ResNet50 CNN with nutritional metadata such as portion size, ingredient type, and preparation style. The combined features are used to train a Random Forest Regressor for accurate calorie prediction.  

It demonstrates the integration of computer vision and machine learning for practical food analytics.

✨ Features
- Extracts deep features from food images using ResNet50
- Processes and one-hot encodes categorical metadata
- Combines image and metadata features for Random Forest Regression
- Evaluates model performance using MAE and R² score
- Visualizes feature importance to interpret model predictions
- Predicts calories for new images with corresponding metadata

🛠️ Technologies & Libraries
- Python
- TensorFlow / Keras
- TensorFlow Datasets
- scikit-learn
- Pandas, NumPy
- Matplotlib
- Joblib

📦 Dataset
- Images: Food-101 Dataset ([link](https://www.tensorflow.org/datasets/catalog/food101))  
- Metadata: nutrition.csv (columns: image_name, portion_size, ingredient_type, prep_style, calories)

⚡ Installation
1. Clone the repository:
   git clone https://github.com/harini032424/food-calorie-estimation.git
2. Navigate to the project folder:food-calorie-estimation
3. Install dependencies
4.Train the Model:python food_calorie_estimation.py
  This will train the Random Forest model and save it as calorie_predictor_rf.pkl.

Predict New Data
from food_calorie_estimation import predict_calories, X_meta_df

image_paths = ["new_food1.jpg", "new_food2.jpg"]
metadata_list = [{'portion_size': 150, 'ingredient_type': 'crab_cakes', 'prep_style': 'fried'},
    {'portion_size': 200, 'ingredient_type': 'salmon', 'prep_style': 'grilled'}
]

predicted_calories = predict_calories(image_paths, metadata_list, X_meta_df.columns)
print(predicted_calories)

📊 Results
Evaluated using MAE and R² score

Visualized feature importance to identify key contributors from image and metadata features

Combines visual and metadata information to improve prediction accuracy


