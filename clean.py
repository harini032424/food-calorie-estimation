import pandas as pd

# Path to your metadata
CSV_PATH = r"C:/Users/03har/food-calorie-estimation/data/nutrition.csv"

# Load CSV
nutrition_df = pd.read_csv(CSV_PATH)

# Preview columns
print(nutrition_df.columns)
# Output should be something like: ['label', 'weight', 'calories', 'protein', ...]

# Clean and rename 'label' to 'food_name'
nutrition_df['food_name'] = nutrition_df['label'].str.lower().str.replace("_", " ").str.strip()

# Keep only name & calories
nutrition_df = nutrition_df[['food_name', 'calories']]

# Preview
print(nutrition_df.head())
