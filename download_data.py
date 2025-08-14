import tensorflow as tf
import os

DATA_DIR = r"C:/Users/03har/food-calorie-estimation/data/food-101-tiny"

# Load training dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, "train"),
    image_size=(224, 224),
    batch_size=32
)

# Load validation dataset
val_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, "valid"),
    image_size=(224, 224),
    batch_size=32
)

# Optional: normalize pixel values to [0,1]
train_ds = train_ds.map(lambda x, y: (x / 255.0, y))
val_ds = val_ds.map(lambda x, y: (x / 255.0, y))
