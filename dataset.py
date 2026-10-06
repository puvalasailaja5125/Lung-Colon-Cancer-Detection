# import tensorflow as tf

# # Dataset paths
# TRAIN_VAL_DIR = r"dataset\LUNG_COLON_IMAGE_SET\Train and Validation Set"

# # Image settings
# IMAGE_SIZE = (224, 224)
# BATCH_SIZE = 16
# SEED = 42

# # Create training dataset
# train_dataset = tf.keras.utils.image_dataset_from_directory(
#     TRAIN_VAL_DIR,
#     validation_split=0.20,
#     subset="training",
#     seed=SEED,
#     image_size=IMAGE_SIZE,
#     batch_size=BATCH_SIZE,
#     shuffle=True
# )

# # Create validation dataset
# validation_dataset = tf.keras.utils.image_dataset_from_directory(
#     TRAIN_VAL_DIR,
#     validation_split=0.20,
#     subset="validation",
#     seed=SEED,
#     image_size=IMAGE_SIZE,
#     batch_size=BATCH_SIZE,
#     shuffle=True
# )

# print("Classes:", train_dataset.class_names)

# print("Training batches:", tf.data.experimental.cardinality(train_dataset).numpy())
# print("Validation batches:", tf.data.experimental.cardinality(validation_dataset).numpy())

import tensorflow as tf
from tensorflow.keras.applications.densenet import preprocess_input

# -----------------------------
# Configuration
# -----------------------------

TRAIN_VAL_DIR = r"dataset\LUNG_COLON_IMAGE_SET\Train and Validation Set"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42


# -----------------------------
# Data Augmentation
# -----------------------------

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.05),
    tf.keras.layers.RandomZoom(0.10),
], name="data_augmentation")


# -----------------------------
# Training Dataset
# -----------------------------

train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_VAL_DIR,
    validation_split=0.20,
    subset="training",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

class_names = train_dataset.class_names

# -----------------------------
# Validation Dataset
# -----------------------------

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_VAL_DIR,
    validation_split=0.20,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# -----------------------------
# Preprocessing Function
# -----------------------------

def preprocess_data(images, labels, training=False):

    images = tf.cast(images, tf.float32)

    if training:
        images = data_augmentation(images, training=True)

    images = preprocess_input(images)

    return images, labels


# Apply preprocessing
train_dataset = train_dataset.map(
    lambda images, labels: preprocess_data(images, labels, training=True),
    num_parallel_calls=tf.data.AUTOTUNE
)

validation_dataset = validation_dataset.map(
    lambda images, labels: preprocess_data(images, labels, training=False),
    num_parallel_calls=tf.data.AUTOTUNE
)


# Improve input pipeline performance
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
validation_dataset = validation_dataset.prefetch(tf.data.AUTOTUNE)


# -----------------------------
# Display information
# -----------------------------

print("Classes:", class_names)
print("Training batches:", tf.data.experimental.cardinality(train_dataset).numpy())
print("Validation batches:", tf.data.experimental.cardinality(validation_dataset).numpy())