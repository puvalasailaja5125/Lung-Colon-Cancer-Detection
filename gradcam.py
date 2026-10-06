# import tensorflow as tf
# import numpy as np
# import matplotlib.pyplot as plt

# from tensorflow.keras.applications.densenet import preprocess_input

# from cbam import (
#     CBAM,
#     ChannelAttention,
#     SpatialAttention
# )

# from focal_loss import SparseCategoricalFocalLoss


# # ============================================================
# # PATHS
# # ============================================================

# MODEL_PATH = r"models\densenet121_cbam_focal.keras"

# IMAGE_PATH = (
#     r"dataset\LUNG_COLON_IMAGE_SET\Test Set"
#     r"\lung_aca\lungaca3915.jpeg"
# )

# IMAGE_SIZE = (224, 224)


# # ============================================================
# # CLASS NAMES
# # ============================================================

# CLASS_NAMES = [
#     "colon_aca",
#     "colon_n",
#     "lung_aca",
#     "lung_n",
#     "lung_scc"
# ]


# # ============================================================
# # LOAD TRAINED MODEL
# # ============================================================

# model = tf.keras.models.load_model(
#     MODEL_PATH,
#     custom_objects={
#         "CBAM": CBAM,
#         "ChannelAttention": ChannelAttention,
#         "SpatialAttention": SpatialAttention,
#         "SparseCategoricalFocalLoss": SparseCategoricalFocalLoss
#     }
# )

# print("Model loaded successfully")


# # ============================================================
# # FIND DENSENET121 BACKBONE
# # ============================================================

# base_model = None

# for layer in model.layers:

#     if isinstance(layer, tf.keras.Model):

#         if "densenet" in layer.name.lower():

#             base_model = layer
#             break


# if base_model is None:

#     raise ValueError(
#         "DenseNet121 base model could not be found."
#     )


# print("DenseNet model:", base_model.name)


# # ============================================================
# # CHECK DENSENET OUTPUT
# # ============================================================

# print(
#     "DenseNet output shape:",
#     base_model.output.shape
# )


# # ============================================================
# # LOAD THE SAME DATASET IMAGE
# # ============================================================

# image = tf.keras.utils.load_img(
#     IMAGE_PATH,
#     target_size=IMAGE_SIZE
# )

# image_array = tf.keras.utils.img_to_array(
#     image
# )

# # Keep an unprocessed copy for visualization
# original_image = image_array.astype(
#     np.uint8
# )


# # ============================================================
# # PREPARE IMAGE FOR DENSENET
# # ============================================================

# input_image = np.expand_dims(
#     image_array,
#     axis=0
# )

# input_image = preprocess_input(
#     input_image
# )


# # ============================================================
# # NORMAL MODEL PREDICTION
# # ============================================================

# predictions = model.predict(
#     input_image,
#     verbose=0
# )

# predicted_class = np.argmax(
#     predictions[0]
# )

# predicted_probability = predictions[0][
#     predicted_class
# ]


# print("\nPrediction:")

# print(
#     "Class:",
#     CLASS_NAMES[predicted_class]
# )

# print(
#     "Confidence:",
#     f"{predicted_probability * 100:.2f}%"
# )


# # ============================================================
# # GRAD-CAM
# #
# # We calculate gradients with respect to the FINAL
# # DenseNet121 feature maps (7 x 7 x 1024).
# # These are exactly the features passed to CBAM.
# # ============================================================

# with tf.GradientTape() as tape:

#     # --------------------------------------------------------
#     # DenseNet121 feature extraction
#     # --------------------------------------------------------

#     dense_features = base_model(
#         input_image,
#         training=False
#     )

#     print(
#         "\nDenseNet feature shape:",
#         dense_features.shape
#     )


#     # --------------------------------------------------------
#     # CBAM
#     # --------------------------------------------------------

#     cbam_layer = model.get_layer(
#         "cbam"
#     )

#     attention_features = cbam_layer(
#         dense_features,
#         training=False
#     )

#     print(
#         "After CBAM:",
#         attention_features.shape
#     )


#     # --------------------------------------------------------
#     # Global Average Pooling
#     # --------------------------------------------------------

#     pooled_features = model.get_layer(
#         "global_average_pooling2d"
#     )(
#         attention_features
#     )


#     # --------------------------------------------------------
#     # Dropout
#     # --------------------------------------------------------

#     dropped_features = model.get_layer(
#         "dropout"
#     )(
#         pooled_features,
#         training=False
#     )


#     # --------------------------------------------------------
#     # Final Dense classifier
#     # --------------------------------------------------------

#     class_predictions = model.get_layer(
#         "dense_2"
#     )(
#         dropped_features
#     )


#     # --------------------------------------------------------
#     # Select predicted class score
#     # --------------------------------------------------------

#     class_score = class_predictions[
#         :,
#         predicted_class
#     ]


# # ============================================================
# # CALCULATE GRADIENTS
# # ============================================================

# gradients = tape.gradient(
#     class_score,
#     dense_features
# )


# if gradients is None:

#     raise ValueError(
#         "Gradients are None. "
#         "Grad-CAM could not calculate gradients."
#     )


# print(
#     "Gradient shape:",
#     gradients.shape
# )


# # ============================================================
# # GLOBAL AVERAGE POOLING OF GRADIENTS
# # ============================================================

# pooled_gradients = tf.reduce_mean(
#     gradients,
#     axis=(0, 1, 2)
# )


# # ============================================================
# # REMOVE BATCH DIMENSION
# # ============================================================

# dense_features = dense_features[0]


# # ============================================================
# # CREATE GRAD-CAM HEATMAP
# # ============================================================

# heatmap = tf.reduce_sum(
#     dense_features * pooled_gradients,
#     axis=-1
# )


# # ============================================================
# # KEEP ONLY POSITIVE ACTIVATIONS
# # ============================================================

# heatmap = tf.maximum(
#     heatmap,
#     0
# )


# # ============================================================
# # NORMALIZE HEATMAP
# # ============================================================

# max_value = tf.reduce_max(
#     heatmap
# )

# heatmap = heatmap / (
#     max_value +
#     tf.keras.backend.epsilon()
# )


# heatmap = heatmap.numpy()


# # ============================================================
# # RESIZE HEATMAP TO ORIGINAL IMAGE SIZE
# # ============================================================

# heatmap = tf.image.resize(
#     heatmap[..., np.newaxis],
#     IMAGE_SIZE
# )

# heatmap = heatmap.numpy().squeeze()


# # ============================================================
# # DISPLAY RESULTS
# # ============================================================

# plt.figure(
#     figsize=(12, 4)
# )


# # ------------------------------------------------------------
# # 1. ORIGINAL IMAGE
# # ------------------------------------------------------------

# plt.subplot(
#     1,
#     3,
#     1
# )

# plt.imshow(
#     original_image
# )

# plt.title(
#     "Original Image"
# )

# plt.axis("off")


# # ------------------------------------------------------------
# # 2. GRAD-CAM HEATMAP
# # ------------------------------------------------------------

# plt.subplot(
#     1,
#     3,
#     2
# )

# plt.imshow(
#     heatmap,
#     cmap="jet"
# )

# plt.title(
#     "Grad-CAM Heatmap"
# )

# plt.axis("off")


# # ------------------------------------------------------------
# # 3. OVERLAY
# # ------------------------------------------------------------

# plt.subplot(
#     1,
#     3,
#     3
# )

# plt.imshow(
#     original_image
# )

# plt.imshow(
#     heatmap,
#     cmap="jet",
#     alpha=0.4
# )

# plt.title(
#     f"{CLASS_NAMES[predicted_class]}\n"
#     f"{predicted_probability * 100:.2f}%"
# )

# plt.axis("off")


# plt.tight_layout()

# plt.show()



import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.applications.densenet import preprocess_input

from cbam import (
    CBAM,
    ChannelAttention,
    SpatialAttention
)

from focal_loss import SparseCategoricalFocalLoss


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = r"models\densenet121_cbam_focal.keras"

TEST_DIR = r"dataset\LUNG_COLON_IMAGE_SET\Test Set"

OUTPUT_DIR = r"results\gradcam"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# IMAGE SETTINGS
# ============================================================

IMAGE_SIZE = (224, 224)


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = [
    "colon_aca",
    "colon_n",
    "lung_aca",
    "lung_n",
    "lung_scc"
]


# ============================================================
# SELECT REPRESENTATIVE TEST IMAGES
# ============================================================

IMAGE_PATHS = {
    "colon_aca": os.path.join(
        TEST_DIR,
        "colon_aca",
        "colonca4273.jpeg"
    ),

    "colon_n": os.path.join(
        TEST_DIR,
        "colon_n",
        "colonn4021.jpeg"
    ),

    "lung_aca": os.path.join(
        TEST_DIR,
        "lung_aca",
        "lungaca3915.jpeg"
    ),

    "lung_n": os.path.join(
        TEST_DIR,
        "lung_n",
        "lungn4429.jpeg"
    ),

    "lung_scc": os.path.join(
        TEST_DIR,
        "lung_scc",
        "lungscc4429.jpeg"
    )
}


# ============================================================
# LOAD MODEL
# ============================================================

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "CBAM": CBAM,
        "ChannelAttention": ChannelAttention,
        "SpatialAttention": SpatialAttention,
        "SparseCategoricalFocalLoss":
            SparseCategoricalFocalLoss
    }
)

print("Model loaded successfully")


# ============================================================
# FIND DENSENET121
# ============================================================

base_model = None

for layer in model.layers:

    if isinstance(layer, tf.keras.Model):

        if "densenet" in layer.name.lower():

            base_model = layer
            break


if base_model is None:

    raise ValueError(
        "DenseNet121 base model could not be found."
    )


print(
    "DenseNet model:",
    base_model.name
)

print(
    "DenseNet output shape:",
    base_model.output.shape
)


# ============================================================
# GET MODEL COMPONENTS
# ============================================================

cbam_layer = model.get_layer("cbam")

gap_layer = model.get_layer(
    "global_average_pooling2d"
)

dropout_layer = model.get_layer(
    "dropout"
)

classifier_layer = model.get_layer(
    "dense_2"
)


# ============================================================
# PROCESS EACH CLASS
# ============================================================

for true_class in CLASS_NAMES:

    image_path = IMAGE_PATHS[true_class]

    print("\n" + "=" * 60)

    print(
        "Processing:",
        true_class
    )

    print(
        "Image:",
        image_path
    )


    # --------------------------------------------------------
    # Check image exists
    # --------------------------------------------------------

    if not os.path.exists(image_path):

        print(
            "WARNING: Image not found."
        )

        print(
            "Skipping:",
            true_class
        )

        continue


    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    image = tf.keras.utils.load_img(
        image_path,
        target_size=IMAGE_SIZE
    )

    image_array = tf.keras.utils.img_to_array(
        image
    )

    original_image = image_array.astype(
        np.uint8
    )


    # --------------------------------------------------------
    # Preprocess image
    # --------------------------------------------------------

    input_image = np.expand_dims(
        image_array,
        axis=0
    )

    input_image = preprocess_input(
        input_image
    )


    # --------------------------------------------------------
    # Prediction + Grad-CAM
    # --------------------------------------------------------

    with tf.GradientTape() as tape:

        # DenseNet feature extraction
        dense_features = base_model(
            input_image,
            training=False
        )

        # Watch the feature tensor
        tape.watch(dense_features)


        # CBAM
        attention_features = cbam_layer(
            dense_features,
            training=False
        )


        # Global Average Pooling
        pooled_features = gap_layer(
            attention_features
        )


        # Dropout
        dropped_features = dropout_layer(
            pooled_features,
            training=False
        )


        # Classifier
        class_predictions = classifier_layer(
            dropped_features
        )


        # Predicted class
        predicted_class = tf.argmax(
            class_predictions[0]
        )

        predicted_probability = tf.reduce_max(
            class_predictions[0]
        )


        # Score of predicted class
        class_score = class_predictions[
            0,
            predicted_class
        ]


    # --------------------------------------------------------
    # Calculate gradients
    # --------------------------------------------------------

    gradients = tape.gradient(
        class_score,
        dense_features
    )


    if gradients is None:

        print(
            "WARNING: Gradients are None."
        )

        continue


    # --------------------------------------------------------
    # Global average pooling of gradients
    # --------------------------------------------------------

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(0, 1, 2)
    )


    # --------------------------------------------------------
    # Remove batch dimension
    # --------------------------------------------------------

    feature_maps = dense_features[0]


    # --------------------------------------------------------
    # Generate heatmap
    # --------------------------------------------------------

    heatmap = tf.reduce_sum(
        feature_maps * pooled_gradients,
        axis=-1
    )


    # --------------------------------------------------------
    # ReLU
    # --------------------------------------------------------

    heatmap = tf.maximum(
        heatmap,
        0
    )


    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    heatmap = heatmap / (
        tf.reduce_max(heatmap)
        + tf.keras.backend.epsilon()
    )


    heatmap = heatmap.numpy()


    # --------------------------------------------------------
    # Resize heatmap
    # --------------------------------------------------------

    heatmap = tf.image.resize(
        heatmap[..., np.newaxis],
        IMAGE_SIZE
    )

    heatmap = heatmap.numpy().squeeze()


    # --------------------------------------------------------
    # Convert prediction to integer
    # --------------------------------------------------------

    predicted_class = int(
        predicted_class.numpy()
    )

    predicted_probability = float(
        predicted_probability.numpy()
    )


    predicted_class_name = CLASS_NAMES[
        predicted_class
    ]


    # --------------------------------------------------------
    # Print result
    # --------------------------------------------------------

    print(
        "Actual class:",
        true_class
    )

    print(
        "Predicted class:",
        predicted_class_name
    )

    print(
        "Confidence:",
        f"{predicted_probability * 100:.2f}%"
    )


    # ========================================================
    # CREATE FIGURE
    # ========================================================

    plt.figure(
        figsize=(15, 5)
    )


    # --------------------------------------------------------
    # Original
    # --------------------------------------------------------

    plt.subplot(
        1,
        3,
        1
    )

    plt.imshow(
        original_image
    )

    plt.title(
        f"Original\nActual: {true_class}"
    )

    plt.axis("off")


    # --------------------------------------------------------
    # Heatmap
    # --------------------------------------------------------

    plt.subplot(
        1,
        3,
        2
    )

    plt.imshow(
        heatmap,
        cmap="jet"
    )

    plt.title(
        "Grad-CAM Heatmap"
    )

    plt.axis("off")


    # --------------------------------------------------------
    # Overlay
    # --------------------------------------------------------

    plt.subplot(
        1,
        3,
        3
    )

    plt.imshow(
        original_image
    )

    plt.imshow(
        heatmap,
        cmap="jet",
        alpha=0.4
    )

    plt.title(
        f"Prediction: {predicted_class_name}\n"
        f"Confidence: "
        f"{predicted_probability * 100:.2f}%"
    )

    plt.axis("off")


    plt.tight_layout()


    # ========================================================
    # SAVE FIGURE
    # ========================================================

    output_path = os.path.join(
        OUTPUT_DIR,
        f"gradcam_{true_class}.png"
    )


    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


    print(
        "Saved:",
        output_path
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 60)

print(
    "Grad-CAM generation completed."
)

print(
    "Results saved in:",
    OUTPUT_DIR
)
