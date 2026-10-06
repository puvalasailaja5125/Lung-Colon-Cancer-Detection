import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from tensorflow.keras.applications.densenet import preprocess_input
from focal_loss import SparseCategoricalFocalLoss
from cbam import CBAM


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = r"models\densenet121_cbam_focal.keras"

IMAGE_SIZE = (224, 224)

CLASS_NAMES = [
    "colon_aca",
    "colon_n",
    "lung_aca",
    "lung_n",
    "lung_scc"
]

CLASS_LABELS = {
    "colon_aca": "Colon Adenocarcinoma",
    "colon_n": "Normal Colon",
    "lung_aca": "Lung Adenocarcinoma",
    "lung_n": "Normal Lung",
    "lung_scc": "Lung Squamous Cell Carcinoma"
}

OUTPUT_DIR = r"results\predictions"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "CBAM": CBAM,
        "SparseCategoricalFocalLoss": SparseCategoricalFocalLoss
    },
    compile=False
)

print("Model loaded successfully.")


# ============================================================
# GET DENSENET BACKBONE
# ============================================================

base_model = model.get_layer("densenet121")

print("DenseNet backbone found:")
print(base_model.name)
print("Output shape:", base_model.output.shape)


# ============================================================
# PREPROCESS IMAGE
# ============================================================

def load_and_preprocess_image(image_path):

    original_image = Image.open(image_path).convert("RGB")

    resized_image = original_image.resize(IMAGE_SIZE)

    image_array = np.array(
        resized_image,
        dtype=np.float32
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    image_array = preprocess_input(
        image_array
    )

    return original_image, image_array


# ============================================================
# GRAD-CAM
# ============================================================

def generate_gradcam(
    image_array,
    predicted_index
):
    """
    Generate Grad-CAM using the DenseNet121 feature maps
    before the CBAM attention layer.
    """

    # Get the layers from the saved model
    cbam_layer = model.get_layer("cbam")
    gap_layer = model.get_layer("global_average_pooling2d")
    dropout_layer = model.get_layer("dropout")
    classifier_layer = model.get_layer("dense_2")

    with tf.GradientTape() as tape:

        # ----------------------------------------------------
        # 1. DenseNet121 feature extraction
        # ----------------------------------------------------

        feature_maps = base_model(
            image_array,
            training=False
        )

        # Tell GradientTape to track these feature maps
        tape.watch(feature_maps)

        # ----------------------------------------------------
        # 2. CBAM attention
        # ----------------------------------------------------

        attention_features = cbam_layer(
            feature_maps
        )

        # ----------------------------------------------------
        # 3. Classification head
        # ----------------------------------------------------

        pooled_features = gap_layer(
            attention_features
        )

        dropped_features = dropout_layer(
            pooled_features,
            training=False
        )

        predictions = classifier_layer(
            dropped_features
        )

        # Score of the predicted class
        target_score = predictions[
            :,
            predicted_index
        ]

    # --------------------------------------------------------
    # 4. Calculate gradients
    # --------------------------------------------------------

    gradients = tape.gradient(
        target_score,
        feature_maps
    )

    # --------------------------------------------------------
    # 5. Calculate importance of each feature map
    # --------------------------------------------------------

    weights = tf.reduce_mean(
        gradients,
        axis=(1, 2)
    )

    # --------------------------------------------------------
    # 6. Weighted combination of feature maps
    # --------------------------------------------------------

    cam = tf.reduce_sum(
        feature_maps *
        weights[:, tf.newaxis, tf.newaxis, :],
        axis=-1
    )

    # --------------------------------------------------------
    # 7. Remove negative values
    # --------------------------------------------------------

    cam = tf.maximum(
        cam,
        0
    )

    # --------------------------------------------------------
    # 8. Normalize heatmap to 0-1
    # --------------------------------------------------------

    cam = cam / (
        tf.reduce_max(cam)
        + tf.keras.backend.epsilon()
    )

    return cam[0].numpy()


# ============================================================
# CREATE GRAD-CAM OVERLAY
# ============================================================

def create_overlay(
    original_image,
    heatmap
):
    """
    Convert Grad-CAM heatmap into a colored heatmap
    and create a colored overlay on the original image.
    """

    # --------------------------------------------------------
    # 1. Resize heatmap to original image size
    # --------------------------------------------------------

    heatmap_image = Image.fromarray(
        np.uint8(heatmap * 255)
    )

    heatmap_image = heatmap_image.resize(
        original_image.size
    )

    heatmap_array = np.array(
        heatmap_image
    )

    # --------------------------------------------------------
    # 2. Apply JET color map
    # --------------------------------------------------------

    colored_heatmap = plt.get_cmap(
        "jet"
    )(heatmap_array / 255.0)[:, :, :3]

    colored_heatmap = np.uint8(
        colored_heatmap * 255
    )

    # --------------------------------------------------------
    # 3. Convert original image to NumPy
    # --------------------------------------------------------

    original_array = np.array(
        original_image
    )

    # --------------------------------------------------------
    # 4. Blend original image + heatmap
    # --------------------------------------------------------

    overlay = (
        0.60 * original_array
        + 0.40 * colored_heatmap
    )

    overlay = np.uint8(
        np.clip(
            overlay,
            0,
            255
        )
    )

    return (
        heatmap_array,
        colored_heatmap,
        overlay
    )

# ============================================================
# MAIN PREDICTION FUNCTION
# ============================================================

def predict_image(image_path):

    if not os.path.exists(image_path):

        print("\nERROR:")
        print("Image not found:")
        print(image_path)

        return

    print("\nProcessing image...")

    original_image, image_array = (
        load_and_preprocess_image(
            image_path
        )
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    predictions = model.predict(
        image_array,
        verbose=0
    )[0]

    predicted_index = int(
        np.argmax(predictions)
    )

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    confidence = (
        predictions[predicted_index]
        * 100
    )

    # --------------------------------------------------------
    # Grad-CAM
    # --------------------------------------------------------

    heatmap = generate_gradcam(
        image_array,
        predicted_index
    )

    heatmap_array, colored_heatmap, overlay = (
    create_overlay(
        original_image,
        heatmap
    )
)

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    base_name = os.path.splitext(
        os.path.basename(image_path)
    )[0]

    original_path = os.path.join(
        OUTPUT_DIR,
        base_name + "_original.png"
    )

    heatmap_path = os.path.join(
        OUTPUT_DIR,
        base_name + "_gradcam.png"
    )

    overlay_path = os.path.join(
        OUTPUT_DIR,
        base_name + "_overlay.png"
    )

    original_image.save(
        original_path
    )

    Image.fromarray(
       colored_heatmap
    ).save(
    heatmap_path
    )

    Image.fromarray(
        overlay
    ).save(
        overlay_path
    )

    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    print("\n")
    print("=" * 65)
    print("              HISTOPATHOLOGY ANALYSIS")
    print("=" * 65)

    print("\nPredicted Class:")
    print(
        CLASS_LABELS[predicted_class]
    )

    print(
        f"\nConfidence: {confidence:.2f}%"
    )

    print("\nClass Probabilities:")
    print("-" * 45)

    for class_name, probability in zip(
        CLASS_NAMES,
        predictions
    ):

        print(
            f"{CLASS_LABELS[class_name]:35s}"
            f"{probability * 100:7.2f}%"
        )

    print("\n")
    print("Grad-CAM:")
    print("The highlighted regions show areas")
    print("that contributed to the prediction.")

    print("\nSaved files:")

    print(
        "Original :",
        original_path
    )

    print(
        "Heatmap  :",
        heatmap_path
    )

    print(
        "Overlay  :",
        overlay_path
    )

    print("=" * 65)

    print(
        "\nNOTE: This is an AI research prototype "
        "and not a clinical diagnostic system."
    )


# ============================================================
# ASK USER FOR IMAGE
# ============================================================

print("\n" + "=" * 65)
print("      DENSENET121 + CBAM + FOCAL LOSS DEMO")
print("=" * 65)

image_path = input(
    "\nEnter the path of the histopathology image:\n"
).strip().strip('"')

predict_image(
    image_path
)