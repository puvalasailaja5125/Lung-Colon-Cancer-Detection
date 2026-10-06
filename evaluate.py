# import tensorflow as tf

# TEST_DIR = r"dataset\LUNG_COLON_IMAGE_SET\Test Set"

# IMAGE_SIZE = (224, 224)
# BATCH_SIZE = 16

# test_dataset = tf.keras.utils.image_dataset_from_directory(
#     TEST_DIR,
#     image_size=IMAGE_SIZE,
#     batch_size=BATCH_SIZE,
#     shuffle=False
# )

# print("Classes:", test_dataset.class_names)
# print(
#     "Test batches:",
#     tf.data.experimental.cardinality(test_dataset).numpy()
# )

# import tensorflow as tf
# from tensorflow.keras.applications.densenet import preprocess_input

# TEST_DIR = r"dataset\LUNG_COLON_IMAGE_SET\Test Set"

# IMAGE_SIZE = (224, 224)
# BATCH_SIZE = 16

# test_dataset = tf.keras.utils.image_dataset_from_directory(
#     TEST_DIR,
#     image_size=IMAGE_SIZE,
#     batch_size=BATCH_SIZE,
#     shuffle=False
# )

# class_names = test_dataset.class_names


# def preprocess_test(images, labels):
#     images = tf.cast(images, tf.float32)
#     images = preprocess_input(images)
#     return images, labels


# test_dataset = test_dataset.map(
#     preprocess_test,
#     num_parallel_calls=tf.data.AUTOTUNE
# )

# test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

# print("Classes:", class_names)
# print(
#     "Test batches:",
#     tf.data.experimental.cardinality(test_dataset).numpy()
# )
# MODEL_PATH = r"models\densenet121_baseline.keras"

# model = tf.keras.models.load_model(MODEL_PATH)

# print("Baseline model loaded successfully")

# test_loss, test_accuracy = model.evaluate(test_dataset)

# import numpy as np
# from sklearn.metrics import classification_report, confusion_matrix

# y_true = []
# y_pred = []

# for images, labels in test_dataset:
#     predictions = model.predict(images, verbose=0)

#     predicted_classes = np.argmax(predictions, axis=1)

#     y_true.extend(labels.numpy())
#     y_pred.extend(predicted_classes)

# y_true = np.array(y_true)
# y_pred = np.array(y_pred)

# print("\nClassification Report:")
# print(
#     classification_report(
#         y_true,
#         y_pred,
#         target_names=class_names
#     )
# )

# print("\nConfusion Matrix:")
# print(confusion_matrix(y_true, y_pred))

# # print("Test Loss:", test_loss)
# # print("Test Accuracy:", test_accuracy)

# ".venv313\Scripts\Activate.ps1" for getting in .venv313 environment


#-------------------------------------------DenseNet + CBAM--------------------------------------------------

# import tensorflow as tf
# import numpy as np
# from tensorflow.keras.applications.densenet import preprocess_input
# from cbam import CBAM
# from cbam import CBAM, ChannelAttention, SpatialAttention
# from sklearn.metrics import classification_report, confusion_matrix


# # --------------------------------------------------
# # 1. Test dataset
# # --------------------------------------------------

# TEST_DIR = r"dataset\LUNG_COLON_IMAGE_SET\Test Set"

# IMAGE_SIZE = (224, 224)
# BATCH_SIZE = 16


# test_dataset = tf.keras.utils.image_dataset_from_directory(
#     TEST_DIR,
#     image_size=IMAGE_SIZE,
#     batch_size=BATCH_SIZE,
#     shuffle=False
# )

# class_names = test_dataset.class_names


# # --------------------------------------------------
# # 2. Preprocessing
# # --------------------------------------------------

# def preprocess_test(images, labels):

#     images = tf.cast(images, tf.float32)

#     images = preprocess_input(images)

#     return images, labels


# test_dataset = test_dataset.map(
#     preprocess_test,
#     num_parallel_calls=tf.data.AUTOTUNE
# )

# test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)


# print("Classes:", class_names)

# print(
#     "Test batches:",
#     tf.data.experimental.cardinality(test_dataset).numpy()
# )


# # --------------------------------------------------
# # 3. Load CBAM model
# # --------------------------------------------------

# MODEL_PATH = r"models\densenet121_cbam.keras"

# model = tf.keras.models.load_model(
#     MODEL_PATH,
#     custom_objects={
#         "CBAM": CBAM,
#         "ChannelAttention": ChannelAttention,
#         "SpatialAttention": SpatialAttention
#     }
# )
# print("CBAM model loaded successfully")


# # --------------------------------------------------
# # 4. Evaluate model
# # --------------------------------------------------

# test_loss, test_accuracy = model.evaluate(test_dataset)

# print("\nTest Loss:", test_loss)
# print("Test Accuracy:", test_accuracy)


# # --------------------------------------------------
# # 5. Generate predictions
# # --------------------------------------------------

# y_true = []
# y_pred = []


# for images, labels in test_dataset:

#     predictions = model.predict(
#         images,
#         verbose=0
#     )

#     predicted_classes = np.argmax(
#         predictions,
#         axis=1
#     )

#     y_true.extend(labels.numpy())
#     y_pred.extend(predicted_classes)


# y_true = np.array(y_true)
# y_pred = np.array(y_pred)


# # --------------------------------------------------
# # 6. Classification report
# # --------------------------------------------------

# print("\nClassification Report:")

# print(
#     classification_report(
#         y_true,
#         y_pred,
#         target_names=class_names
#     )
# )


# # --------------------------------------------------
# # 7. Confusion matrix
# # --------------------------------------------------

# print("\nConfusion Matrix:")

# print(
#     confusion_matrix(
#         y_true,
#         y_pred
#     )
# )
# ".venv313\Scripts\Activate.ps1" for getting in .venv313 environment

#----------------------------------------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------DenseNet + CBAM + focal loss--------------------------------------------------    

import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.densenet import preprocess_input

from cbam import (
    CBAM,
    ChannelAttention,
    SpatialAttention
)

from focal_loss import SparseCategoricalFocalLoss


# ==============================
# 1. Test dataset
# ==============================

TEST_DIR = r"dataset\LUNG_COLON_IMAGE_SET\Test Set"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16


test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = test_dataset.class_names

print("Classes:", class_names)


# ==============================
# 2. Preprocessing
# ==============================

def preprocess_test(images, labels):

    images = tf.cast(images, tf.float32)

    images = preprocess_input(images)

    return images, labels


test_dataset = test_dataset.map(
    preprocess_test,
    num_parallel_calls=tf.data.AUTOTUNE
)

test_dataset = test_dataset.prefetch(
    tf.data.AUTOTUNE
)


print(
    "Test batches:",
    tf.data.experimental.cardinality(test_dataset).numpy()
)


# ==============================
# 3. Load Focal Loss model
# ==============================

MODEL_PATH = r"models\densenet121_cbam_focal.keras"


model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "CBAM": CBAM,
        "ChannelAttention": ChannelAttention,
        "SpatialAttention": SpatialAttention,
        "SparseCategoricalFocalLoss": SparseCategoricalFocalLoss
    }
)


print("Focal model loaded successfully")


# ==============================
# 4. Evaluate model
# ==============================

test_loss, test_accuracy = model.evaluate(
    test_dataset
)


print("\nTest Loss:", test_loss)
print("Test Accuracy:", test_accuracy)


# ==============================
# 5. Generate predictions
# ==============================

num_classes = len(class_names)

confusion_matrix = np.zeros(
    (num_classes, num_classes),
    dtype=np.int64
)

correct = 0
total = 0


for images, labels in test_dataset:

    predictions = model.predict(
        images,
        verbose=0
    )

    predicted_classes = np.argmax(
        predictions,
        axis=1
    )

    true_classes = labels.numpy()

    correct += np.sum(
        predicted_classes == true_classes
    )

    total += len(true_classes)

    for true_class, predicted_class in zip(
        true_classes,
        predicted_classes
    ):

        confusion_matrix[
            true_class,
            predicted_class
        ] += 1


# ==============================
# 6. Accuracy from predictions
# ==============================

accuracy = correct / total

print("\nAccuracy from predictions:", accuracy)


# ==============================
# 7. Confusion Matrix
# ==============================

print("\nConfusion Matrix:")

print(confusion_matrix)


# ==============================
# 8. Precision / Recall / F1
# ==============================

print("\nClassification Report:")

precisions = []
recalls = []
f1_scores = []
supports = []


for i in range(num_classes):

    true_positive = confusion_matrix[i, i]

    false_positive = (
        np.sum(confusion_matrix[:, i])
        - true_positive
    )

    false_negative = (
        np.sum(confusion_matrix[i, :])
        - true_positive
    )

    support = np.sum(
        confusion_matrix[i, :]
    )

    # Precision
    if true_positive + false_positive > 0:
        precision = (
            true_positive
            / (true_positive + false_positive)
        )
    else:
        precision = 0.0

    # Recall
    if true_positive + false_negative > 0:
        recall = (
            true_positive
            / (true_positive + false_negative)
        )
    else:
        recall = 0.0

    # F1
    if precision + recall > 0:
        f1 = (
            2 * precision * recall
            / (precision + recall)
        )
    else:
        f1 = 0.0

    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)
    supports.append(support)

    print(
        f"{class_names[i]:15s} "
        f"Precision: {precision:.2f} "
        f"Recall: {recall:.2f} "
        f"F1: {f1:.2f} "
        f"Support: {support}"
    )


# ==============================
# 9. Macro average
# ==============================

macro_precision = np.mean(precisions)
macro_recall = np.mean(recalls)
macro_f1 = np.mean(f1_scores)


# ==============================
# 10. Weighted average
# ==============================

total_support = np.sum(supports)

weighted_precision = (
    np.sum(
        np.array(precisions)
        * np.array(supports)
    )
    / total_support
)

weighted_recall = (
    np.sum(
        np.array(recalls)
        * np.array(supports)
    )
    / total_support
)

weighted_f1 = (
    np.sum(
        np.array(f1_scores)
        * np.array(supports)
    )
    / total_support
)


print("\nMacro Average:")
print(
    f"Precision: {macro_precision:.2f}"
)
print(
    f"Recall: {macro_recall:.2f}"
)
print(
    f"F1: {macro_f1:.2f}"
)


print("\nWeighted Average:")
print(
    f"Precision: {weighted_precision:.2f}"
)
print(
    f"Recall: {weighted_recall:.2f}"
)
print(
    f"F1: {weighted_f1:.2f}"
)