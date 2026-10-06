# import tensorflow as tf


# # --------------------------------------------------
# # Configuration
# # --------------------------------------------------

# IMAGE_SIZE = (224, 224)
# NUM_CLASSES = 5


# # --------------------------------------------------
# # Build DenseNet121 Baseline
# # --------------------------------------------------

# def build_densenet_model():

#     # Pretrained DenseNet121
#     base_model = tf.keras.applications.DenseNet121(
#         include_top=False,
#         weights="imagenet",
#         input_shape=(224, 224, 3)
#     )

#     # Freeze DenseNet121 initially
#     base_model.trainable = False

#     # Create model
#     inputs = tf.keras.Input(shape=(224, 224, 3))

#     x = base_model(inputs, training=False)

#     # Convert feature maps into one feature vector
#     x = tf.keras.layers.GlobalAveragePooling2D()(x)

#     # Reduce overfitting
#     x = tf.keras.layers.Dropout(0.3)(x)

#     # Five-class output
#     outputs = tf.keras.layers.Dense(
#         NUM_CLASSES,
#         activation="softmax"
#     )(x)

#     model = tf.keras.Model(
#         inputs=inputs,
#         outputs=outputs,
#         name="DenseNet121_Baseline"
#     )

#     return model


# # --------------------------------------------------
# # Test model
# # --------------------------------------------------

# model = build_densenet_model()

# # model.summary()
# model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
#     loss="sparse_categorical_crossentropy",
#     metrics=["accuracy"]
# )

# print("Model compiled successfully")


# from dataset import train_dataset, validation_dataset

# # images, labels = next(iter(train_dataset))

# # predictions = model(images, training=False)

# # # print("Input shape:", images.shape)
# # # print("Prediction shape:", predictions.shape)
# # # print("First prediction:", predictions[0].numpy())
# # # print("First label:", labels[0].numpy())

# history = model.fit(
#     train_dataset,
#     validation_data=validation_dataset,
#     epochs=1
# )

# model.save("models/densenet121_baseline.keras")

# print("Baseline model saved successfully")

# import tensorflow as tf
# from cbam import CBAM

# IMAGE_SIZE = (224, 224)
# NUM_CLASSES = 5


# def build_densenet_cbam_model():

#     base_model = tf.keras.applications.DenseNet121(
#         include_top=False,
#         weights="imagenet",
#         input_shape=(224, 224, 3)
#     )

#     base_model.trainable = False

#     inputs = tf.keras.Input(
#         shape=(224, 224, 3)
#     )

#     x = base_model(
#         inputs,
#         training=False
#     )

#     x = CBAM()(x)

#     x = tf.keras.layers.GlobalAveragePooling2D()(x)

#     x = tf.keras.layers.Dropout(0.3)(x)

#     outputs = tf.keras.layers.Dense(
#         NUM_CLASSES,
#         activation="softmax"
#     )(x)

#     model = tf.keras.Model(
#         inputs=inputs,
#         outputs=outputs,
#         name="DenseNet121_CBAM"
#     )

#     return model


# model = build_densenet_cbam_model()

# # model.summary()

# # from dataset import train_dataset

# # images, labels = next(iter(train_dataset))

# # predictions = model(images, training=False)

# # # print("Input shape:", images.shape)
# # # print("Prediction shape:", predictions.shape)
# # # print("First prediction:", predictions[0].numpy())
# # # print("First label:", labels[0].numpy())

# model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
#     loss="sparse_categorical_crossentropy",
#     metrics=["accuracy"]
# )

# print("CBAM model compiled successfully")

# from dataset import train_dataset, validation_dataset

# history = model.fit(
#     train_dataset,
#     validation_data=validation_dataset,
#     epochs=1
# )

# model.save("models/densenet121_cbam.keras")

# print("CBAM model trained and saved successfully")



import tensorflow as tf
from cbam import CBAM
from focal_loss import SparseCategoricalFocalLoss

IMAGE_SIZE = (224, 224)
NUM_CLASSES = 5


def build_densenet_cbam_model():

    base_model = tf.keras.applications.DenseNet121(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3)
    )

    base_model.trainable = False

    inputs = tf.keras.Input(
        shape=(224, 224, 3)
    )

    x = base_model(
        inputs,
        training=False
    )

    x = CBAM()(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Dropout(0.3)(x)

    outputs = tf.keras.layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="DenseNet121_CBAM_Focal"
    )

    return model


model = build_densenet_cbam_model()

# # model.summary()

# from dataset import train_dataset

# images, labels = next(iter(train_dataset))

# predictions = model(images, training=False)

# print("Input shape:", images.shape)
# print("Prediction shape:", predictions.shape)
# print("First prediction:", predictions[0].numpy())
# print("First label:", labels[0].numpy())

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=SparseCategoricalFocalLoss(
        gamma=2.0,
        alpha=0.25
    ),
    metrics=["accuracy"]
)


# print("CBAM model compiled successfully")

from dataset import train_dataset, validation_dataset

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=1
)

model.save("models/densenet121_cbam_focal.keras")

print("CBAM model compiled with Focal Loss successfully")