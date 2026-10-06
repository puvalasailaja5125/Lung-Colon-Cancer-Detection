import tensorflow as tf


class ChannelAttention(tf.keras.layers.Layer):

    def __init__(self, reduction_ratio=8, **kwargs):
        super().__init__(**kwargs)
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):

        channels = input_shape[-1]

        self.shared_dense_1 = tf.keras.layers.Dense(
            channels // self.reduction_ratio,
            activation="relu"
        )

        self.shared_dense_2 = tf.keras.layers.Dense(
            channels,
            activation=None
        )

    def call(self, inputs):

        # Average pooling across spatial dimensions
        avg_pool = tf.reduce_mean(
            inputs,
            axis=[1, 2]
        )

        # Maximum pooling across spatial dimensions
        max_pool = tf.reduce_max(
            inputs,
            axis=[1, 2]
        )

        # Shared MLP
        avg_attention = self.shared_dense_2(
            self.shared_dense_1(avg_pool)
        )

        max_attention = self.shared_dense_2(
            self.shared_dense_1(max_pool)
        )

        # Combine both attention signals
        attention = avg_attention + max_attention

        # Convert values to 0-1
        attention = tf.nn.sigmoid(attention)

        # Restore spatial dimensions
        attention = tf.reshape(
            attention,
            [-1, 1, 1, tf.shape(attention)[-1]]
        )

        # Apply attention
        return inputs * attention

class SpatialAttention(tf.keras.layers.Layer):

    def __init__(self, kernel_size=7, **kwargs):
        super().__init__(**kwargs)

        self.kernel_size = kernel_size

        self.conv = tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=kernel_size,
            padding="same",
            activation=None
        )
    
    def call(self, inputs):

        # Average across channels
        avg_pool = tf.reduce_mean(
            inputs,
            axis=-1,
            keepdims=True
        )

        # Maximum across channels
        max_pool = tf.reduce_max(
            inputs,
            axis=-1,
            keepdims=True
        )

        # Combine the two spatial descriptions
        combined = tf.concat(
            [avg_pool, max_pool],
            axis=-1
        )

        # Generate spatial attention map
        attention = self.conv(combined)

        # Convert values to 0-1
        attention = tf.nn.sigmoid(attention)

        # Apply spatial attention
        return inputs * attention

class CBAM(tf.keras.layers.Layer):

    def __init__(self, reduction_ratio=8, kernel_size=7, **kwargs):
        super().__init__(**kwargs)
    
        self.reduction_ratio = reduction_ratio
        self.kernel_size = kernel_size
    
        self.channel_attention = ChannelAttention(
            reduction_ratio=reduction_ratio
        )
    
        self.spatial_attention = SpatialAttention(
            kernel_size=kernel_size
        )

    def call(self, inputs):

        x = self.channel_attention(inputs)

        x = self.spatial_attention(x)

        return x

if __name__ == "__main__":

    test_input = tf.random.normal(
        (2, 7, 7, 1024)
    )

    cbam = CBAM()

    output = cbam(test_input)

    print("Input shape:", test_input.shape)
    print("CBAM output shape:", output.shape)