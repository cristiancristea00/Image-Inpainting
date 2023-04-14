from typing import Any

import tensorflow as tf

from metrics_and_losses import LPIPS


class GatedConv2D(tf.keras.layers.Layer):
    def __init__(self, filters: int, kernel_size: int | tuple[int, int], strides: int | tuple[int, int] = 1, padding: str = 'same',
                 dilation_rate: int | tuple[int, int] = 1, activation: str | tf.keras.layers.Layer = None, kernel_initializer: str = 'he_normal',
                 bias_initializer: str = 'he_normal', kernel_regularizer: tf.keras.regularizers.Regularizer = None,
                 bias_regularizer: tf.keras.regularizers.Regularizer = None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.dilation_rate = dilation_rate
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer

        if isinstance(activation, str) or activation is None:
            self.activation = tf.keras.layers.Activation(activation)
        else:
            self.activation = activation

        self.conv_layer = tf.keras.layers.Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            dilation_rate=self.dilation_rate,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer
        )

        self.gate_layer = tf.keras.layers.Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            dilation_rate=self.dilation_rate,
            activation='sigmoid',
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer
        )

    def call(self, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        conv_output = self.activation(self.conv_layer(inputs))
        gate_output = self.gate_layer(inputs)
        outputs = conv_output * gate_output

        return outputs

    def compute_output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape:
        return self.conv_layer.compute_output_shape(input_shape)

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()

        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'dilation_rate': self.dilation_rate,
            'activation': self.activation,
            'kernel_initializer': self.kernel_initializer,
            'bias_initializer': self.bias_initializer,
            'kernel_regularizer': self.kernel_regularizer,
            'bias_regularizer': self.bias_regularizer
        })

        return config


if __name__ == '__main__':
    image = tf.random.uniform((1, 200, 200, 3))

    lpips = LPIPS(input_shape=(200, 200), all_layers=True)

    loss = lpips(image, image)
    print(loss)
