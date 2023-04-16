from __future__ import annotations

from enum import Enum
from typing import Any, TypeAlias, Literal

import tensorflow as tf


class UpsampleMode(Enum):
    """
    Upsample mode for UNet.
    """

    DECONVOLUTION = 'deconvolution'
    BILINEAR = 'bilinear'
    NEAREST_NEIGHBOR = 'nearest'


UpsampleModeType: TypeAlias = Literal[
    UpsampleMode.DECONVOLUTION,
    UpsampleMode.BILINEAR,
    UpsampleMode.NEAREST_NEIGHBOR
]


class DownsampleMode(Enum):
    """
    Downsample mode for UNet.
    """

    STRIDE = 'stride'
    MAX_POOL = 'max_pool'
    AVERAGE_POOL = 'average_pool'


DownsampleModeType: TypeAlias = Literal[
    DownsampleMode.STRIDE,
    DownsampleMode.MAX_POOL,
    DownsampleMode.AVERAGE_POOL
]


class PadMode(Enum):
    """
    Padding mode for UNet.
    """

    CONSTANT = 'CONSTANT'
    REFLECT = 'REFLECT'
    SYMMETRIC = 'SYMMETRIC'


PadModeType: TypeAlias = Literal[
    PadMode.CONSTANT,
    PadMode.REFLECT,
    PadMode.SYMMETRIC
]


class Padding2D(tf.keras.layers.Layer):
    """
    Padding layer for 2D tensors.
    """

    def __init__(self, padding: int | tuple[int, int] = 1, pad_mode: PadModeType = PadMode.REFLECT, **kwargs) -> None:
        """
        Initialize Padding Layer.

        Args:
            padding (int | tuple[int, int], optional): Padding size. Defaults to 1
            pad_mode (PadModeType, optional): Padding mode. Defaults to PadMode.REFLECT
        """
        super().__init__(**kwargs)

        self.pad_mode = pad_mode

        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding

    def get_config(self) -> dict[str, Any]:
        """
        Get layer configuration.

        Returns:
            dict[str, Any]: Layer configuration
        """

        config = super().get_config()

        config.update({
            'padding': self.padding,
            'pad_mode': self.pad_mode
        })

        return config

    def compute_output_shape(self, input_shape: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        """
        Compute output shape.

        Args:
            input_shape (tuple[int, int, int, int]): Input shape

        Returns:
            tuple[int, int, int, int]: Output shape
        """

        batch = input_shape[0]
        height = input_shape[1] + 2 * self.padding[0]
        width = input_shape[2] + 2 * self.padding[1]
        channels = input_shape[3]

        return batch, height, width, channels

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Forward pass.

        Args:
            inputs (tf.Tensor): Input tensor

        Returns:
            tf.Tensor: Output tensor
        """

        width, height = self.padding
        paddings = ([0, 0], [height, height], [width, width], [0, 0])
        result = tf.pad(tensor=inputs, paddings=paddings, mode=self.pad_mode.value, constant_values=0)
        return result


class GatedConv2D(tf.keras.layers.Layer):
    """
    Gated Convolutional Layer
    """

    def __init__(self, filters: int, kernel_size: int | tuple[int, int], strides: int | tuple[int, int] = 1, padding: str = 'same',
                 dilation_rate: int | tuple[int, int] = 1, activation: str | tf.keras.layers.Layer = None, kernel_initializer: str = 'he_normal',
                 bias_initializer: str = 'he_normal', kernel_regularizer: tf.keras.regularizers.Regularizer = None,
                 bias_regularizer: tf.keras.regularizers.Regularizer = None, **kwargs) -> None:
        """
        Initialize Gated Convolutional Layer

        Args:
            filters (int): Number of filters
            kernel_size (int | tuple[int, int]): Kernel size
            strides (int | tuple[int, int], optional): Strides. Defaults to 1.
            padding (str, optional): Padding. Defaults to 'same'.
            dilation_rate (int | tuple[int, int], optional): Dilation rate. Defaults to 1.
            activation (str | tf.keras.layers.Layer, optional): Activation function. Defaults to None.
            kernel_initializer (str, optional): Kernel initializer. Defaults to 'he_normal'.
            bias_initializer (str, optional): Bias initializer. Defaults to 'he_normal'.
            kernel_regularizer (tf.keras.regularizers.Regularizer, optional): Kernel regularizer. Defaults to None.
            bias_regularizer (tf.keras.regularizers.Regularizer, optional): Bias regularizer. Defaults to None.
        """

        super().__init__(**kwargs)

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
            bias_regularizer=self.bias_regularizer,
            name='normal_conv'
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
            bias_regularizer=self.bias_regularizer,
            name='gate_conv'
        )

    def call(self, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        """
        Forward pass.

        Args:
            inputs (tf.Tensor): Input tensor

        Returns:
            tf.Tensor: Output tensor
        """

        conv_output = self.activation(self.conv_layer(inputs))
        gate_output = self.gate_layer(inputs)
        outputs = conv_output * gate_output

        return outputs

    def get_config(self) -> dict[str, Any]:
        """
        Get layer configuration.

        Returns:
            dict[str, Any]: Layer configuration
        """

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
