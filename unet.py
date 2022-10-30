from __future__ import annotations
from typing import Final

import tensorflow as tf
from enum import Enum

LEAKY_RELU_ALPHA: Final[float] = 0.01


class UpsampleMode(Enum):
    DECONVOLUTION = 'deconvolution'
    BILINEAR = 'bilinear'
    NEAREST_NEIGHBOR = 'nearest'


class DownsampleMode(Enum):
    STRIDE = 'stride'
    MAX_POOL = 'max_pool'
    AVERAGE_POOL = 'average_pool'


class PadMode(Enum):
    CONSTANT = 'CONSTANT'
    REFLECT = 'REFLECT'
    SYMMETRIC = 'SYMMETRIC'


class Padding2D(tf.keras.layers.Layer):

    def __init__(self, padding: int | tuple[int, int] = 1, pad_mode: PadMode = PadMode.CONSTANT, *args, **kwargs) -> None:

        super().__init__(*args, **kwargs)
        self.pad_mode = pad_mode
        self.input_spec = [tf.keras.layers.InputSpec(ndim=4)]

        if type(padding) is int:
            self.padding = (padding, padding)
        else:
            self.padding = padding

    def compute_output_shape(self, shape) -> tuple[int, int, int, int]:

        return shape[0], shape[1] + 2 * self.padding[0], shape[2] + 2 * self.padding[1], shape[3]

    def call(self, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:

        width_pad, height_pad = self.padding
        return tf.pad(tensor=inputs, paddings=[[0, 0], [height_pad, height_pad], [width_pad, width_pad], [0, 0]], mode=self.pad_mode.value,
                      constant_values=0)


class UNet2DConvolutionBlock(tf.keras.layers.Layer):

    def __init__(self, filters: int, kernel_size: int, stride: int = 1, use_bias: bool = True, pad_mode: PadMode = PadMode.CONSTANT, *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.pad = Padding2D(padding=(kernel_size - 1) // 2, pad_mode=pad_mode)
        self.convolve = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=stride, padding='valid', use_bias=use_bias)
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.activation = tf.keras.layers.LeakyReLU(alpha=LEAKY_RELU_ALPHA)

    def call(self, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        inputs = self.pad(inputs)
        inputs = self.convolve(inputs)
        inputs = self.batch_norm(inputs)
        inputs = self.activation(inputs)
        return inputs


class UNetDownLayer(tf.keras.layers.Layer):

    def __init__(self, filters: int, kernel_size: int, stride: int = 1, use_bias: bool = True,
                 downsample_mode: DownsampleMode = DownsampleMode.MAX_POOL,
                 pad_mode: PadMode = PadMode.CONSTANT, *args, **kwargs) -> None:

        super().__init__(*args, **kwargs)

        if downsample_mode is not DownsampleMode.STRIDE and stride != 1:

            if downsample_mode is DownsampleMode.MAX_POOL:

                self.downsample = tf.keras.layers.MaxPool2D(pool_size=stride, strides=1)

            elif downsample_mode is DownsampleMode.AVERAGE_POOL:

                self.downsample = tf.keras.layers.AveragePooling2D(pool_size=stride, strides=1)

        self.convolve = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=1, padding='same', use_bias=use_bias)
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.activation = tf.keras.layers.LeakyReLU(alpha=LEAKY_RELU_ALPHA)
        self.unet_conv_block = UNet2DConvolutionBlock(filters=filters, kernel_size=kernel_size, stride=stride, use_bias=use_bias, pad_mode=pad_mode)

    def call(self, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:

        inputs = self.convolve(inputs)
        inputs = self.downsample(inputs) if hasattr(self, 'downsample') else inputs
        inputs = self.batch_norm(inputs)
        inputs = self.activation(inputs)
        inputs = self.unet_conv_block(inputs)

        return inputs


class UNetUpLayer(tf.keras.layers.Layer):

    def __init__(self, filters: int, kernel_size: int, use_bias: bool = True, upsample_mode: UpsampleMode = UpsampleMode.DECONVOLUTION,
                 pad_mode: PadMode = PadMode.CONSTANT, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.unet_conv_block_1 = UNet2DConvolutionBlock(filters=filters, kernel_size=kernel_size, stride=1, use_bias=use_bias, pad_mode=pad_mode)
        self.unet_conv_block_2 = UNet2DConvolutionBlock(filters=filters, kernel_size=1, stride=1, use_bias=use_bias, pad_mode=pad_mode)

        if upsample_mode is UpsampleMode.DECONVOLUTION:

            self.upsample = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=2, strides=2, padding='same', use_bias=use_bias)

        elif upsample_mode is UpsampleMode.BILINEAR or upsample_mode is UpsampleMode.NEAREST_NEIGHBOR:

            self.upsample = tf.keras.layers.UpSampling2D(size=2, interpolation=upsample_mode.value)

    def call(self, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:

        inputs = self.batch_norm(inputs)
        inputs = self.unet_conv_block_1(inputs)
        inputs = self.unet_conv_block_2(inputs)
        inputs = self.upsample(inputs)

        return inputs


class UNet(tf.keras.Model):

    def __init__(self, filters: list[int], kernels: list[int], output_channels: int = 3, upsample_mode: UpsampleMode = UpsampleMode.DECONVOLUTION,
                 downsample_mode: DownsampleMode = DownsampleMode.MAX_POOL, pad_mode: PadMode = PadMode.CONSTANT, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.filters = filters

        self.down_layers: list[tf.keras.layers.Layer] = []

        for i, (filter_num, kernel) in enumerate(zip(filters, kernels)):
            self.down_layers.append(
                UNetDownLayer(filters=filter_num, kernel_size=kernel, stride=2, downsample_mode=downsample_mode, pad_mode=pad_mode, name=f'down_{i}'))

        self.up_layers: list[tf.keras.layers.Layer] = []

        for i, (filter_num, kernel) in enumerate(zip(filters[::-1], kernels[::-1])):
            self.up_layers.append(
                UNetUpLayer(filters=filter_num, kernel_size=kernel, upsample_mode=upsample_mode, pad_mode=pad_mode, name=f'up_{i}'))

        self.convolve = tf.keras.layers.Conv2D(filters=output_channels, kernel_size=1, strides=1, padding='same', use_bias=True)

    def call(self, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:

        for down_layer in self.down_layers:
            inputs = down_layer(inputs)

        for up_layer in self.up_layers:
            inputs = up_layer(inputs)

        return self.convolve(inputs)

    def summary(self, *args, **kwargs) -> None:
        inputs = tf.keras.layers.Input(shape=(512, 512, 3))
        model = tf.keras.Model(inputs=inputs, outputs=self.call(inputs))
        return model.summary()
