from __future__ import annotations

from enum import Enum
from typing import Any, Final, Literal, TypeAlias

import tensorflow as tf

LEAKY_RELU_ALPHA: Final[float] = 0.2
DROPOUT_RATE: Final[float] = 0.05


class UpsampleMode(Enum):
    DECONVOLUTION = 'deconvolution'
    BILINEAR = 'bilinear'
    NEAREST_NEIGHBOR = 'nearest'


UpsampleModeType: TypeAlias = Literal[
    UpsampleMode.DECONVOLUTION,
    UpsampleMode.BILINEAR,
    UpsampleMode.NEAREST_NEIGHBOR
]


class DownsampleMode(Enum):
    STRIDE = 'stride'
    MAX_POOL = 'max_pool'
    AVERAGE_POOL = 'average_pool'


DownsampleModeType: TypeAlias = Literal[
    DownsampleMode.STRIDE,
    DownsampleMode.MAX_POOL,
    DownsampleMode.AVERAGE_POOL
]


class PadMode(Enum):
    CONSTANT = 'CONSTANT'
    REFLECT = 'REFLECT'
    SYMMETRIC = 'SYMMETRIC'


PadModeType: TypeAlias = Literal[
    PadMode.CONSTANT,
    PadMode.REFLECT,
    PadMode.SYMMETRIC
]


UNetConfig: TypeAlias = tuple[int, int, int, int, int, int]


class Padding2D(tf.keras.layers.Layer):

    def __init__(self, padding: int | tuple[int, int] = 1, pad_mode: PadModeType = PadMode.CONSTANT, *args, **kwargs) -> None:

        super().__init__(*args, **kwargs)

        self.pad_mode = pad_mode

        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()

        config.update({
            'padding': self.padding,
            'pad_mode': self.pad_mode
        })

        return config

    def compute_output_shape(self, input_shape) -> tuple[int, int, int, int]:

        batch = input_shape[0]
        height = input_shape[1] + 2 * self.padding[0]
        width = input_shape[2] + 2 * self.padding[1]
        chanels = input_shape[3]
        return batch, height, width, chanels

    def call(self, inputs: tf.Tensor) -> tf.Tensor:

        width, height = self.padding
        paddings = ([0, 0], [height, height], [width, width], [0, 0])
        result = tf.pad(tensor=inputs, paddings=paddings, mode=self.pad_mode.value, constant_values=0)
        return result


class UNet2DConvolutionBlock(tf.keras.layers.Layer):

    def __init__(self, filters: int, kernel_size: int, stride: int = 1, pad_mode: PadModeType = PadMode.CONSTANT, *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad_mode = pad_mode

        self.pad = Padding2D(padding=(kernel_size - 1) // 2, pad_mode=pad_mode)
        self.conv = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=stride,
            padding='valid',
            kernel_initializer='he_normal',
            bias_initializer='he_normal'
        )
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.activation = tf.keras.layers.LeakyReLU(alpha=LEAKY_RELU_ALPHA)

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()

        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'pad_mode': self.pad_mode
        })

        return config

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        padded = self.pad(inputs)
        convolved = self.conv(padded)
        normed = self.batch_norm(convolved)
        activated = self.activation(normed)
        outputs = tf.keras.layers.Dropout(DROPOUT_RATE)(activated)
        return outputs


class UNetDownLayer(tf.keras.layers.Layer):

    def __init__(self, filters: int, kernel_size: int, stride: int = 1, downsample_mode: DownsampleModeType = DownsampleMode.MAX_POOL,
                 pad_mode: PadModeType = PadMode.CONSTANT, *args, **kwargs) -> None:

        super().__init__(*args, **kwargs)

        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.downsample_mode = downsample_mode
        self.pad_mode = pad_mode

        if downsample_mode is not DownsampleMode.STRIDE and stride != 1:

            if downsample_mode is DownsampleMode.MAX_POOL:

                self.downsample = tf.keras.layers.MaxPool2D(pool_size=stride, strides=1)

            elif downsample_mode is DownsampleMode.AVERAGE_POOL:

                self.downsample = tf.keras.layers.AveragePooling2D(pool_size=stride, strides=1)

        self.conv = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=1,
            padding='same',
            kernel_initializer='he_normal',
            bias_initializer='he_normal'
        )
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.activation = tf.keras.layers.LeakyReLU(alpha=LEAKY_RELU_ALPHA)
        self.unet_conv_block = UNet2DConvolutionBlock(filters=filters, kernel_size=kernel_size, stride=stride, pad_mode=pad_mode)

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()

        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'downsample_mode': self.downsample_mode,
            'pad_mode': self.pad_mode
        })

        return config

    def call(self, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:

        convolved = self.conv(inputs)
        downsampled = self.downsample(convolved) if hasattr(self, 'downsample') else convolved
        normed = self.batch_norm(downsampled)
        activated = self.activation(normed)
        outputs = self.unet_conv_block(activated)

        return outputs


class UNetUpLayer(tf.keras.layers.Layer):

    def __init__(self, filters: int, kernel_size: int, upsample_mode: UpsampleModeType = UpsampleMode.DECONVOLUTION,
                 pad_mode: PadModeType = PadMode.CONSTANT, *args, **kwargs) -> None:

        super().__init__(*args, **kwargs)

        self.filters = filters
        self.kernel_size = kernel_size
        self.upsample_mode = upsample_mode
        self.pad_mode = pad_mode

        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.unet_conv_block_1 = UNet2DConvolutionBlock(filters=filters, kernel_size=kernel_size, stride=1, pad_mode=pad_mode)
        self.unet_conv_block_2 = UNet2DConvolutionBlock(filters=filters, kernel_size=1, stride=1, pad_mode=pad_mode)

        if upsample_mode is UpsampleMode.DECONVOLUTION:

            self.upsample = tf.keras.layers.Conv2DTranspose(
                filters=filters,
                kernel_size=2,
                strides=2,
                padding='same',
                kernel_initializer='he_normal',
                bias_initializer='he_normal'
            )

        elif upsample_mode is UpsampleMode.BILINEAR or upsample_mode is UpsampleMode.NEAREST_NEIGHBOR:

            self.upsample = tf.keras.layers.UpSampling2D(size=2, interpolation=upsample_mode.value)

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()

        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'upsample_mode': self.upsample_mode,
            'pad_mode': self.pad_mode
        })
        return config

    def call(self, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:

        normed = self.batch_norm(inputs)
        convolved1 = self.unet_conv_block_1(normed)
        convolved2 = self.unet_conv_block_2(convolved1)
        outputs = self.upsample(convolved2)

        return outputs


class UNet(tf.keras.Model):

    def __init__(self, filters: UNetConfig, kernels: UNetConfig, skip_filters: UNetConfig, skip_kernels: UNetConfig, output_channels: int = 3,
                 downsample_mode: DownsampleModeType = DownsampleMode.MAX_POOL, pad_mode: PadModeType = PadMode.CONSTANT,
                 upsample_mode: UpsampleModeType = UpsampleMode.DECONVOLUTION, *args, **kwargs) -> None:

        super().__init__(*args, **kwargs)

        self.filters = filters
        self.kernels = kernels
        self.skip_filters = skip_filters
        self.skip_kernels = skip_kernels
        self.output_channels = output_channels
        self.downsample_mode = downsample_mode
        self.pad_mode = pad_mode
        self.upsample_mode = upsample_mode

        self.down_layers: list[tf.keras.layers.Layer] = []

        for idx, (filter_num, kernel) in enumerate(zip(filters, kernels)):
            self.down_layers.append(
                UNetDownLayer(filters=filter_num, kernel_size=kernel, stride=2, downsample_mode=downsample_mode, pad_mode=pad_mode,
                              name=f'unet_down_{idx}')
            )

        self.up_layers: list[tf.keras.layers.Layer] = []

        for idx, (filter_num, kernel) in enumerate(zip(filters, kernels)):
            self.up_layers.append(
                UNetUpLayer(filters=filter_num, kernel_size=kernel, upsample_mode=upsample_mode, pad_mode=pad_mode, name=f'unet_up_{idx}')
            )

        self.skip_layers: list[tf.keras.layers.Layer] = []

        for idx, (filter_num, kernel) in enumerate(zip(skip_filters, skip_kernels)):
            self.skip_layers.append(
                UNet2DConvolutionBlock(filters=filter_num, kernel_size=kernel, pad_mode=pad_mode, name=f'unet_skip_{idx}')
            )

        self.conv = tf.keras.layers.Conv2D(
            filters=output_channels,
            kernel_size=1,
            strides=1,
            padding='same',
            kernel_initializer='he_normal',
            bias_initializer='he_normal'
        )

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()

        config.update({
            'filters': self.filters,
            'kernels': self.kernels,
            'skip_filters': self.skip_filters,
            'skip_kernels': self.skip_kernels,
            'output_channels': self.output_channels,
            'downsample_mode': self.downsample_mode,
            'pad_mode': self.pad_mode,
            'upsample_mode': self.upsample_mode
        })

        return config

    def call(self, inputs: tf.Tensor) -> tf.Tensor:

        down_computed_0 = self.down_layers[0](inputs)
        down_computed_1 = self.down_layers[1](down_computed_0)
        down_computed_2 = self.down_layers[2](down_computed_1)
        down_computed_3 = self.down_layers[3](down_computed_2)
        down_computed_4 = self.down_layers[4](down_computed_3)
        down_computed_5 = self.down_layers[5](down_computed_4)

        skip_computed_0 = self.skip_layers[0](down_computed_0)
        skip_computed_1 = self.skip_layers[1](down_computed_1)
        skip_computed_2 = self.skip_layers[2](down_computed_2)
        skip_computed_3 = self.skip_layers[3](down_computed_3)
        skip_computed_4 = self.skip_layers[4](down_computed_4)
        skip_computed_5 = self.skip_layers[5](down_computed_5)

        up_computed_5 = tf.keras.layers.Concatenate()([self.up_layers[5](skip_computed_5), skip_computed_4])
        up_computed_4 = tf.keras.layers.Concatenate()([self.up_layers[4](up_computed_5), skip_computed_3])
        up_computed_3 = tf.keras.layers.Concatenate()([self.up_layers[3](up_computed_4), skip_computed_2])
        up_computed_2 = tf.keras.layers.Concatenate()([self.up_layers[2](up_computed_3), skip_computed_1])
        up_computed_1 = tf.keras.layers.Concatenate()([self.up_layers[1](up_computed_2), skip_computed_0])
        up_computed_0 = self.up_layers[0](up_computed_1)

        return self.conv(up_computed_0)

    @property
    def name(self) -> Literal['UNet']:
        return 'UNet'

    def build_model(self, input_shape: tuple[int, int, int]) -> tf.keras.Model:

        inputs = tf.keras.Input(shape=input_shape)
        return tf.keras.Model(inputs=inputs, outputs=self.call(inputs), name=self.name)
