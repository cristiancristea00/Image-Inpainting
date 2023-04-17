from __future__ import annotations

from pathlib import Path
from typing import Any, Final, TypeAlias

import tensorflow as tf

from layers import PadModeType, PadMode, Padding2D, DownsampleModeType, DownsampleMode, UpsampleModeType, UpsampleMode, GatedConv2D

LEAKY_RELU_ALPHA: Final[float] = 0.2
DROPOUT_RATE: Final[float] = 0.05


class UNet2DConvolutionBlock(tf.keras.layers.Layer):
    """
    UNet 2D convolution block.
    """

    def __init__(self, is_gated: bool, filters: int, kernel_size: int, stride: int = 1, dilation_rate: int = 2,
                 pad_mode: PadModeType = PadMode.REFLECT, **kwargs) -> None:
        """
        Initialize UNet 2D convolution block.

        Args:
            is_gated (bool): Apply gated convolution
            filters (int): Number of filters
            kernel_size (int): Kernel size
            stride (int, optional): Stride. Defaults to 1
            dilation_rate (int, optional): Dilation rate. Defaults to 2
            pad_mode (PadModeType, optional): Padding mode. Defaults to PadMode.REFLECT
        """

        super().__init__(**kwargs)

        self.is_gated = is_gated
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation_rate = dilation_rate
        self.pad_mode = pad_mode

        if self.is_gated:
            Conv2D = GatedConv2D
        else:
            Conv2D = tf.keras.layers.Conv2D

        padding = (self.kernel_size + (self.kernel_size - 1) * (self.dilation_rate - 1) - 1) // 2
        self.pad = Padding2D(padding=padding, pad_mode=self.pad_mode)

        self.conv = Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.stride,
            dilation_rate=self.dilation_rate,
            padding='valid',
            kernel_initializer='he_normal',
            bias_initializer='he_normal'
        )
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.activation = tf.keras.layers.LeakyReLU(alpha=LEAKY_RELU_ALPHA)

    def get_config(self) -> dict[str, Any]:
        """
        Get layer configuration.

        Returns:
            dict[str, Any]: Layer configuration
        """

        config = super().get_config()

        config.update({
            'is_gated': self.is_gated,
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'dilation_rate': self.dilation_rate,
            'pad_mode': self.pad_mode
        })

        return config

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Forward pass.

        Args:
            inputs (tf.Tensor): Input tensor

        Returns:
            tf.Tensor: Output tensor
        """

        padded = self.pad(inputs)
        convolved = self.conv(padded)
        normed = self.batch_norm(convolved)
        activated = self.activation(normed)
        outputs = tf.keras.layers.Dropout(DROPOUT_RATE)(activated)
        return outputs


class UNetDownLayer(tf.keras.layers.Layer):
    """
    UNet down layer.
    """

    def __init__(self, is_gated: bool, filters: int, kernel_size: int, stride: int = 1, dilation_rate: int = 2,
                 downsample_mode: DownsampleModeType = DownsampleMode.MAX_POOL, pad_mode: PadModeType = PadMode.REFLECT, **kwargs) -> None:
        """
        Initialize UNet down layer.

        Args:
            is_gated (bool): Apply gated convolution
            filters (int): Number of filters
            kernel_size (int): Kernel size
            stride (int, optional): Stride. Defaults to 1
            dilation_rate (int, optional): Dilation rate. Defaults to 2
            downsample_mode (DownsampleModeType, optional): Downsample mode. Defaults to DownsampleMode.MAX_POOL
            pad_mode (PadModeType, optional): Padding mode. Defaults to PadMode.REFLECT
        """

        super().__init__(**kwargs)

        self.is_gated = is_gated
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation_rate = dilation_rate
        self.downsample_mode = downsample_mode
        self.pad_mode = pad_mode

        if self.is_gated:
            Conv2D = GatedConv2D
        else:
            Conv2D = tf.keras.layers.Conv2D

        if self.downsample_mode != DownsampleMode.STRIDE and self.stride != 1:

            if self.downsample_mode == DownsampleMode.MAX_POOL:

                self.downsample = tf.keras.layers.MaxPool2D(pool_size=self.stride, strides=1)

            elif self.downsample_mode == DownsampleMode.AVERAGE_POOL:

                self.downsample = tf.keras.layers.AveragePooling2D(pool_size=self.stride, strides=1)

        self.conv = Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=1,
            dilation_rate=self.dilation_rate,
            padding='same',
            kernel_initializer='he_normal',
            bias_initializer='he_normal'
        )
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.activation = tf.keras.layers.LeakyReLU(alpha=LEAKY_RELU_ALPHA)
        self.unet_conv_block = UNet2DConvolutionBlock(
            is_gated=self.is_gated,
            filters=self.filters,
            kernel_size=self.kernel_size,
            stride=self.stride,
            pad_mode=self.pad_mode,
            dilation_rate=self.dilation_rate
        )

    def get_config(self) -> dict[str, Any]:
        """
        Get layer configuration.

        Returns:
            dict[str, Any]: Layer configuration
        """

        config = super().get_config()

        config.update({
            'is_gated': self.is_gated,
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'dilation_rate': self.dilation_rate,
            'downsample_mode': self.downsample_mode,
            'pad_mode': self.pad_mode
        })

        return config

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Forward pass.

        Args:
            inputs (tf.Tensor): Input tensor

        Returns:
            tf.Tensor: Output tensor
        """

        convolved = self.conv(inputs)
        downsampled = self.downsample(convolved) if hasattr(self, 'downsample') else convolved
        normed = self.batch_norm(downsampled)
        activated = self.activation(normed)
        outputs = self.unet_conv_block(activated)

        return outputs


class UNetUpLayer(tf.keras.layers.Layer):
    """
    UNet up layer.
    """

    def __init__(self, is_gated: bool, filters: int, kernel_size: int, dilation_rate: int = 2,
                 upsample_mode: UpsampleModeType = UpsampleMode.DECONVOLUTION, pad_mode: PadModeType = PadMode.REFLECT, **kwargs) -> None:
        """
        Initialize UNet up layer.

        Args:
            is_gated (bool): Apply gated convolution
            filters (int): Number of filters
            kernel_size (int): Kernel size
            dilation_rate (int, optional): Dilation rate. Defaults to 2
            upsample_mode (UpsampleModeType, optional): Upsample mode. Defaults to UpsampleMode.DECONVOLUTION
            pad_mode (PadModeType, optional): Padding mode. Defaults to PadMode.REFLECT
        """

        super().__init__(**kwargs)

        self.is_gated = is_gated
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.upsample_mode = upsample_mode
        self.pad_mode = pad_mode

        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.unet_conv_block_1 = UNet2DConvolutionBlock(
            is_gated=self.is_gated,
            filters=self.filters,
            kernel_size=self.kernel_size,
            stride=1,
            dilation_rate=self.dilation_rate,
            pad_mode=self.pad_mode
        )
        self.unet_conv_block_2 = UNet2DConvolutionBlock(
            self.is_gated,
            filters=self.filters,
            kernel_size=1,
            stride=1,
            dilation_rate=self.dilation_rate,
            pad_mode=self.pad_mode
        )

        if upsample_mode == UpsampleMode.DECONVOLUTION:

            self.upsample = tf.keras.layers.Conv2DTranspose(
                filters=filters,
                kernel_size=2,
                strides=2,
                padding='same',
                kernel_initializer='he_normal',
                bias_initializer='he_normal'
            )

        elif upsample_mode in (UpsampleMode.BILINEAR, UpsampleMode.NEAREST_NEIGHBOR):

            self.upsample = tf.keras.layers.UpSampling2D(size=2, interpolation=upsample_mode.value)

    def get_config(self) -> dict[str, Any]:
        """
        Get layer configuration.

        Returns:
            dict[str, Any]: Layer configuration
        """

        config = super().get_config()

        config.update({
            'is_gated': self.is_gated,
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'dilation_rate': self.dilation_rate,
            'upsample_mode': self.upsample_mode,
            'pad_mode': self.pad_mode
        })
        return config

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Forward pass.

        Args:
            inputs (tf.Tensor): Input tensor

        Returns:
            tf.Tensor: Output tensor
        """

        normed = self.batch_norm(inputs)
        convolved1 = self.unet_conv_block_1(normed)
        convolved2 = self.unet_conv_block_2(convolved1)
        outputs = self.upsample(convolved2)

        return outputs


UNetConfig: TypeAlias = tuple[int, ...]


class UNet(tf.keras.Model):
    """
    UNet Model
    """

    def __init__(self, input_shape: int | tuple[int, int], filters: UNetConfig, kernels: UNetConfig, skip_filters: UNetConfig,
                 skip_kernels: UNetConfig, is_gated: bool = True, downsample_mode: DownsampleModeType = DownsampleMode.MAX_POOL,
                 pad_mode: PadModeType = PadMode.REFLECT, upsample_mode: UpsampleModeType = UpsampleMode.DECONVOLUTION, **kwargs) -> None:
        """
        Initialize the UNet model.

        Args:
            input_shape (int | tuple[int, int]): Input shape
            filters (UNetConfig): Number of filters per layer
            kernels (UNetConfig): Kernel size per layer
            skip_filters (UNetConfig): Number of filters per skip layer
            skip_kernels (UNetConfig): Kernel size per skip layer
            is_gated (bool, optional): Apply gated convolution. Defaults to True.
            downsample_mode (DownsampleModeType, optional): Downsample mode. Defaults to DownsampleMode.MAX_POOL.
            pad_mode (PadModeType, optional): Padding mode. Defaults to PadMode.CONSTANT.
            upsample_mode (UpsampleModeType, optional): Upsample mode. Defaults to UpsampleMode.DECONVOLUTION.
        """

        super().__init__(**kwargs)

        self.input_shape = input_shape  # type: ignore
        self.filters = filters
        self.kernels = kernels
        self.skip_filters = skip_filters
        self.skip_kernels = skip_kernels
        self.is_gated = is_gated
        self.downsample_mode = downsample_mode
        self.pad_mode = pad_mode
        self.upsample_mode = upsample_mode

        self.down_layers: list[tf.keras.layers.Layer] = []

        for idx, (filter_num, kernel) in enumerate(zip(filters, kernels)):
            self.down_layers.append(
                UNetDownLayer(
                    is_gated=self.is_gated,
                    filters=filter_num,
                    kernel_size=kernel,
                    stride=2,
                    dilation_rate=1,
                    downsample_mode=self.downsample_mode,
                    pad_mode=self.pad_mode,
                    name=f'unet_down_{idx}'
                )
            )

        self.up_layers: list[tf.keras.layers.Layer] = []

        for idx, (filter_num, kernel) in enumerate(zip(filters, kernels)):
            self.up_layers.append(
                UNetUpLayer(
                    is_gated=self.is_gated,
                    filters=filter_num,
                    kernel_size=kernel,
                    dilation_rate=3,
                    upsample_mode=self.upsample_mode,
                    pad_mode=PadMode.CONSTANT if idx == len(filters) - 1 else self.pad_mode, # Other pad modes don't work with small feature maps
                    name=f'unet_up_{idx}'
                )
            )

        self.skip_layers: list[tf.keras.layers.Layer] = []

        for idx, (filter_num, kernel) in enumerate(zip(skip_filters, skip_kernels)):
            self.skip_layers.append(
                UNet2DConvolutionBlock(
                    is_gated=self.is_gated,
                    filters=filter_num,
                    kernel_size=kernel,
                    stride=1,
                    dilation_rate=3,
                    pad_mode=self.pad_mode,
                    name=f'unet_skip_{idx}'
                )
            )

        self.final_conv = tf.keras.layers.Conv2D(
            filters=3,
            kernel_size=1,
            strides=1,
            padding='same',
            kernel_initializer='he_normal',
            bias_initializer='he_normal'
        )

    def get_config(self) -> dict[str, Any]:
        """
        Get layer configuration.

        Returns:
            dict[str, Any]: Layer configuration
        """

        config = super().get_config()

        config.update({
            'filters': self.filters,
            'kernels': self.kernels,
            'skip_filters': self.skip_filters,
            'skip_kernels': self.skip_kernels,
            'is_gated': self.is_gated,
            'output_channels': self.output_channels,
            'downsample_mode': self.downsample_mode,
            'pad_mode': self.pad_mode,
            'upsample_mode': self.upsample_mode
        })

        return config

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Forward pass.

        Args:
            inputs (tf.Tensor): Input tensor

        Returns:
            tf.Tensor: Output tensor
        """

        down_computed = inputs
        down_outputs = []

        for down_layer in self.down_layers:
            down_computed = down_layer(down_computed)
            down_outputs.append(down_computed)

        skip_outputs = []

        for skip_layer, down_output in zip(self.skip_layers, down_outputs):
            skip_outputs.append(skip_layer(down_output))

        up_computed = skip_outputs[-1]

        for up_layer, skip_output in zip(reversed(self.up_layers), reversed(skip_outputs[:-1])):
            concatenator = tf.keras.layers.Concatenate()
            up_computed = concatenator((up_layer(up_computed), skip_output))

        return self.final_conv(self.up_layers[0](up_computed))

    def __build_graph(self) -> tf.keras.Model:
        """
        Builds a functional graph model. This is used for summary and drawing,
        because the summary and drawing methods do not work with subclassed
        models.

        Returns:
            tf.keras.Model: Graph model
        """

        input_layer = tf.keras.Input(shape=self.input_shape)
        return tf.keras.Model(inputs=input_layer, outputs=self.call(input_layer))

    def summary(self, *args, **kwargs) -> None:
        """
        Prints a summary of the model.
        """

        self.__build_graph().summary(*args, **kwargs)

    def draw(self, file_path: Path | str | None = None) -> None:
        """
        Draws the model.

        Args:
            file_path (Path | str, optional): File path. Defaults to None.
        """

        if file_path is None:
            file_path = Path(f'{self.name}.pdf')

        tf.keras.utils.plot_model(self.__build_graph(), to_file=file_path, show_shapes=True, rankdir='TB',
                                  show_layer_names=True, expand_nested=True, dpi=100)

    def create(self) -> tf.keras.Model:
        """
        Creates the model.

        Returns:
            tf.keras.Model: UNet model
        """

        return self.__build_graph()

    @property
    def name(self) -> str:
        """
        Get model name.

        Returns:
            str: Model name
        """

        return 'UNet'

    @property
    def type(self) -> str:
        """
        Get model type.

        Returns:
            str: Model type
        """

        return 'unet'

    @property
    def input_shape(self) -> tuple[int, int, int]:
        """
        Get input shape.

        Returns:
            tuple[int, int, int]: Input shape
        """

        return self.__input_shape

    @input_shape.setter
    def input_shape(self, shape: int | tuple[int, int]) -> None:
        """
        Sets the  input shape.

        Args:
            shape (int | tuple[int, int]): Input shape
        """

        if isinstance(shape, int):
            self.__input_shape = (shape, shape, 3)
        else:
            self.__input_shape = (*shape, 3)
