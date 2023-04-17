from __future__ import annotations

from pathlib import Path
from typing import Any, Final

import tensorflow as tf

from layers import GatedConv2D, Padding2D, PadModeType, PadMode

LEAKY_RELU_ALPHA: Final[float] = 0.2
DROPOUT_RATE: Final[float] = 0.1


class ResidualBlock(tf.keras.layers.Layer):
    """
    Residual Block
    """

    def __init__(self, is_gated: bool, filters: int, kernel_size: int | tuple[int, int], strides: int | tuple[int, int] = 1,
                 dilation_rate: int | tuple[int, int] = 1, pad_mode: PadModeType = PadMode.REFLECT, **kwargs) -> None:
        """
        Initialize Residual Block.

        Args:
            is_gated (bool): Apply gated convolution
            filters (int): Number of filters
            kernel_size (int | tuple[int, int]): Kernel size
            strides (int | tuple[int, int], optional): Strides. Defaults to 1.
            dilation_rate (int | tuple[int, int], optional): Dilation rate. Defaults to 1.
            pad_mode (PadModeType, optional): Padding mode. Defaults to PadMode.REFLECT
        """

        super().__init__(**kwargs)

        self.is_gated = is_gated
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.dilation_rate = dilation_rate
        self.pad_mode = pad_mode

        if self.is_gated:
            Conv2D = GatedConv2D
        else:
            Conv2D = tf.keras.layers.Conv2D

        if isinstance(self.kernel_size, int):
            padding = (self.kernel_size + (self.kernel_size - 1) * (self.dilation_rate - 1) - 1) // 2

            self.pad1 = Padding2D(padding=padding, pad_mode=pad_mode)
            self.pad2 = Padding2D(padding=padding, pad_mode=pad_mode)
        else:
            padding_x = (self.kernel_size[0] + (self.kernel_size[0] - 1) * (self.dilation_rate[0] - 1) - 1) // 2
            padding_y = (self.kernel_size[1] + (self.kernel_size[1] - 1) * (self.dilation_rate[1] - 1) - 1) // 2

            self.pad1 = Padding2D(padding=(padding_x, padding_y), pad_mode=pad_mode)
            self.pad2 = Padding2D(padding=(padding_x, padding_y), pad_mode=pad_mode)

        self.conv1 = Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding='valid',
            dilation_rate=self.dilation_rate,
            kernel_initializer='he_normal',
            bias_initializer='he_normal',
            name='conv1'
        )
        self.batch_norm1 = tf.keras.layers.BatchNormalization(name='batch_norm1')
        self.conv2 = Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding='valid',
            dilation_rate=self.dilation_rate,
            kernel_initializer='he_normal',
            bias_initializer='he_normal',
            name='conv2'
        )
        self.batch_norm2 = tf.keras.layers.BatchNormalization(name='batch_norm2')

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Forward pass.

        Args:
            inputs (tf.Tensor): Input tensor

        Returns:
            tf.Tensor: Output tensor
        """

        padded1 = self.pad1(inputs)
        convolved1 = self.conv1(padded1)
        normed1 = self.batch_norm1(convolved1)
        dropped1 = tf.keras.layers.Dropout(DROPOUT_RATE)(normed1)
        activation = tf.keras.layers.LeakyReLU(LEAKY_RELU_ALPHA)
        activated = activation(dropped1)
        padded2 = self.pad2(activated)
        convolved2 = self.conv2(padded2)
        outputs = self.batch_norm2(convolved2)

        return outputs

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
            'strides': self.strides,
            'dilation_rate': self.dilation_rate,
            'pad_mode': self.pad_mode
        })

        return config


class ResNet(tf.keras.Model):
    """
    ResNet Model
    """

    def __init__(self, input_shape: int | tuple[int, int], num_blocks: int = 13, is_gated: bool = True, **kwargs) -> None:
        """
        Initialize the ResNet Model.

        Args:
            input_shape (int | tuple[int, int]): Input shape
            num_blocks (int, optional): Number of residual blocks. Defaults to 8.
            is_gated (bool, optional): Apply gated convolution. Defaults to True.
        """

        super().__init__(**kwargs)

        self.input_shape = input_shape
        self.num_blocks = num_blocks
        self.is_gated = is_gated

        self.initial_pad = Padding2D(
            padding=3,
            name='initial_pad'
        )
        self.initial_conv = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=3,
            strides=1,
            padding='same',
            kernel_initializer='he_normal',
            bias_initializer='he_normal',
            name='initial_conv'
        )
        self.initial_batch_norm = tf.keras.layers.BatchNormalization(name='initial_batch_norm')
        self.inital_max_pool = tf.keras.layers.MaxPool2D(
            pool_size=3,
            strides=2,
            padding='same',
            name='initial_max_pool'
        )

        self.residual_blocks: list[ResidualBlock] = []

        for idx in range(self.num_blocks):
            self.residual_blocks.append(ResidualBlock(
                is_gated=self.is_gated,
                filters=32,
                kernel_size=3,
                strides=1,
                dilation_rate=3,
                name=F'residual_{idx}'
            ))

        self.final_conv = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=5,
            strides=1,
            padding='valid',
            kernel_initializer='he_normal',
            bias_initializer='he_normal',
            name='final_conv'
        )

        self.final_batch_norm = tf.keras.layers.BatchNormalization(name='final_batch_norm')

        self.final_activation = tf.keras.layers.LeakyReLU(LEAKY_RELU_ALPHA)

        self.upsample = tf.keras.layers.Conv2DTranspose(
            filters=3,
            kernel_size=4,
            strides=2,
            padding='valid',
            kernel_initializer='he_normal',
            bias_initializer='he_normal',
            name='upsample'
        )

    def get_config(self) -> dict[str, Any]:
        """
        Get layer configuration.

        Returns:
            dict[str, Any]: Layer configuration
        """

        config = super().get_config()

        config.update({
            'num_blocks': self.num_blocks,
            'is_gated': self.is_gated,
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

        padded = self.initial_pad(inputs)
        convolved = self.initial_conv(padded)
        batch_normed = self.initial_batch_norm(convolved)
        pooled = self.inital_max_pool(batch_normed)
        activation = tf.keras.layers.LeakyReLU(LEAKY_RELU_ALPHA)
        activated = activation(pooled)

        block_input = activated

        for block in self.residual_blocks:
            block_output = block(block_input)
            adder = tf.keras.layers.Add()
            summed = adder([block_input, block_output])
            activation = tf.keras.layers.LeakyReLU(LEAKY_RELU_ALPHA)
            block_input = activation(summed)

        convolved = self.final_conv(block_input)
        batch_normed = self.final_batch_norm(convolved)
        activated = self.final_activation(batch_normed)
        outputs = self.upsample(activated)
        return outputs

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

        if self.is_gated:
            return F'GatedResNet-{self.num_blocks}'

        return F'ResNet-{self.num_blocks}'

    @property
    def type(self) -> str:
        """
        Get model type.

        Returns:
            str: Model type
        """

        return 'resnet'

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
