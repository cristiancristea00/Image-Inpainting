from enum import Enum
from typing import Literal, TypeAlias

import tensorflow as tf


class SSIM(tf.keras.metrics.MeanMetricWrapper):
    """
    SSIM metric.
    """

    def __init__(self, name: str = 'ssim', dtype=None) -> None:
        """
        Initialize the SSIM metric.

        Args:
            name (str, optional): The name of the metric. Defaults to 'ssim'.
            dtype (optional): The data type of the metric. Defaults to None.
        """

        super().__init__(self.__ssim, name, dtype=dtype)

    @staticmethod
    def __ssim(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Wrapper for the SSIM computation.

        Args:
            y_true (tf.Tensor): The ground truth
            y_pred (tf.Tensor): The prediction

        Returns:
            tf.Tensor: The SSIM value
        """

        return tf.image.ssim(y_true, y_pred, max_val=1.0)


class PSNR(tf.keras.metrics.MeanMetricWrapper):
    """
    PSNR metric.
    """

    def __init__(self, name: str = 'psnr', dtype=None) -> None:
        """
        Initialize the PSNR metric.

        Args:
            name (str, optional): The name of the metric. Defaults to 'psnr'.
            dtype (optional): The data type of the metric. Defaults to None.
        """

        super().__init__(self.__psnr, name, dtype=dtype)

    @staticmethod
    def __psnr(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Wrapper for the PSNR computation.

        Args:
            y_true (tf.Tensor): The ground truth
            y_pred (tf.Tensor): The prediction

        Returns:
            tf.Tensor: The PSNR value
        """

        return tf.image.psnr(y_true, y_pred, max_val=1.0)


class SSIM_L1(tf.keras.losses.Loss):
    """
    SSIM + L1 loss.

    A version of the original one from the following paper:
    'Loss Functions for Image Restoration With Neural Networks'

    https://arxiv.org/pdf/1511.08861.pdf
    """

    def __init__(self, alpha: float = 0.75, name='ssim_l1', *args, **kwargs) -> None:
        """
        Initialize the SSIM + L1 loss.

        Args:
            alpha (float, optional): The alpha value. Defaults to 0.75.
            name (str, optional): The name of the loss. Defaults to 'ssim_l1'.
        """

        super().__init__(name=name, *args, **kwargs)

        self.alpha = alpha

    @property
    def alpha(self) -> float:
        """
        Get the alpha value.

        Returns:
            float: The alpha value
        """

        return self.__alpha

    @alpha.setter
    def alpha(self, alpha: float) -> None:
        """
        Set the alpha value.

        Args:
            alpha (float): The alpha value
        """

        if not 0 <= alpha <= 1:
            raise ValueError('Alpha must be between 0 and 1.')

        self.__alpha = alpha

    @staticmethod
    def ssim_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Compute the SSIM loss.

        Args:
            y_true (tf.Tensor): The ground truth
            y_pred (tf.Tensor): The prediction

        Returns:
            tf.Tensor: The SSIM loss
        """

        return 1 - tf.image.ssim(y_true, y_pred, max_val=1.0)

    @staticmethod
    def l1_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Compute the L1 loss.

        Args:
            y_true (tf.Tensor): The ground truth
            y_pred (tf.Tensor): The prediction

        Returns:
            tf.Tensor: The L1 loss
        """

        return tf.reduce_mean(tf.abs(y_true - y_pred))

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Compute the SSIM + L1 loss.

        Args:
            y_true (tf.Tensor): The ground truth
            y_pred (tf.Tensor): The prediction

        Returns:
            tf.Tensor: The SSIM + L1 loss
        """

        return self.alpha * self.l1_loss(y_true, y_pred) + (1 - self.alpha) * self.ssim_loss(y_true, y_pred)


class LPIPSLoss(Enum):
    """
    The LPIPS loss variants.
    """

    VGG16 = 'VGG16'
    VGG19 = 'VGG19'


LPIPSLossType: TypeAlias = Literal[
    LPIPSLoss.VGG16,
    LPIPSLoss.VGG19,
]


class LinearNet(tf.keras.layers.Layer):
    """
    Convolutional layer with a linear activation function and a dropout layer.
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize the linear network.
        """

        super().__init__(*args, **kwargs)

        self.dropout = tf.keras.layers.Dropout(0.5)
        self.conv = tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=1,
            use_bias=False,
            kernel_initializer='he_normal',
            bias_initializer='he_normal'
        )

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Forward pass.

        Args:
            inputs (tf.Tensor): The input tensor

        Returns:
            tf.Tensor: The output tensor
        """

        dropped = self.dropout(inputs)
        outputs = self.conv(dropped)

        return outputs


class LPIPS(tf.keras.losses.Loss):
    """
    Learned Perceptual Image Patch Similarity (LPIPS) loss.

    A version of the original one from the following paper:
    'The Unreasonable Effectiveness of Deep Features as a Perceptual Metric'

    https://arxiv.org/pdf/1801.03924.pdf
    """

    def __init__(self, input_shape: int | tuple[int, int], model: LPIPSLossType = LPIPSLoss.VGG16,
                 all_layers: bool = False, name: str = 'lpips', *args, **kwargs) -> None:
        """
        Initialize the LPIPS loss.

        Args:
            input_shape (int | tuple[int, int]): The input shape
            model (LPIPSLossType, optional): The model type. Defaults to LPIPSLoss.VGG16.
            all_layers (bool, optional): Whether to use all layers. Defaults to False.
            name (str, optional): The name of the loss. Defaults to 'lpips'.

        Raises:
            ValueError: If the model type is invalid
        """

        super().__init__(name=name, *args, **kwargs)

        self.input_shape = input_shape
        self.all_layers = all_layers
        self.model = model
        self.linears: list[LinearNet] = []

        if self.model == LPIPSLoss.VGG16:
            self.feature_extractor = self._vgg16_feature_extractor()
        elif self.model == LPIPSLoss.VGG19:
            self.feature_extractor = self._vgg19_feature_extractor()
        else:
            raise ValueError(F'Invalid model type: {self.model}')

    @property
    def input_shape(self) -> tuple[int, int, int]:
        """
        Get the input shape.

        Returns:
            tuple[int, int, int]: The input shape
        """

        return self.__input_shape

    @input_shape.setter
    def input_shape(self, input_shape: int | tuple[int, int]) -> None:
        """
        Set the input shape.

        Args:
            input_shape (int | tuple[int, int]): The input shape

        Raises:
            ValueError: If the input shape is invalid
        """

        if isinstance(input_shape, int):
            self.__input_shape = (input_shape, input_shape, 3)
        elif isinstance(input_shape, tuple):
            if len(input_shape) != 2:
                raise ValueError('Input shape must be a tuple of two integers.')
            self.__input_shape = (*input_shape, 3)
        else:
            raise ValueError('Input shape must be an integer or a tuple of two integers.')

    def _vgg16_feature_extractor(self) -> tf.keras.Model:
        """
        Create the feature extractor based on the VGG16 model.

        Returns:
            tf.keras.Model: The feature extractor
        """

        vgg16 = tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_shape=self.input_shape)
        vgg16.trainable = False

        if self.all_layers:
            selected_layers = (layer.name for layer in vgg16.layers)
        else:
            selected_layers = ('block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1')

        outputs = [vgg16.get_layer(layer).output for layer in selected_layers]

        feature_extractor = tf.keras.Model(inputs=vgg16.input, outputs=outputs, name='vgg16_lpips')

        for layer in feature_extractor.layers:
            self.linears.append(LinearNet(name=F'{layer.name}_linear'))

        return feature_extractor

    def _vgg19_feature_extractor(self) -> tf.keras.Model:
        """
        Create the feature extractor based on the VGG19 model.

        Returns:
            tf.keras.Model: The feature extractor
        """

        vgg19 = tf.keras.applications.VGG19(include_top=False, weights='imagenet', input_shape=self.input_shape)
        vgg19.trainable = False

        if self.all_layers:
            selected_layers = (layer.name for layer in vgg19.layers)
        else:
            selected_layers = ('block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1')

        outputs = [vgg19.get_layer(layer).output for layer in selected_layers]

        feature_extractor = tf.keras.Model(inputs=vgg19.input, outputs=outputs, name='vgg19_lpips')

        for layer in feature_extractor.layers:
            self.linears.append(LinearNet(name=F'{layer.name}_linear'))

        return feature_extractor

    def _preprocess_images(self, images: tf.Tensor) -> tf.Tensor:
        """
        Preprocess the images to the expected input format of the feature
        extractor.

        Args:
            images (tf.Tensor): The images

        Returns:
            tf.Tensor: The preprocessed images
        """

        result = tf.keras.applications.vgg16.preprocess_input(images)
        return result

    def _extract_features(self, image: tf.Tensor) -> tf.Tensor:
        """
        Extract the features from the image.

        Args:
            image (tf.Tensor): The image

        Returns:
            tf.Tensor: The features
        """

        features = self.feature_extractor(image)
        return features

    def _norm_features(self, features: tf.Tensor) -> tf.Tensor:
        """
        Normalize the features using the L2 norm.

        Args:
            features (tf.Tensor): The features

        Returns:
            tf.Tensor: The normalized features
        """

        result = [tf.math.l2_normalize(elem, axis=(1, 2, 3)) for elem in features]

        return result

    def _get_features(self, image: tf.Tensor) -> tf.Tensor:
        """
        Get the features from the image.

        Args:
            image (tf.Tensor): The image

        Returns:
            tf.Tensor: The features
        """

        image = self._preprocess_images(image)
        features = self._extract_features(image)
        features = self._norm_features(features)

        return features

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Compute the LPIPS loss.

        Args:
            y_true (tf.Tensor): The ground truth
            y_pred (tf.Tensor): The prediction

        Returns:
            tf.Tensor: The LPIPS loss
        """

        y_true = self._get_features(y_true)
        y_pred = self._get_features(y_pred)

        diffs = [tf.square(y_true_elem - y_pred_elem) for y_true_elem, y_pred_elem in zip(y_true, y_pred)]

        for diff, linear in zip(diffs, self.linears):
            diff = linear(diff)

        means = (tf.reduce_mean(diff, axis=(1, 2), keepdims=True) for diff in diffs)
        means = [tf.squeeze(mean, axis=(1, 2)) for mean in means]
        means = tf.concat(means, axis=1)

        result = tf.reduce_sum(means, axis=1)

        return result


class LPIPS_L1(LPIPS):
    """
    LPIPS + L1 loss.
    """

    def __init__(self, input_shape: int | tuple[int, int], alpha: float = 0.75, name: str = 'lpips_l1', *args, **kwargs) -> None:
        """
        Initialize the LPIPS + L1 loss.

        Args:
            input_shape (int | tuple[int, int]): The input shape
            alpha (float, optional): The alpha value. Defaults to 0.75.
            name (str, optional): The name of the loss. Defaults to 'lpips_l1'.
        """

        super().__init__(name=name, input_shape=input_shape,*args, **kwargs)

        self.alpha = alpha

    @property
    def alpha(self) -> float:
        """
        Get the alpha value.

        Returns:
            float: The alpha value
        """

        return self.__alpha

    @alpha.setter
    def alpha(self, alpha: float) -> None:
        """
        Set the alpha value.

        Args:
            alpha (float): The alpha value
        """

        if not 0 <= alpha <= 1:
            raise ValueError('Alpha must be between 0 and 1.')

        self.__alpha = alpha

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Compute the LPIPS + L1 loss.

        Args:
            y_true (tf.Tensor): The ground truth
            y_pred (tf.Tensor): The prediction

        Returns:
            tf.Tensor: The LPIPS + L1 loss
        """

        lpips = super().call(y_true, y_pred)
        l1 = tf.reduce_mean(tf.abs(y_true - y_pred), axis=(1, 2, 3))

        result = self.alpha * l1 + (1 - self.alpha) * lpips

        return result
