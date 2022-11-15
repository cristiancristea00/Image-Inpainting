import tensorflow as tf

from image_comparator import ImageComparator


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
        return ImageComparator.compute_ssim(y_true, y_pred)


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
        return ImageComparator.compute_psnr(y_true, y_pred)


class MS_SSIM_L1(tf.keras.losses.Loss):
    """
    MS-SSIM + L1 loss.

    A version of the original one from the following paper:
    'Loss Functions for Image Restoration With Neural Networks'

    https://arxiv.org/pdf/1511.08861.pdf
    """

    def __init__(self, name: str = 'ms_ssim_l1', alpha: float = 0.84) -> None:
        """
        Initialize the MS-SSIM + L1 loss.

        Args:
            name (str, optional): The name of the loss. Defaults to 'ms_ssim_l1'.
            dtype (optional): The data type of the loss. Defaults to None.
        """
        super().__init__(name=name)
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
        if alpha < 0 or alpha > 1:
            raise ValueError('Alpha must be between 0 and 1.')
        self.__alpha = alpha

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Compute the MS-SSIM + L1 loss.

        Args:
            y_true (tf.Tensor): The ground truth
            y_pred (tf.Tensor): The prediction

        Returns:
            tf.Tensor: The MS-SSIM + L1 loss
        """
        return self.alpha * tf.image.ssim_multiscale(y_true, y_pred, 1) + (1 - self.alpha) * tf.reduce_mean(tf.abs(y_true - y_pred))
