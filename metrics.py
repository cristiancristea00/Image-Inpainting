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
