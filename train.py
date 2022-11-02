from datetime import datetime
from pathlib import Path
from typing import Final

import tensorflow as tf

from image_processor import ImageProcessor
from unet import UNet
from utils import set_global_seed

EPOCHS: Final[int] = 1


def get_dataset_pair(image_batch):
    return tf.map_fn(ImageProcessor.apply_mask, image_batch), image_batch


def run() -> None:
    set_global_seed(42)
    now = datetime.now().strftime("%H.%M-%Y.%m.%d")

    reduce_learning_rate_callback = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        mode='min',
        min_delta=1e-3,
        factor=0.5,
        patience=20,
        cooldown=10,
        min_lr=1e-6,
        verbose=1
    )

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        mode='min',
        min_delta=1e-4,
        patience=50,
        restore_best_weights=True,
        verbose=1
    )

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=Path('models', F'model_{now}'),
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode='auto',
        save_freq='epoch'
    )

    tf.config.run_functions_eagerly(True)
    tf.data.experimental.enable_debug_mode()
    train_images = tf.keras.utils.image_dataset_from_directory('images/CIFAR10-64/train', labels=None, batch_size=64, image_size=(64, 64), shuffle=True)
    test_images = tf.keras.utils.image_dataset_from_directory('images/CIFAR10-64/test', labels=None, batch_size=64, image_size=(64, 64), shuffle=True)

    train_images = train_images.map(get_dataset_pair, num_parallel_calls=tf.data.AUTOTUNE).cache().prefetch(tf.data.AUTOTUNE)
    test_images = test_images.map(get_dataset_pair, num_parallel_calls=tf.data.AUTOTUNE).cache().prefetch(tf.data.AUTOTUNE)

    unet = UNet(filters=[16, 32, 64, 128, 128, 128], kernels=[7, 7, 5, 5, 3, 3], skip_filters=[4] * 6, skip_kernels=[1] * 6)
    unet = unet.build_model(input_shape=(64, 64, 3))
    tf.keras.utils.plot_model(unet, show_shapes=True, expand_nested=True, dpi=200, to_file='results/unet.png')
    print(unet.summary(expand_nested=True))

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, amsgrad=True)
    loss = tf.keras.losses.MeanSquaredError()
    metrics = [tf.keras.metrics.MeanSquaredError()]

    unet.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    model_fitting = unet.fit(
        train_images,
        validation_data=test_images,
        epochs=EPOCHS,
        use_multiprocessing=True,
        workers=12,
        verbose=1,
        callbacks=[reduce_learning_rate_callback, early_stopping_callback, checkpoint_callback]
    )

    model_evaluation = unet.evaluate(
        test_images,
        use_multiprocessing=True,
        workers=12,
        verbose=1,
        return_dict=True
    )


if __name__ == '__main__':
    run()
