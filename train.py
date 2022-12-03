from __future__ import annotations

import json
import time
import traceback
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path
from typing import Final

import tensorflow as tf
from colorama import Fore, Style

from graphs_generator import GraphsGenerator
from image_browser import ImageBrowser
from image_processor import ImageProcessor
from mask_generator import MaskGenerator
from metrics_and_losses import PSNR, SSIM
from unet import UNet
from utils import NumpyEncoder

EPOCHS: Final[int] = 1
BATCH: Final[int] = 128
IMAGE_SIZE: Final[int] = 64
MASK_RATIO: Final[tuple[float, float]] = (5, 10)  # (5, 10), (15, 20) and (25, 30)


def run() -> None:
    now = datetime.now().strftime('%Y.%m.%d_%H:%M')
    model_path = Path('models', F'model_{now}')
    # logs_dir = model_path / 'logs'

    reduce_learning_rate_callback = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        mode='min',
        min_delta=1e-3,
        factor=0.2,
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
        filepath=model_path,
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode='auto',
        save_freq='epoch'
    )

    # tensorboard_callback = tf.keras.callbacks.TensorBoard(
    #     log_dir=logs_dir,
    #     histogram_freq=1,
    #     profile_batch='20, 100'
    # )

    mask_generator = MaskGenerator(IMAGE_SIZE, MASK_RATIO)
    image_processor = ImageProcessor(mask_generator, BATCH)
    image_browser = ImageBrowser(image_processor)

    print(Fore.GREEN + 'Loading train dataset...' + Style.RESET_ALL)
    train_images = image_browser.get_train_dataset()

    print(Fore.GREEN + 'Loading test dataset...' + Style.RESET_ALL)
    test_images = image_browser.get_test_dataset()

    print(Fore.GREEN + 'Creating model...' + Style.RESET_ALL)
    unet = UNet(filters=[16, 32, 64, 64, 64, 128], kernels=[7, 7, 5, 5, 3, 3], skip_filters=[4] * 6, skip_kernels=[1] * 6)
    unet = unet.build_model(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01, amsgrad=True)
    loss = tf.keras.losses.MeanSquaredError()
    metrics = [PSNR(), SSIM()]
    callbacks = [reduce_learning_rate_callback, early_stopping_callback, checkpoint_callback]

    unet.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    print(Fore.GREEN + 'Training model...' + Style.RESET_ALL)
    print(Fore.CYAN)
    model_training = unet.fit(
        train_images,
        validation_data=test_images,
        epochs=EPOCHS,
        use_multiprocessing=True,
        workers=tf.data.AUTOTUNE,
        verbose=1,
        callbacks=callbacks
    )
    print(Style.RESET_ALL)

    print(Fore.GREEN + 'Saving model architecture text summary...' + Style.RESET_ALL)
    with open(model_path / 'architecture.txt', 'w') as file:
        with redirect_stdout(file):
            unet.summary(expand_nested=True)

    print(Fore.GREEN + 'Saving model architecture graph...' + Style.RESET_ALL)
    tf.keras.utils.plot_model(unet, show_shapes=True, expand_nested=True, dpi=200, to_file=model_path / 'architecture.png')

    print(Fore.GREEN + 'Saving model training history...' + Style.RESET_ALL)
    with open(model_path / R'model_training.json', 'w') as file:
        history = json.dumps(model_training.history, cls=NumpyEncoder, indent=4)
        file.write(history)

    unet.save(model_path / 'model.h5', save_format='h5')

    print(Fore.GREEN + 'Evaluating model...' + Style.RESET_ALL)
    print(Fore.CYAN)
    model_evaluation = unet.evaluate(
        test_images,
        use_multiprocessing=True,
        workers=tf.data.AUTOTUNE,
        verbose=1,
        return_dict=True
    )
    print(Style.RESET_ALL)

    print(Fore.GREEN + 'Saving model evaluation...' + Style.RESET_ALL)
    with open(model_path / R'model_evaluation.json', 'w') as file:
        evaluation = json.dumps(model_evaluation, cls=NumpyEncoder)
        file.write(evaluation)

    print(Fore.GREEN + 'Generating training graphs...' + Style.RESET_ALL)
    graphs_generator = GraphsGenerator(model_training.history, model_path)
    graphs_generator.create_all_graphs()


def main() -> None:
    print(Fore.MAGENTA + 'Starting script...' + Style.RESET_ALL)

    start_time = time.time()
    try:
        run()
    except KeyboardInterrupt:
        print(Fore.RED + '\nScript interrupted by the user!' + Style.RESET_ALL)
    except:
        print(Fore.RED)
        print('Script failed with the following error:')
        traceback.print_exc()
        print(Style.RESET_ALL)
    end_time = time.time()

    elapsed_time = end_time - start_time
    elapsed_time = time.strftime('%H:%M:%S', time.gmtime(elapsed_time))

    print(Fore.YELLOW + F'Total time: {elapsed_time}' + Style.RESET_ALL)
    print(Fore.MAGENTA + 'Everything done!' + Style.RESET_ALL)


if __name__ == '__main__':
    main()
