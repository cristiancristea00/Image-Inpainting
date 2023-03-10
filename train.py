from __future__ import annotations

import json
import time
import traceback
from contextlib import redirect_stdout
from pathlib import Path
from sys import argv
from typing import Final

import tensorflow as tf
from colorama import Fore, Style

from graphs_generator import GraphsGenerator
from image_browser import ImageBrowser
from image_processor import ImageProcessor
from mask_generator import MaskGenerator
from metrics_and_losses import PSNR, SSIM, SSIM_L1
from unet import UNet
from utils import NumpyEncoder

EPOCHS: Final[int] = 5000
BATCH: Final[int] = 64
IMAGE_SIZE: Final[int] = 128


def run() -> None:
    if len(argv) != 4:
        raise ValueError('Invalid number of arguments!')

    prefix = argv[1]
    min_value = argv[2]
    max_value = argv[3]
    file_name: str = F'{prefix}_{min_value}-{max_value}'

    model_path = Path('models', F'model_{file_name}')

    MASK_RATIO: Final[tuple[float, float]] = (float(min_value), float(max_value))

    reduce_learning_rate_callback = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        mode='min',
        min_delta=1e-3,
        factor=0.2,
        patience=20,
        cooldown=5,
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

    mask_generator = MaskGenerator(IMAGE_SIZE, MASK_RATIO)
    image_processor = ImageProcessor(mask_generator)
    image_browser = ImageBrowser(image_processor, batch_size=BATCH, should_crop=True)

    print(Fore.GREEN + 'Loading train dataset information...' + Style.RESET_ALL, flush=True)
    train_images = image_browser.get_train_dataset()

    print(Fore.GREEN + 'Loading validation dataset information...' + Style.RESET_ALL, flush=True)
    val_images = image_browser.get_val_dataset()

    print(Fore.GREEN + 'Loading test dataset information...' + Style.RESET_ALL, flush=True)
    test_images = image_browser.get_test_dataset()

    print(Fore.GREEN + 'Creating model...' + Style.RESET_ALL, flush=True)
    unet = UNet(filters=(16, 32, 64, 64, 64, 128), kernels=(7, 7, 5, 5, 3, 3), skip_filters=(4,) * 6, skip_kernels=(1,) * 6)
    unet = unet.build_model(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01, amsgrad=True)
    loss = SSIM_L1()
    metrics = [PSNR(), SSIM()]
    callbacks = [reduce_learning_rate_callback, early_stopping_callback, checkpoint_callback]

    unet.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    print(Fore.GREEN + 'Training model...' + Style.RESET_ALL, flush=True)
    print(Fore.CYAN, flush=True)
    model_training = unet.fit(
        train_images,
        validation_data=val_images,
        epochs=EPOCHS,
        use_multiprocessing=True,
        workers=tf.data.AUTOTUNE,
        verbose=2,
        callbacks=callbacks
    )
    print(Style.RESET_ALL, flush=True)

    print(Fore.GREEN + 'Saving model architecture text summary...' + Style.RESET_ALL, flush=True)
    with open(model_path / 'architecture.txt', 'w', encoding='UTF-8') as file:
        with redirect_stdout(file):
            unet.summary(expand_nested=True)

    print(Fore.GREEN + 'Saving model training history...' + Style.RESET_ALL, flush=True)
    with open(model_path / R'model_training.json', 'w', encoding='UTF-8') as file:
        history = json.dumps(model_training.history, cls=NumpyEncoder, indent=4)
        file.write(history)

    unet.save(model_path / 'model.h5', save_format='h5')

    print(Fore.GREEN + 'Evaluating model...' + Style.RESET_ALL, flush=True)
    print(Fore.CYAN, flush=True)
    model_evaluation = unet.evaluate(
        test_images,
        use_multiprocessing=True,
        workers=tf.data.AUTOTUNE,
        verbose=2,
        return_dict=True
    )
    print(Style.RESET_ALL, flush=True)

    print(Fore.GREEN + 'Saving model evaluation...' + Style.RESET_ALL, flush=True)
    with open(model_path / R'model_evaluation.json', 'w', encoding='UTF-8') as file:
        evaluation = json.dumps(model_evaluation, cls=NumpyEncoder)
        file.write(evaluation)

    print(Fore.GREEN + 'Generating training graphs...' + Style.RESET_ALL, flush=True)
    graphs_generator = GraphsGenerator(model_training.history, model_path)
    graphs_generator.create_all_graphs()


def main() -> None:
    print(Fore.MAGENTA + 'Starting script...' + Style.RESET_ALL, flush=True)

    start_time = time.perf_counter()
    try:

        gpus = tf.config.list_logical_devices('GPU')
        strategy = tf.distribute.MirroredStrategy(gpus)
        with strategy.scope():
            run()

    except KeyboardInterrupt:

        print(Fore.RED + '\nScript interrupted by the user!' + Style.RESET_ALL, flush=True)

    except Exception:

        print(Fore.RED, flush=True)
        print('Script failed with the following error:', flush=True)
        traceback.print_exc()
        print(Style.RESET_ALL, flush=True)

    end_time = time.perf_counter()

    elapsed = end_time - start_time
    elapsed_time = time.strftime('%H:%M:%S', time.gmtime(elapsed))

    print(Fore.YELLOW + F'Total time: {elapsed_time}' + Style.RESET_ALL, flush=True)
    print(Fore.MAGENTA + 'Everything done!' + Style.RESET_ALL, flush=True)


if __name__ == '__main__':
    main()
