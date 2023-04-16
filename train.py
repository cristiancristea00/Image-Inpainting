from __future__ import annotations

import json
from contextlib import redirect_stdout
from pathlib import Path
from typing import Final

import tensorflow as tf
from colorama import Fore, Style

from graphs_generator import GraphsGenerator
from image_browser import ImageBrowser
from image_processor import ImageProcessor
from mask_generator import MaskGenerator
from metrics_and_losses import PSNR, SSIM
from resnet import ResNet
from unet import UNet
from utils import NumpyEncoder

EPOCHS: Final[int] = 500
BATCH: Final[int] = 64
IMAGE_SIZE: Final[int] = 128


def train(model: UNet | ResNet, arguments: list[str]) -> None:
    if len(arguments) != 4:
        raise ValueError(F'Expected 3 arguments, but got {len(arguments) - 1}. '
                         F'You must pass the prefix, the minimum value and the maximum value of the mask ratio.')

    prefix = arguments[1]
    min_value = arguments[2]
    max_value = arguments[3]
    file_name: str = F'{prefix}_{min_value}-{max_value}'

    model_path = Path('models', F'{model.type}_{file_name}')
    model_path.mkdir(parents=True, exist_ok=True)

    mask_ratio: Final[tuple[float, float]] = float(min_value), float(max_value)

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

    mask_generator = MaskGenerator(IMAGE_SIZE, mask_ratio)
    image_processor = ImageProcessor(mask_generator)
    image_browser = ImageBrowser(image_processor, batch_size=BATCH, should_crop=True)

    print(Fore.GREEN + 'Loading train dataset information...' + Style.RESET_ALL, flush=True)
    train_images = image_browser.get_train_dataset()

    print(Fore.GREEN + 'Loading validation dataset information...' + Style.RESET_ALL, flush=True)
    val_images = image_browser.get_val_dataset()

    print(Fore.GREEN + 'Loading test dataset information...' + Style.RESET_ALL, flush=True)
    test_images = image_browser.get_test_dataset()

    print(Fore.GREEN + 'Saving model architecture text summary...' + Style.RESET_ALL, flush=True)
    with open(model_path / 'architecture.txt', 'w', encoding='UTF-8') as file:
        with redirect_stdout(file):
            model.summary(expand_nested=True)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01, amsgrad=True)
    loss = tf.keras.losses.MeanAbsoluteError()
    metrics = [PSNR(), SSIM()]
    callbacks = [reduce_learning_rate_callback, early_stopping_callback, checkpoint_callback]

    model = model.create()
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    print(Fore.GREEN + 'Training model...' + Style.RESET_ALL, flush=True)
    print(Fore.CYAN, flush=True)
    model_training = model.fit(
        train_images,
        validation_data=val_images,
        epochs=EPOCHS,
        use_multiprocessing=True,
        workers=tf.data.AUTOTUNE,
        verbose=2,
        callbacks=callbacks
    )
    print(Style.RESET_ALL, flush=True)

    print(Fore.GREEN + 'Saving model training history...' + Style.RESET_ALL, flush=True)
    with open(model_path / R'model_training.json', 'w', encoding='UTF-8') as file:
        history = json.dumps(model_training.history, cls=NumpyEncoder, indent=4)
        file.write(history)

    model.save(model_path / 'model.h5', save_format='h5')

    print(Fore.GREEN + 'Evaluating model...' + Style.RESET_ALL, flush=True)
    print(Fore.CYAN, flush=True)
    model_evaluation = model.evaluate(
        test_images,
        use_multiprocessing=True,
        workers=tf.data.AUTOTUNE,
        verbose=2,
        return_dict=True
    )
    print(Style.RESET_ALL, flush=True)

    print(Fore.GREEN + 'Saving model evaluation...' + Style.RESET_ALL, flush=True)
    with open(model_path / 'model_evaluation.json', 'w', encoding='UTF-8') as file:
        evaluation = json.dumps(model_evaluation, cls=NumpyEncoder)
        file.write(evaluation)

    print(Fore.GREEN + 'Generating training graphs...' + Style.RESET_ALL, flush=True)
    graphs_generator = GraphsGenerator(model_training.history, model_path)
    graphs_generator.create_all_graphs()
