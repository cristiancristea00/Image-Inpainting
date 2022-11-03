from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path
import time
import traceback
from typing import Final

from colorama import Fore, Style
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from image_comparator import ImageComparator
from image_processor import ImageProcessor
from unet import UNet
from utils import set_global_seed

EPOCHS: Final[int] = 1
BATCH: Final[int] = 128
PLOT_SIZE: Final[tuple[int, int]] = (16, 9)


def get_dataset_pair(image_batch: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    return tf.cast(tf.map_fn(ImageProcessor.apply_mask, image_batch), tf.uint8), tf.cast(image_batch, tf.uint8)


def ssim(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    return tf.reduce_mean(ImageComparator.compute_ssim(y_true, y_pred))


def psnr(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    return tf.reduce_mean(ImageComparator.compute_psnr(y_true, y_pred))


def run() -> None:

    set_global_seed(42)
    now = datetime.now().strftime('%Y.%m.%d_%H:%M')
    model_path = Path('models', F'model_{now}')

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
        filepath=model_path,
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode='auto',
        save_freq='epoch'
    )

    dataset = Path('images', 'CIFAR10-64')

    print(Fore.GREEN + 'Loading train dataset...' + Style.RESET_ALL)
    train_images = tf.keras.utils.image_dataset_from_directory(dataset / 'train', image_size=(64, 64), labels=None, batch_size=BATCH, shuffle=True)
    train_images = train_images.map(get_dataset_pair, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

    print(Fore.GREEN + 'Loading test dataset...' + Style.RESET_ALL)
    test_images = tf.keras.utils.image_dataset_from_directory(dataset / 'test', image_size=(64, 64), labels=None, batch_size=BATCH)
    test_images = test_images.map(get_dataset_pair, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

    print(Fore.GREEN + 'Creating model...' + Style.RESET_ALL)
    unet = UNet(filters=[16, 32, 64, 128, 128, 128], kernels=[7, 7, 5, 5, 3, 3], skip_filters=[4] * 6, skip_kernels=[1] * 6)
    unet = unet.build_model(input_shape=(64, 64, 3))

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01, amsgrad=True)
    loss = tf.keras.losses.MeanSquaredError()
    metrics = [ssim, psnr]

    unet.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    print(Fore.GREEN + 'Training model...' + Style.RESET_ALL)
    print(Fore.CYAN)
    model_training = unet.fit(
        train_images,
        validation_data=test_images,
        epochs=EPOCHS,
        use_multiprocessing=True,
        workers=12,
        verbose=1,
        callbacks=[reduce_learning_rate_callback, early_stopping_callback, checkpoint_callback]
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
        history = pd.DataFrame(model_training.history)
        history.to_json(file, indent=4)

    print(Fore.GREEN + 'Evaluating model...' + Style.RESET_ALL)
    print(Fore.CYAN)
    model_evaluation = unet.evaluate(
        test_images,
        use_multiprocessing=True,
        workers=12,
        verbose=1,
        return_dict=True
    )
    print(Style.RESET_ALL)

    print(Fore.GREEN + 'Saving model evaluation...' + Style.RESET_ALL)
    with open(model_path / R'model_evaluation.json', 'w') as file:
        evaluation = pd.DataFrame(model_evaluation, index=[0])
        evaluation.to_json(file, indent=4)

    mpl.rcParams['font.sans-serif'] = ['Readex Pro']
    plt.style.use('ggplot')

    fig1, ax1 = plt.subplots(1, 1, figsize=PLOT_SIZE)
    fig2, ax2 = plt.subplots(1, 1, figsize=PLOT_SIZE)
    fig3, ax3 = plt.subplots(1, 1, figsize=PLOT_SIZE)

    length = len(model_training.history['loss'])
    dims = np.arange(1, length + 1)

    ax1.plot(dims, model_training.history['loss'], label='Training loss', color='green')
    ax1.plot(dims, model_training.history['val_loss'], label='Validation loss', color='purple')
    ax1.set(title='Loss (MSE)', xlabel='Epoch', ylabel='Loss')

    ax2.plot(dims, model_training.history['ssim'], label='Training SSIM', color='green')
    ax2.plot(dims, model_training.history['val_ssim'], label='Validation SSIM', color='purple')
    ax2.set(title='SSIM', xlabel='Epoch', ylabel='SSIM')

    ax3.plot(dims, model_training.history['psnr'], label='Training PSNR', color='green')
    ax3.plot(dims, model_training.history['val_psnr'], label='Validation PSNR', color='purple')
    ax3.set(title='PSNR', xlabel='Epoch', ylabel='PSNR')

    for ax in (ax1, ax2, ax3):
        ax.set_xlim(xmin=1, xmax=length)
        ax.legend(fontsize=26, facecolor='white', framealpha=0.75)
        ax.set_xticks(dims)
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.xaxis.label.set_size(28)
        ax.yaxis.label.set_size(28)
        ax.xaxis.labelpad = 10
        ax.yaxis.labelpad = 10
        ax.title.set_size(32)

    print(Fore.GREEN + 'Saving model history graphs...' + Style.RESET_ALL)
    fig1.savefig(model_path / R'model_history_loss.png', dpi=100, bbox_inches='tight')
    fig2.savefig(model_path / R'model_history_ssim.png', dpi=100, bbox_inches='tight')
    fig3.savefig(model_path / R'model_history_psnr.png', dpi=100, bbox_inches='tight')


def main() -> None:
    print(Fore.MAGENTA + 'Starting script...' + Style.RESET_ALL)

    start_time = time.time()
    try:
        run()
    except KeyboardInterrupt:
        print(Fore.RED + 'Script interrupted by the user!' + Style.RESET_ALL)
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
