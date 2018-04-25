from argparse import ArgumentParser
from math import ceil
import os

import cv2
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Conv2D, Dense, Cropping2D, Lambda, Flatten, Dropout
from keras.models import Sequential
from keras.utils import plot_model
import numpy as np
from sklearn.model_selection import train_test_split


def generate_data(paths, angles, batch_size):
    """Returns generator over the data

    :param paths: list of image paths
    :param angles: list of correspongind angles
    :param batch_size: batch size
    """
    num_batches = ceil(len(paths) / batch_size)
    half_batch = batch_size // 2
    while True:
        for i in range(num_batches):
            batch_paths = paths[i * half_batch:(i + 1) * half_batch]
            batch_angles = angles[i * half_batch:(i + 1) * half_batch]
            batch_X = np.zeros((len(batch_paths) * 2, 160, 320, 3))
            batch_Y = np.zeros((len(batch_paths) * 2))
            # using step = 2 because of left-right flip augmentation
            for j in range(0, len(batch_paths) * 2, 2):
                img_path = batch_paths[j // 2]
                img = cv2.imread(img_path)[..., ::-1]
                flipped = np.fliplr(img)
                angle = batch_angles[j // 2]
                batch_X[j] = img
                batch_Y[j] = angle
                batch_X[j + 1] = flipped
                batch_Y[j + 1] = -angle
            yield batch_X, batch_Y


def read_paths(data_file):
    """Reads the measurements file and extract image paths and angles

    :param data_file: path to measurements file
    :return: list of image paths and list of angles
    """
    images = []
    angles = []
    f = open(data_file)
    line = f.readline().rstrip()
    while line:
        s = line.split(",")
        img_path = s[0]
        angle = float(s[3])
        images.append(img_path)
        angles.append(angle)
        line = f.readline().rstrip()
    f.close()
    return images, angles


def create_model():
    """Creates model as described in NVidia article
    https://devblogs.nvidia.com/deep-learning-self-driving-cars/
    """
    model = Sequential()
    # image normalization
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
    # cropping
    model.add(Cropping2D(cropping=((60, 25), (0, 0))))
    model.add(Conv2D(filters=24, kernel_size=5, strides=2, activation="relu"))
    model.add(Conv2D(filters=36, kernel_size=5, strides=2, activation="relu"))
    model.add(Conv2D(filters=48, kernel_size=5, strides=2, activation="relu"))
    model.add(Conv2D(filters=64, kernel_size=3, activation="relu"))
    model.add(Conv2D(filters=64, kernel_size=3, activation="relu"))
    model.add(Flatten())
    model.add(Dense(100, activation="relu"))
    model.add(Dense(50, activation="relu"))
    model.add(Dense(10, activation="relu"))
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])
    model.summary()
    plot_model(model, to_file="model.png", show_shapes=True)

    return model


def main(args):
    measurements_file = args.file
    batch_size = args.batch_size
    epochs = args.epochs
    images, angles = read_paths(measurements_file)
    train_x, val_x, train_y, val_y = train_test_split(images, angles, shuffle=True, test_size=0.1)
    train_gen = generate_data(train_x, train_y, batch_size)
    val_gen = generate_data(val_x, val_y, batch_size)
    model = create_model()
    checkpoint = ModelCheckpoint("weights-{epoch:02d}-{val_acc:.2f}.hdf5", verbose=2,
        save_best_only=False, mode="auto")
    stopping = EarlyStopping(monitor="val_loss", verbose=1)
    model.fit_generator(train_gen, steps_per_epoch=len(train_x) / (batch_size / 2),
        epochs=epochs, verbose=1, callbacks=[checkpoint, stopping], validation_data=val_gen,
        validation_steps=len(val_x) / (batch_size / 2))


parser = ArgumentParser()
parser.add_argument("-e", dest="epochs", help="number of epochs", default=10, type=int)
parser.add_argument("-b", dest="batch_size", help="batch size", default=128, type=int)
parser.add_argument("-f", dest="file", help="input file with measurements")

args = parser.parse_args()

if __name__ == "__main__":
    main(args)




