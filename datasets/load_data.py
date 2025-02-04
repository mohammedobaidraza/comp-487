import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

def load_mnist(flatten=False):
    
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')
    x_train = x_train / 255
    x_test = x_test / 255

    # create dev set using 80% of the training data
    val_size = int(0.8 * x_train.shape[0])
    x_train, x_val = x_train[:val_size], x_train[val_size:]
    y_train, y_val = y_train[:val_size], y_train[val_size:]

    if flatten:
        x_train = x_train.reshape(x_train.shape[0], -1)
        x_val = x_val.reshape(x_val.shape[0], -1)
        x_test = x_test.reshape(x_test.shape[0], -1)

    # one-hot encode the labels
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    y_val = to_categorical(y_val, 10)
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def load_cifar10(flatten=False):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train = x_train / 255
    x_test = x_test / 255

    # create dev set using 80% of the training data
    val_size = int(0.8 * x_train.shape[0])
    x_train, x_val = x_train[:val_size], x_train[val_size:]
    y_train, y_val = y_train[:val_size], y_train[val_size:]

    if flatten:
        x_train = x_train.reshape(x_train.shape[0], -1)
        x_val = x_val.reshape(x_val.shape[0], -1)
        x_test = x_test.reshape(x_test.shape[0], -1)

    # one-hot encode the labels
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    y_val = to_categorical(y_val, 10)
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

def load_fashion_mnist(flatten=False):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')
    x_train = x_train / 255
    x_test = x_test / 255

    # create dev set using 80% of the training data
    val_size = int(0.8 * x_train.shape[0])
    x_train, x_val = x_train[:val_size], x_train[val_size:]
    y_train, y_val = y_train[:val_size], y_train[val_size:]

    if flatten:
        x_train = x_train.reshape(x_train.shape[0], -1)
        x_val = x_val.reshape(x_val.shape[0], -1)
        x_test = x_test.reshape(x_test.shape[0], -1)

    # one-hot encode the labels
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    y_val = to_categorical(y_val, 10)
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def load_cifar100(flatten=False):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train = x_train / 255
    x_test = x_test / 255

    # create dev set using 80% of the training data
    val_size = int(0.8 * x_train.shape[0])
    x_train, x_val = x_train[:val_size], x_train[val_size:]
    y_train, y_val = y_train[:val_size], y_train[val_size:]

    if flatten:
        x_train = x_train.reshape(x_train.shape[0], -1)
        x_val = x_val.reshape(x_val.shape[0], -1)
        x_test = x_test.reshape(x_test.shape[0], -1)

    # one-hot encode the labels
    y_train = to_categorical(y_train, 100)
    y_test = to_categorical(y_test, 100)
    y_val = to_categorical(y_val, 100)
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)