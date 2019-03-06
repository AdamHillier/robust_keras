import tensorflow as tf
import numpy as np

import keras.backend as K
from keras.models import load_model
from keras.optimizers import Adam
from keras.datasets import cifar10

from resnet_model import ResnetModel

import math
import os

from keras.backend import manual_variable_initialization
manual_variable_initialization(True)

config = tf.ConfigProto(intra_op_parallelism_threads=4,
                        inter_op_parallelism_threads=4,
                        allow_soft_placement=True,
                        device_count = {"CPU" : 1,
                                        "GPU" : 0})

session = tf.Session(config=config)
K.set_session(session)

config = {
    "subtract_pixel_mean": True,
    "depth": 8,
    "num_classes": 10,
    "epochs": 200,
    "batch_size": 32,
    "num_validation_samples": 5000,
    "epsilon": 2 / 255.0,
    "l1_coef": 1e-3,
    "rs_coef": 1e-3
}

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Leave aside a validation set
x_valid = x_train[-config["num_validation_samples"]:]
y_valid = y_train[-config["num_validation_samples"]:]
x_train = x_train[:-config["num_validation_samples"]]
y_train = y_train[:-config["num_validation_samples"]]

# Input image dimensions.
input_shape = x_train.shape[1:]

# Normalize data.
x_train = x_train.astype("float32") / 255
x_valid = x_valid.astype("float32") / 255
x_test = x_test.astype("float32") / 255

print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print(x_train.shape[0], "train samples")
print(x_valid.shape[0], "validation samples")
print(x_test.shape[0], "test samples")

model = ResnetModel(input_shape=input_shape,
                    depth=config["depth"],
                    l1_coef=config["l1_coef"])

model.summary()

model.compile(loss="sparse_categorical_crossentropy",
              optimizer=Adam(lr=1e-4),
              metrics=["accuracy"])

model.load_weights("saved_models/cifar10_ResNet8_0.008/weights_200_0.58.h5")

[loss, acc] = model.evaluate(x_valid, y_valid, batch_size=config["batch_size"])
print("Validation accuracy: ", acc)
print("Validation loss: ", loss)

[loss, acc] = model.evaluate(x_test, y_test, batch_size=config["batch_size"])
print("Test accuracy: ", acc)
print("Test loss: ", loss)
