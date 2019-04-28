import numpy as np

from keras.models import Model
from keras.layers import Activation, Add, AveragePooling2D, BatchNormalization, \
                         Conv2D, Dense, Flatten, Input, MaxPooling2D, ReLU, Softmax
import keras.backend as K
from keras.callbacks import Callback

import tensorflow as tf

import math

def IBP_CNN(input_shape, layers, eps):
    inputs = Input(shape=input_shape)

    x = inputs
    # Hacky way of forcing the learning phase tensor to actually be updated,
    # because we'll need it for selecting the right epsilon and k values based
    # on train/test time.
    x._uses_learning_phase = True
    lb, ub = K.maximum(x - eps, 0), K.minimum(x + eps, 1)

    for layer in layers:
        x = layer(x)
        lb, ub = compute_ia_bounds(layer, lb, ub)

    outputs = Softmax()(x)

    return Model(inputs=inputs, outputs=outputs), lb, ub

def SmallCNN(input_shape, eps, num_classes=10):
    layers = [
        Conv2D(16, (4, 4), strides=2, padding="VALID"),
        ReLU(),
        Conv2D(32, (4, 4), strides=1, padding="VALID"),
        ReLU(),
        Flatten(),
        Dense(100),
        ReLU(),
        Dense(num_classes)
    ]
    return IBP_CNN(input_shape=input_shape, layers=layers, eps=eps)

def MediumCNN(input_shape, eps, num_classes=10):
    layers = [
        Conv2D(32, (3, 3), strides=1, padding="VALID"),
        ReLU(),
        Conv2D(32, (4, 4), strides=2, padding="VALID"),
        ReLU(),
        Conv2D(64, (3, 3), strides=1, padding="VALID"),
        ReLU(),
        Conv2D(64, (4, 4), strides=2, padding="VALID"),
        ReLU(),
        Flatten(),
        Dense(512),
        ReLU(),
        Dense(512),
        ReLU(),
        Dense(num_classes)
    ]
    return IBP_CNN(input_shape=input_shape, layers=layers, eps=eps)

def LargeCNN(input_shape, eps, num_classes=10):
    layers = [
        Conv2D(64, (3, 3), strides=1, padding="VALID"),
        ReLU(),
        Conv2D(64, (3, 3), strides=1, padding="VALID"),
        ReLU(),
        Conv2D(128, (3, 3), strides=2, padding="VALID"),
        ReLU(),
        Conv2D(128, (3, 3), strides=1, padding="VALID"),
        ReLU(),
        Conv2D(128, (3, 3), strides=1, padding="VALID"),
        ReLU(),
        Flatten(),
        Dense(200),
        ReLU(),
        Dense(num_classes)
    ]
    return IBP_CNN(input_shape=input_shape, layers=layers, eps=eps)

def compute_ia_bounds(layer, lb, ub):
    if isinstance(layer, (Activation, AveragePooling2D, BatchNormalization, Flatten, MaxPooling2D, ReLU)):
        # Assuming monotonic, FIXME might not be for general Activation?
        return layer(lb), layer(ub)
    elif isinstance(layer, Conv2D):
        W, b = layer.kernel, layer.bias
        strides, padding = layer.strides, layer.padding.upper()
        c = (lb + ub) / 2.
        r = (ub - lb) / 2.
        c = tf.nn.convolution(c, W, padding=padding, strides=strides)
        if b is not None:
            c = c + b
        r = tf.nn.convolution(r, tf.abs(W), padding=padding, strides=strides)
        return c - r, c + r
    elif isinstance(layer, Dense):
        # FIXME need to assert no non-linear `activation' attribute
        W, b = layer.kernel, layer.bias
        c = (lb + ub) / 2.
        r = (ub - lb) / 2.
        c = tf.matmul(c, W)
        if b is not None:
            c = c + b
        r = tf.matmul(r, tf.abs(W))
        return c - r, c + r
    else:
        raise ValueError("Unfamiliar layer " + layer.name + ", can't compute bounds")

class InterpolateSchedule():
    def __init__(self, start_value, end_value, start_time, end_time):
        assert(start_time < end_time)
        self.start_value = start_value
        self.end_value = end_value
        self.start_time = start_time
        self.end_time = end_time

    def get(self, time):
        if time <= self.start_time:
            return self.start_value
        elif self.end_time <= time:
            return self.end_value
        else:
            frac = (time - self.start_time) / (self.end_time - self.start_time)
            return (1 - frac) * self.start_value + frac * self.end_value

class ConstantSchedule():
    def __init__(self, value):
        self.value = value

    def get(self, time):
        return self.value

class ScheduleHyperParamCallback(Callback):
    def __init__(self, name, variable, schedule, update_every, verbose=0):
        assert(len(variable.shape) == 0)
        self.name = name
        self.variable = variable
        self.schedule = schedule
        self.update_every = update_every
        self.verbose = verbose

        self.samples_seen = 0
        self.samples_seen_at_last_update = 0

    def _update(self, logs=None):
        new_value = self.schedule.get(self.samples_seen)
        K.set_value(self.variable, new_value)
        if logs is not None:
            logs[self.name] = new_value
        self.samples_seen_at_last_update = self.samples_seen
        if self.verbose > 0:
            print("ScheduleHyperParam:", self.name, "updated to", new_value)

    def on_train_begin(self, logs=None):
        self._update()

    def on_batch_end(self, batch, logs=None):
        self.samples_seen += logs["size"]
        samples_seen_since = self.samples_seen - self.samples_seen_at_last_update
        if self.update_every == "batch":
            self._update(logs)
        elif isinstance(self.update_every, int) and samples_seen_since >= self.update_every:
            self._update(logs)

    def on_epoch_end(self, epoch, logs):
        if self.update_every == "epoch":
            self._update(logs)
