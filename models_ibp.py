import numpy as np

from keras.models import Model
from keras.layers import Activation, Add, AveragePooling2D, BatchNormalization, \
                         Conv2D, Dense, Flatten, Input, MaxPooling2D, ReLU, Softmax
import keras.backend as K
from keras.callbacks import Callback

import tensorflow as tf

import math

def _build_indices(label, _num_classes=10):
    indices = []
    for i in range(_num_classes):
        indices.append(list(range(i)) + list(range(i + 1, _num_classes)))
    _js = tf.constant(indices, dtype=tf.int32)

    batch_size = tf.shape(label)[0]
    i = tf.range(batch_size, dtype=tf.int32)
    correct_idx = tf.stack([i, tf.cast(label, tf.int32)], axis=1)
    wrong_idx = tf.stack([
        tf.tile(tf.reshape(i, [batch_size, 1]), [1, _num_classes - 1]),
        tf.gather(_js, label),
    ], axis=2)
    return correct_idx, wrong_idx

# Adapted from https://github.com/deepmind/interval-bound-propagation
# FIXME: copy licence
def ibp_loss(y_true, y_pred, model, eps, k, _num_classes=10, elision=False, mean=None, std=None):
    y_true = K.argmax(y_true, axis=-1)
    # Compute indices
    _correct_idx, _wrong_idx = _build_indices(y_true)

    if mean is not None and std is not None:
        min_value = (0 - mean) / std
        max_value = (1 - mean) / std
        print("Min image value", min_value)
        print("Max image value", max_value)
        scaled_eps = eps / K.constant(std)
    else:
        min_value, max_value = 0, 1
        scaled_eps = eps

    lb, ub = K.maximum(model.input - scaled_eps, min_value), K.minimum(model.input + scaled_eps, max_value)
    if elision:
        # Exclude final layer
        for layer in model._layers[1:-1]:
            lb, ub = compute_ia_bounds(layer, lb, ub)
        batch_size = tf.shape(lb)[0]
        w = model._layers[-1].kernel
        b = model._layers[-1].bias
        w_t = tf.tile(tf.expand_dims(tf.transpose(w), 0), [batch_size, 1, 1])
        b_t = tf.tile(tf.expand_dims(b, 0), [batch_size, 1])
        w_correct = tf.expand_dims(tf.gather_nd(w_t, _correct_idx), -1)
        b_correct = tf.expand_dims(tf.gather_nd(b_t, _correct_idx), 1)
        w_wrong = tf.transpose(tf.gather_nd(w_t, _wrong_idx), [0, 2, 1])
        b_wrong = tf.gather_nd(b_t, _wrong_idx)
        w = w_wrong - w_correct
        b = b_wrong - b_correct
        # Maximize z * w + b s.t. lower <= z <= upper.
        c = (lb + ub) / 2.
        r = (ub - lb) / 2.
        c = tf.einsum("ij,ijk->ik", c, w)
        if b is not None:
            c += b
        r = tf.einsum("ij,ijk->ik", r, tf.abs(w))
        bounds = c + r
    else:
        for layer in model._layers[1:]:
            lb, ub = compute_ia_bounds(layer, lb, ub)

        correct_class_logit = tf.gather_nd(lb, _correct_idx)
        wrong_class_logits = tf.gather_nd(ub, _wrong_idx)
        bounds = wrong_class_logits - tf.expand_dims(correct_class_logit, 1)

    v = tf.concat(
        [bounds, tf.zeros([tf.shape(bounds)[0], 1], dtype=bounds.dtype)],
        axis=1)
    l = tf.concat(
        [tf.zeros_like(bounds),
            tf.ones([tf.shape(bounds)[0], 1], dtype=bounds.dtype)],
        axis=1)
    _verified_loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=tf.stop_gradient(l), logits=v))

    _cross_entropy = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=y_true, logits=model.layers[-1].output))

    v = tf.reduce_max(bounds, axis=1)
    model.robust_accuracy = tf.reduce_mean(tf.cast(v <= 0., tf.float32))

    return k * _cross_entropy + (1 - k) * _verified_loss

def IBP_CNN(input_shape, layers):
    inputs = Input(shape=input_shape)
    # Hacky way of forcing the learning phase tensor to actually be updated,
    # because we'll need it for selecting the right epsilon and k values based
    # on train/test time.
    inputs._uses_learning_phase = True

    x = inputs
    for layer in layers:
        x = layer(x)

    return Model(inputs=inputs, outputs=x)

def SmallCNN(input_shape, num_classes=10):
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
    return IBP_CNN(input_shape=input_shape, layers=layers)

def MediumCNN(input_shape, num_classes=10):
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
    return IBP_CNN(input_shape=input_shape, layers=layers)

def LargeCNN(input_shape, num_classes=10):
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
    return IBP_CNN(input_shape=input_shape, layers=layers)

def LargeCNN_2(input_shape, num_classes=10):
    layers = [
        Conv2D(64, (3, 3), strides=1, padding="VALID"),
        ReLU(),
        Conv2D(64, (3, 3), strides=1, padding="VALID"),
        ReLU(),
        Conv2D(128, (3, 3), strides=2, padding="VALID"),
        ReLU(),
        Conv2D(128, (3, 3), strides=1, padding="VALID"),
        ReLU(),
        Flatten(),
        Dense(512),
        ReLU(),
        Dense(num_classes)
    ]
    return IBP_CNN(input_shape=input_shape, layers=layers)

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
