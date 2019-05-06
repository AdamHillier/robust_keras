import numpy as np

import keras.backend as K
from keras.layers import Softmax
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.datasets import cifar10, mnist

import tensorflow as tf

from models_ibp import SmallCNN, MediumCNN

import math
import argparse
from pathlib import Path

def evaluate(model, dataset, section, epsilon, validation_size=5000, verbose=1):
    if dataset == "CIFAR10":
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    elif dataset == "MNIST":
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = np.expand_dims(x_train, axis=-1)
        x_test = np.expand_dims(x_test, axis=-1)
    else:
        raise ValueError("Unrecognised dataset")

    # Leave aside a validation set
    x_valid = x_train[-validation_size:]
    y_valid = y_train[-validation_size:]
    x_train = x_train[:-validation_size]
    y_train = y_train[:-validation_size]

    if section == "train":
        x = x_train
        y = y_train
    elif section == "validation":
        x = x_valid
        y = y_valid
    elif section == "test":
        x = x_test
        y = y_test
    else:
        raise ValueError("Invalid dataset section")

    # Normalize data
    x = x.astype("float32") / 255

    return model.evaluate([x, np.full(len(x), epsilon), np.zeros(len(x))], y, verbose=verbose)[2]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("model_type", choices=["SmallCNN", "MediumCNN"])
    parser.add_argument("model_path", type=Path)
    parser.add_argument("dataset", choices=["MNIST", "CIFAR10"])
    parser.add_argument("section", choices=["train", "validation", "test"])
    parser.add_argument("epsilon", type=float)

    parser.add_argument("--validation_size", type=int, default=5000)

    parser.add_argument("--set_gpu", type=int)

    config = parser.parse_args()

    # Restrict GPU memory usage
    if config.set_gpu is not None:
        import tensorflow as tf
        from keras.backend.tensorflow_backend import set_session
        conf = tf.ConfigProto()
        conf.gpu_options.allow_growth = True
        conf.gpu_options.visible_device_list = str(config.set_gpu)
        sess = tf.Session(config=conf)
        set_session(sess)

    input_shape = (28, 28, 1) if config.dataset == "MNIST" else (32, 32, 1)

    if config.model_type == "SmallCNN":
        model, lb, ub = SmallCNN(input_shape=input_shape)
    elif config.model_type == "MediumCNN":
        model, lb, ub = MediumCNN(input_shape=input_shape)
    else:
        raise ValueError("Unrecognised model")

    model.load_weights(str(config.model_path))

    k = model.inputs[2][0]

    def ibp_loss(y_true, y_pred):
        adv_logits = tf.math.multiply(lb, y_true) + tf.math.multiply(ub, 1 - y_true)
        adv_pred = Softmax()(adv_logits)

        return k * categorical_crossentropy(y_true, y_pred) + \
            (1 - k) * categorical_crossentropy(y_true, adv_pred)

    def robust_acc(y_true, y_pred):
        # The multiply is zeros except for the lb of the true logit
        true_lb = K.max(tf.math.multiply(lb, y_true), axis=1)
        false_ub = K.max(tf.math.multiply(ub, 1 - y_true), axis=1)
        # print(lb.shape)
        # print(y_true.shape)
        # print(true_lb.shape)
        # print(false_ub.shape)
        # print((true_lb >= false_ub).shape)
        # print(K.mean(true_lb >= false_ub).shape)
        print(tf.math.greater(true_lb, false_ub).shape)
        return K.mean(tf.math.greater(true_lb, false_ub), axis=0)

    metrics = ["accuracy", robust_acc]

    model.compile(loss=ibp_loss, optimizer=Adam(1e-3), metrics=metrics)

    acc = evaluate(model, config.dataset, config.section, config.epsilon,
                   validation_size=config.validation_size)
    print("Accuracy:", acc)
