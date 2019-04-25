import tensorflow as tf
import numpy as np

from keras.models import load_model
from keras.losses import sparse_categorical_crossentropy
from keras.datasets import cifar10, mnist
from keras.backend.tensorflow_backend import set_session

from pgd_attack import AdversarialExampleGenerator

import math
import argparse
from pathlib import Path

def evaluate(model, dataset, section, adv=None, validation_size=5000,
             adv_iterations=40, adv_restarts=1, verbose=1):
    if dataset == "CIFAR10":
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    elif dataset == "MNIST":
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = np.expand_dims(x_train, axis=-1)
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

    if adv is None or adv == 0:
        return model.evaluate(x, y, verbose=verbose)[1]
    else:
        # Keras gives no easy way of just getting the cross-entropy loss (without the
        # regularisation/rs loss) and it's needed for PGD, so we need to create it again
        model.xent_loss = sparse_categorical_crossentropy(model.targets[0], model.outputs[0])

        adv_generator = AdversarialExampleGenerator(model, x, y,
                                                    batch_size=64,
                                                    epsilon=adv,
                                                    k=adv_iterations,
                                                    a=adv / 10.0,
                                                    incremental=False)

        return model.evaluate_generator(adv_generator, workers=0, verbose=verbose)[1]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("model_path", type=Path)
    parser.add_argument("dataset", choices=["MNIST", "CIFAR10"])
    parser.add_argument("section", choices=["train", "validation", "test"])

    parser.add_argument("--adversary", type=float)
    parser.add_argument("--adversary_iterations", type=int, default=40)
    parser.add_argument("--adversary_restarts", type=int, default=1)

    parser.add_argument("--validation_size", type=int, default=5000)

    config = parser.parse_args()

    model = load_model(config.model_path)

    evaluate(model, config.dataset, config.section, adv=config.adversary,
             validation_size=config.validation_size,
             adv_iterations=config.adversary_iterations,
             adv_restarts=config.adversary_restarts)
