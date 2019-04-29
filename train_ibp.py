import numpy as np

import keras.backend as K
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint, \
                            ReduceLROnPlateau, TensorBoard
from keras.datasets import cifar10, mnist

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

from models_ibp import SmallCNN, MediumCNN, LargeCNN, ScheduleHyperParamCallback, \
                       ConstantSchedule, InterpolateSchedule, ibp_loss

import math
import argparse
from pathlib import Path
from datetime import datetime
import json

#######################
# Parse configuration #
#######################

parser = argparse.ArgumentParser()

def add_bool_arg(parser, name, default=True):
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--" + name, dest=name, action="store_true")
    group.add_argument("--no_" + name, dest=name, action="store_false")
    parser.set_defaults(**{name:default})

parser.add_argument("model_name", choices=["SmallCNN", "MediumCNN", "LargeCNN"])
parser.add_argument("dataset", choices=["MNIST", "CIFAR10"])
parser.add_argument("eval_epsilon", type=float)
parser.add_argument("train_epsilon", type=float)

# Model config
parser.add_argument("--num_classes", type=int, default=10)
parser.add_argument("--load_weights_from", type=Path)
add_bool_arg(parser, "elide_final_layer", default=False)

# Training
add_bool_arg(parser, "augmentation", default=False)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--initial_epoch", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=100)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--lr_schedule", type=str)
parser.add_argument("--k_warmup", type=int, default=0)
parser.add_argument("--k_rampup", type=int, default=20)
parser.add_argument("--epsilon_warmup", type=int, default=0)
parser.add_argument("--epsilon_rampup", type=int, default=20)
parser.add_argument("--min_k", type=float, default=0.5)
parser.add_argument("--validation_size", type=int, default=5000)
parser.add_argument("--set_gpu", type=int)

# Callbacks
add_bool_arg(parser, "early_stop")
parser.add_argument("--early_stop_patience", type=int, default=30)
add_bool_arg(parser, "lr_reduce")
parser.add_argument("--lr_reduce_patience", type=int, default=10)
parser.add_argument("--lr_reduce_factor", type=float, default=math.sqrt(0.1))
parser.add_argument("--lr_reduce_min", type=float, default=1e-6)

config = parser.parse_args()

######################
# Initialise dataset #
######################

if config.dataset == "CIFAR10":
    (x_train, y_train), _ = cifar10.load_data()
elif config.dataset == "MNIST":
    (x_train, y_train), _ = mnist.load_data()
    x_train = np.expand_dims(x_train, axis=-1)
else:
    raise ValueError("Unrecognised dataset")

# Leave aside a validation set
x_valid = x_train[-config.validation_size:].astype("float32") / 255
y_valid = to_categorical(y_train[-config.validation_size:], num_classes=10)
x_train = x_train[:-config.validation_size].astype("float32") / 255
y_train = to_categorical(y_train[:-config.validation_size], num_classes=10)

# Input image dimensions
input_shape = x_train.shape[1:]

####################
# Initialise model #
####################

# Restrict GPU memory usage
if config.set_gpu is not None:
    conf = tf.ConfigProto()
    conf.gpu_options.allow_growth = True
    conf.gpu_options.visible_device_list = str(config.set_gpu)
    sess = tf.Session(config=conf)
    set_session(sess)
del config.set_gpu

eps_train_var = K.variable(config.train_epsilon)
eps = K.in_train_phase(K.stop_gradient(eps_train_var), K.constant(config.eval_epsilon))
k_train_var = K.variable(1)
k = K.in_train_phase(K.stop_gradient(k_train_var), K.constant(config.min_k))

if config.model_name == "SmallCNN":
    model = SmallCNN(input_shape=input_shape)
elif config.model_name == "MediumCNN":
    model = MediumCNN(input_shape=input_shape)
elif config.model_name == "LargeCNN":
    model = LargeCNN(input_shape=input_shape)
else:
    raise ValueError("Unrecognised model")

def loss(y_true, y_pred):
    return ibp_loss(y_true, y_pred, model, eps, k)

def robust_acc(y_true, y_pred):
    return model.robust_accuracy

if config.load_weights_from is not None:
    model.load_weights(config.load_weights_from)

metrics = ["accuracy", robust_acc]

model.compile(loss=loss, optimizer=Adam(lr=config.lr), metrics=metrics)

model.summary()

##################
# Setup training #
##################

# Prepare model model saving directory
model_type = config.model_name
elision = "elide" if config.elide_final_layer else "no_elide"
model_name = "IBP_%s_%s_train_%.3f_eval_%.3f_%s" % (config.dataset, model_type, config.train_epsilon, config.eval_epsilon, elision)
if not config.load_weights_from:
    save_dir = Path("saved_models") / model_name / datetime.now().strftime("%b%d_%H-%M-%S")
    if not save_dir.exists():
        save_dir.mkdir(parents=True)
else:
    save_dir = config.load_weights_from.parent
file_path = save_dir / "weights_{epoch:03d}_{val_robust_acc:.3f}.h5"

# Save config to json
with open(str(save_dir / ("config_%d.json" % config.initial_epoch)), "w") as fp:
    json.dump(vars(config), fp, sort_keys=True, indent=4)

# Set up training callbacks
checkpoint = ModelCheckpoint(filepath=str(file_path),
                             monitor="val_robust_acc",
                             verbose=1)
tensor_board = TensorBoard(log_dir=save_dir,
                           histogram_freq=0,
                           batch_size=config.batch_size,
                           write_graph=True,
                           write_grads=False,
                           write_images=False,
                           update_freq=5000)
tensor_board.samples_seen = config.initial_epoch * len(x_train)
tensor_board.samples_seen_at_last_write = config.initial_epoch * len(x_train)

callbacks = [checkpoint, tensor_board]

if config.lr_schedule is not None:
    chunks = config.lr_schedule.split(",")
    schedule = [(float(lr), int(epoch)) for (lr, epoch) in [c.split("@") for c in chunks]]
    def scheduler(epoch, current_lr):
        lr = config.lr
        for (rate, e) in schedule:
            if epoch >= e:
                lr = rate
            else:
                break
        return lr
    callbacks.insert(0, LearningRateScheduler(scheduler, verbose=1))

if config.lr_reduce:
    callbacks.insert(0, ReduceLROnPlateau(monitor="val_loss",
                                          factor=config.lr_reduce_factor,
                                          cooldown=0,
                                          patience=config.lr_reduce_patience,
                                          min_lr=config.lr_reduce_min,
                                          verbose=1))
if config.early_stop:
    callbacks.insert(0, EarlyStopping(monitor="val_loss",
                                      patience=config.early_stop_patience,
                                      verbose=1))

if config.epsilon_rampup > 0:
    start = config.epsilon_warmup * len(x_train)
    end = start + config.epsilon_rampup * len(x_train)
    eps_schedule = InterpolateSchedule(0, config.train_epsilon, start, end)
    callbacks.insert(0, ScheduleHyperParamCallback(name="epsilon",
                                                   variable=eps_train_var,
                                                   schedule=eps_schedule,
                                                   update_every=1000,
                                                   verbose=0))

if config.k_rampup > 0:
    start = config.k_warmup * len(x_train)
    end = start + config.k_rampup * len(x_train)
    k_schedule = InterpolateSchedule(1, config.min_k, start, end)
    callbacks.insert(0, ScheduleHyperParamCallback(name="k",
                                                   variable=k_train_var,
                                                   schedule=k_schedule,
                                                   update_every=1000,
                                                   verbose=0))

# Run training, with or without data augmentation.
if not config.augmentation:
    print('Not using data augmentation.')
    model.fit(x_train, y_train,
              validation_data=(x_valid, y_valid),
              epochs=config.epochs,
              initial_epoch=config.initial_epoch,
              batch_size=config.batch_size,
              callbacks=callbacks)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        # randomly rotate images in the range (deg 0 to 30)
        rotation_range=30,
        # randomly shift images horizontally
        width_shift_range=0.1,
        # randomly shift images vertically
        height_shift_range=0.1,
        # set mode for filling points outside the input boundaries
        fill_mode="nearest",
        # randomly flip images
        horizontal_flip=True)

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=config.batch_size),
                        validation_data=(x_valid, y_valid), steps_per_epoch=(len(x_train) / config.batch_size),
                        epochs=config.epochs, initial_epoch=config.initial_epoch,
                        verbose=1, workers=4, callbacks=callbacks)
