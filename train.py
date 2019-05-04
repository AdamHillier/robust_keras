import numpy as np

from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, \
                            TensorBoard
from keras.datasets import cifar10, mnist

from models import SmallCNN, LargeCNN, ResNet, relu_stability_naive, relu_stability_improved
from pgd_attack import AdversarialExampleGenerator

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

parser.add_argument("model_name", choices=["SmallCNN", "LargeCNN", "SmallResNet"])
parser.add_argument("dataset", choices=["MNIST", "CIFAR10"])
parser.add_argument("epsilon", type=float)

# Model config
add_bool_arg(parser, "rs")
parser.add_argument("--rs_type", choices=["naive", "improved"], default="naive")
parser.add_argument("--l1_coef", type=float, default=0)
parser.add_argument("--rs_coef", type=float, default=0)
parser.add_argument("--num_classes", type=int, default=10)
parser.add_argument("--load_weights_from", type=Path)

# Training
add_bool_arg(parser, "adv_train")
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--initial_epoch", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--pgd_iter_train", type=int, default=8)
parser.add_argument("--pgd_iter_eval", type=int, default=30)
parser.add_argument("--epsilon_incremental", type=int, default=50)
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
x_valid = x_train[-config.validation_size:]
y_valid = y_train[-config.validation_size:]
x_train = x_train[:-config.validation_size]
y_train = y_train[:-config.validation_size]

# Input image dimensions
input_shape = x_train.shape[1:]

# Normalize data
x_train = x_train.astype("float32") / 255
x_valid = x_valid.astype("float32") / 255

####################
# Initialise model #
####################

# Restrict GPU memory usage
print(config.set_gpu)
if config.set_gpu is not None:
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    conf = tf.ConfigProto()
    conf.gpu_options.allow_growth = True
    conf.gpu_options.visible_device_list = str(config.set_gpu)
    sess = tf.Session(config=conf)
    set_session(sess)
del config.set_gpu

if config.model_name == "SmallCNN":
    model = SmallCNN(input_shape=input_shape, l1_coef=config.l1_coef)
elif config.model_name == "LargeCNN":
    model = LargeCNN(input_shape=input_shape, l1_coef=config.l1_coef)
elif config.model_name == "SmallResNet":
    model = ResNet(input_shape=input_shape, l1_coef=config.l1_coef)
else:
    raise ValueError("Unrecognised model")

if config.rs:
    if config.rs_type == "naive":
        rs_loss, num_unstable = relu_stability_naive(model, rs_coef=config.rs_coef, epsilon=config.epsilon)
    elif config.rs_type == "improved":
        rs_loss, num_unstable = relu_stability_improved(model, rs_coef=config.rs_coef, epsilon=config.epsilon)
    else:
        raise ValueError("Unsupported `rs_type` argument")

if config.load_weights_from is not None:
    model.load_weights(config.load_weights_from)

metrics = ["accuracy"]
if config.rs:
    def rs_loss_metric(a, b):
        return rs_loss
    def num_unstable_metric(a, b):
        return num_unstable
    metrics.extend([rs_loss_metric, num_unstable_metric])

model.compile(loss=sparse_categorical_crossentropy,
              optimizer=Adam(lr=config.lr),
              metrics=metrics)

model.summary()

##################
# Setup training #
##################

# Prepare model model saving directory
model_type = config.model_name
model_name = "RS_%s_%s_eps_%.3f" % (config.dataset, model_type, config.epsilon)
if not config.load_weights_from:
    save_dir = Path("saved_models") / model_name / datetime.now().strftime("%b%d_%H-%M-%S")
    if not save_dir.exists():
        save_dir.mkdir(parents=True)
else:
    save_dir = config.load_weights_from.parent
    del config.load_weights_from
file_path = save_dir / "weights_{epoch:03d}_{val_acc:.2f}.h5"

# Save config to json
with open(str(save_dir / ("config_%d.json" % config.initial_epoch)), "w") as fp:
    json.dump(vars(config), fp, sort_keys=True, indent=4)

# Set up training callbacks
checkpoint =   ModelCheckpoint(filepath=str(file_path),
                               monitor="val_acc",
                               verbose=1)
tensor_board =     TensorBoard(log_dir=save_dir,
                               histogram_freq=0,
                               batch_size=config.batch_size,
                               write_graph=True,
                               write_grads=False,
                               write_images=False,
                               update_freq=2500)
tensor_board.samples_seen = config.initial_epoch * len(x_train)
tensor_board.samples_seen_at_last_write = config.initial_epoch * len(x_train)
lr_reducer = ReduceLROnPlateau(monitor="val_loss",
                               factor=config.lr_reduce_factor,
                               cooldown=0,
                               patience=config.lr_reduce_patience,
                               min_lr=config.lr_reduce_min,
                               verbose=1)
early_stopping = EarlyStopping(monitor="val_loss",
                               patience=config.early_stop_patience,
                               verbose=1)

callbacks = [checkpoint, tensor_board]
if (config.lr_reduce):
    callbacks.append(lr_reducer)
if (config.early_stop):
    callbacks.append(early_stopping)

if config.adv_train:
    # Keras gives no easy way of just getting the cross-entropy loss (without the
    # regularisation/rs loss) and it's needed for PGD, so we need to create it again
    model.xent_loss = sparse_categorical_crossentropy(model.targets[0], model.outputs[0])

    incremental = (1, config.epsilon_incremental) if config.epsilon_incremental > 0 else False
    train_generator = AdversarialExampleGenerator(model, x_train, y_train,
                                                config.batch_size,
                                                epsilon=config.epsilon,
                                                k=config.pgd_iter_train,
                                                a=0.03,
                                                incremental=incremental)
    valid_generator = AdversarialExampleGenerator(model, x_valid, y_valid,
                                                config.batch_size,
                                                epsilon=config.epsilon,
                                                k=config.pgd_iter_eval,
                                                a=config.epsilon / 10.0,
                                                incremental=False)
    model.fit_generator(train_generator,
                        validation_data=valid_generator,
                        epochs=config.epochs,
                        initial_epoch=config.initial_epoch,
                        callbacks=callbacks,
                        workers=0, # Important for the generators
                        shuffle=False) # Shuffling done in the generators

else:
    model.fit(x_train, y_train,
              validation_data=(x_valid, y_valid),
              epochs=config.epochs,
              initial_epoch=config.initial_epoch,
              callbacks=callbacks)
