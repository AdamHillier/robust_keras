import tensorflow as tf
import numpy as np

from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy
from keras.callbacks import ModelCheckpoint, LearningRateScheduler,\
                            ReduceLROnPlateau, TensorBoard
from keras.datasets import cifar10

from resnet_model import ResnetModel, compute_bounds
from pgd_attack import AdversarialExampleGenerator

import math
import os

config = {
    "depth": 8,
    "num_classes": 10,
    "epochs": 200,
    "batch_size": 32,
    "num_validation_samples": 5000,
    "epsilon": 2 / 255.0,
    "l1_coef": 1e-3,
    "rs_coef": 1e-3
}

(x_train, y_train), _ = cifar10.load_data()

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

print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print(x_train.shape[0], "train samples")
print(x_valid.shape[0], "validation samples")

def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

def rs_loss(a, b):
    return model.rs_loss

def num_unstable(a, b):
    return model.num_unstable

model = ResnetModel(input_shape=input_shape,
                    depth=config["depth"],
                    l1_coef=config["l1_coef"])
compute_bounds(model, rs_loss_coef=config["rs_coef"])
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=Adam(lr=lr_schedule(0)),
              metrics=["accuracy", rs_loss, num_unstable])
model.xent_loss = model.total_loss
for loss in model.losses:
    model.xent_loss -= loss

model.summary()

# Prepare model model saving directory.
model_type = 'ResNet%d' % config["depth"]
model_name = "cifar10_%s_%.3f" % (model_type, config["epsilon"])
save_dir = os.path.join(os.getcwd(), "saved_models", model_name)
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, "weights_{epoch:03d}_{val_acc:.2f}.h5")

# Prepare callbacks for model saving, tensorboard, and learning rate adjustment.
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor="val_acc",
                             verbose=1)

tensor_board = TensorBoard(log_dir=save_dir,
                           histogram_freq=0,
                           batch_size=config["batch_size"],
                           write_graph=True,
                           write_grads=False,
                           write_images=False,
                           update_freq=2500)

lr_scheduler = LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)

callbacks = [checkpoint, lr_reducer, lr_scheduler, tensor_board]

train_generator = AdversarialExampleGenerator(model, x_train, y_train,
                                              config["batch_size"],
                                              epsilon=config["epsilon"],
                                              k=6,
                                              a=0.03,
                                              incremental=(20, config["epochs"] - 20))
val_generator =   AdversarialExampleGenerator(model, x_valid, y_valid,
                                              config["batch_size"],
                                              epsilon=config["epsilon"],
                                              k=20,
                                              a=config["epsilon"] / 10.0,
                                              incremental=False)

model.fit_generator(train_generator,
                    validation_data=val_generator,
                    epochs=config["epochs"],
                    callbacks=callbacks,
                    workers=0, # Important for the generators
                    shuffle=False) # Shuffling done in the generators
