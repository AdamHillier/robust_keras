import tensorflow as tf
import numpy as np

from keras.models import Model
from keras.layers import Activation, Add, AveragePooling2D, BatchNormalization,\
                         Conv2D, Dense, Input, Flatten
from keras.regularizers import l1

# Derived from the ResNet example code here:
# https://keras.io/examples/cifar10_resnet

def ResnetModel(input_shape, depth, num_classes=10, l1_coef=1e-4):
    if (depth - 2) % 6 != 0:
        raise ValueError("Depth should be 6n+2 (eg 20, 32, 44 in [a])")
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)

    x = resnet_layer(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides,
                             l1_coef=l1_coef)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None,
                             l1_coef=l1_coef)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False,
                                 l1_coef=l1_coef)
            x = Add()([x, y])
            x = Activation("relu")(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation="softmax",
                    kernel_initializer="he_normal")(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model

def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation="relu",
                 batch_normalization=True,
                 conv_first=True,
                 l1_coef=1e-4):

    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding="same",
                  kernel_initializer="he_normal",
                  kernel_regularizer=l1(l1_coef))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x

def compute_bounds(model, rs_loss_coef, epsilon=0.1):
    input_layer = model.layers[0]

    input_layer.lb = tf.maximum(input_layer.output - epsilon, 0)
    input_layer.ub = tf.minimum(input_layer.output + epsilon, 1)

    def _rs_loss(lb, ub, norm_constant=1.0):
        loss = -tf.reduce_mean(tf.reduce_sum(tf.tanh(1.0 + norm_constant * lb * ub), axis=-1))
        return rs_loss_coef * loss

    def _num_unstable(lb, ub):
        is_unstable = tf.cast(lb * ub < 0.0, tf.int32)
        all_but_first_dim = np.arange(len(is_unstable.shape))[1:]
        result = tf.reduce_sum(is_unstable, all_but_first_dim)
        return result

    total_rs_loss = tf.zeros(1)
    num_unstable = tf.zeros(1, dtype=tf.int32)

    for l in model.layers[1:]:
        inbound_layers = []
        for n in l._inbound_nodes:
            inbound_layers.extend(n.inbound_layers)

        if isinstance(l, Activation):
            assert(len(inbound_layers) == 1)
            x = inbound_layers[0]
            rs_loss = _rs_loss(x.lb, x.ub)
            total_rs_loss += rs_loss
            x.add_loss(rs_loss)
            num_unstable += _num_unstable(x.lb, x.ub)
            # Assuming monotonic increasing
            l.lb = l(x.lb)
            l.ub = l(x.ub)
        elif isinstance(l, Add):
            assert(len(inbound_layers) == 2)
            l.lb = inbound_layers[0].lb + inbound_layers[1].lb
            l.ub = inbound_layers[0].ub + inbound_layers[1].ub
        elif isinstance(l, (AveragePooling2D, BatchNormalization, Flatten)):
            assert(len(inbound_layers) == 1)
            x = inbound_layers[0]
            # Assuming monotonic increasing
            l.lb = l(x.lb)
            l.ub = l(x.ub)
        elif isinstance(l, Conv2D):
            assert(len(inbound_layers) == 1)
            x = inbound_layers[0]
            W, b = l.kernel, l.bias
            stride, padding = l.strides, l.padding
            W_max = tf.maximum(W, 0.0)
            W_min = tf.minimum(W, 0.0)
            l.lb = tf.nn.conv2d(x.lb, W_max, strides=[1, stride[0], stride[1], 1], padding=padding.upper()) + \
                   tf.nn.conv2d(x.ub, W_min, strides=[1, stride[0], stride[1], 1], padding=padding.upper()) + b
            l.ub = tf.nn.conv2d(x.ub, W_max, strides=[1, stride[0], stride[1], 1], padding=padding.upper()) + \
                   tf.nn.conv2d(x.lb, W_min, strides=[1, stride[0], stride[1], 1], padding=padding.upper()) + b
        elif isinstance(l, Dense):
            assert(len(inbound_layers) == 1)
            x = inbound_layers[0]
            W, b = l.kernel, l.bias
            W_max = tf.maximum(W, 0.0)
            W_min = tf.minimum(W, 0.0)
            l.lb = tf.matmul(x.lb, W_max) + tf.matmul(x.ub, W_min) + b
            l.ub = tf.matmul(x.ub, W_max) + tf.matmul(x.lb, W_min) + b
        else:
            raise Exception("Unfamiliar layer " + l.name + ", can't compute bounds")

    model.rs_loss = total_rs_loss
    model.num_unstable = num_unstable
