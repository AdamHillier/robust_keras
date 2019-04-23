from keras.models import Model, Sequential
from keras.layers import Activation, Add, AveragePooling2D, BatchNormalization, \
                         Conv2D, Dense, Dropout, Flatten, MaxPooling2D, ReLU
from keras.engine.input_layer import Input
from keras.regularizers import l1, l2
from keras import activations

import tensorflow as tf
import numpy as np

import math

def SmallCNN(input_shape, l1_coef, num_classes=10):
    model = Sequential()
    width = math.ceil(max(input_shape[0:2]) / 2)
    model.add(Conv2D(16, (4, 4), strides=2, padding="SAME",
                     kernel_regularizer=l1(width * width * l1_coef),
                     input_shape=input_shape))
    width = math.ceil(width / 2)
    model.add(Activation("relu"))
    model.add(Conv2D(32, (4, 4), strides=2, padding="SAME",
                     kernel_regularizer=l1(width * width * l1_coef)))
    width = math.ceil(width / 2)
    model.add(Activation("relu"))
    model.add(Flatten())
    model.add(Dense(100, kernel_regularizer=l1(l1_coef)))
    model.add(Activation("relu"))
    model.add(Dense(num_classes, activation="softmax"))
    return model

def LargeCNN(input_shape, l1_coef, num_classes=10):
    model = Sequential()
    width = math.ceil(max(input_shape[0:2]) / 2)
    model.add(Conv2D(32, (3, 3), strides=1, padding="SAME",
                     kernel_regularizer=l1(width * width * l1_coef),
                     input_shape=input_shape))
    model.add(Activation("relu"))
    model.add(Conv2D(32, (4, 4), strides=2, padding="SAME",
                     kernel_regularizer=l1(width * width * l1_coef)))
    width = math.ceil(width / 2)
    model.add(Activation("relu"))
    model.add(Conv2D(64, (3, 3), strides=1, padding="SAME",
                     kernel_regularizer=l1(width * width * l1_coef)))
    model.add(Activation("relu"))
    model.add(Conv2D(64, (4, 4), strides=2, padding="SAME",
                     kernel_regularizer=l1(width * width * l1_coef)))
    width = math.ceil(width / 2)
    model.add(Activation("relu"))
    model.add(Flatten())
    model.add(Dense(512, kernel_regularizer=l1(l1_coef)))
    model.add(Activation("relu"))
    model.add(Dense(512, kernel_regularizer=l1(l1_coef)))
    model.add(Activation("relu"))
    model.add(Dense(num_classes, activation="softmax"))
    return model

# Adapted from https://keras.io/examples/cifar10_resnet/
def ResNet(input_shape, l1_coef, depth=20, dropout=0.2, num_classes=10):
    def resnet_layer(inputs,
                    num_filters=16,
                    kernel_size=3,
                    strides=1,
                    activation="relu",
                    batch_normalization=True,
                    conv_first=True):

        conv = Conv2D(num_filters,
                    kernel_size=kernel_size,
                    strides=strides,
                    padding="same",
                    kernel_initializer="he_normal",
                    kernel_regularizer=l2(1e-4))
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

    if (depth - 2) % 6 != 0:
        raise ValueError("depth should be 6n+2 (eg 20, 32, 44 in [a])")

    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)

    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # First layer but not first stack
                strides = 2  # Downsample
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y = Dropout(dropout)(y)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None)
            if stack > 0 and res_block == 0:  # First layer but not first stack
                # Linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = Add()([x, y])
            x = Activation("relu")(x)
        num_filters *= 2
    x = AveragePooling2D(pool_size=(int(x.shape[1]), int(x.shape[2])))(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation="softmax",
                    kernel_initializer="he_normal")(y)

    # Instantiate model
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Adapted from https://github.com/MadryLab/relu_stable/blob/master/models/MNIST_naive_ia.py
def relu_stability_naive(model, rs_coef, epsilon):
    def _rs_loss(lb, ub, norm_constant=1.0):
        return rs_coef * -tf.reduce_mean(tf.reduce_sum(tf.tanh(1.0 + norm_constant * lb * ub), axis=-1))

    def _num_unstable(lb, ub):
        is_unstable = tf.cast(lb * ub < 0.0, tf.int32)
        all_but_first_dim = np.arange(len(is_unstable.shape))[1:]
        return tf.reduce_sum(is_unstable, all_but_first_dim)

    total_rs_loss = tf.zeros(1)
    total_num_unstable = tf.zeros(1, dtype=tf.int32)

    # Use ._layers as Sequential models hide the input
    input_layer = model._layers[0]

    input_layer.lb = tf.maximum(input_layer.output - epsilon, 0)
    input_layer.ub = tf.minimum(input_layer.output + epsilon, 1)

    for l in model._layers[1:-1]:
        inbound_layers = []
        for n in l._inbound_nodes:
            inbound_layers.extend(n.inbound_layers)

        if isinstance(l, Activation):
            assert(len(inbound_layers) == 1)
            x = inbound_layers[0]
            if l.activation == activations.get("relu"):
                rs_loss = _rs_loss(x.lb, x.ub)
                l.add_loss(rs_loss)
                total_rs_loss += rs_loss
                total_num_unstable += _num_unstable(x.lb, x.ub)
            # Assuming monotonic increasing
            l.lb = l(x.lb)
            l.ub = l(x.ub)
        elif isinstance(l, Add):
            assert(len(inbound_layers) == 2)
            l.lb = inbound_layers[0].lb + inbound_layers[1].lb
            l.ub = inbound_layers[0].ub + inbound_layers[1].ub
        elif isinstance(l, (AveragePooling2D, BatchNormalization, Flatten, MaxPooling2D)):
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
        elif isinstance(l, ReLU):
            assert(len(inbound_layers) == 1)
            x = inbound_layers[0]
            rs_loss = _rs_loss(x.lb, x.ub)
            l.add_loss(rs_loss)
            total_rs_loss += rs_loss
            total_num_unstable += _num_unstable(x.lb, x.ub)
            l.lb = l(x.lb)
            l.ub = l(x.ub)
        else:
            raise ValueError("Unfamiliar layer " + l.name + ", can't compute bounds")

    return total_rs_loss, total_num_unstable

# Adapted from https://github.com/MadryLab/relu_stable/blob/master/models/MNIST_improved_ia.py
def relu_stability_improved(model, rs_coef, epsilon):
    def _rs_loss(lb, ub, norm_constant=1.0):
        loss = rs_coef * -tf.reduce_mean(tf.reduce_sum(tf.tanh(1.0 + norm_constant * lb * ub), axis=-1))
        return loss

    def _num_unstable(lb, ub):
        is_unstable = tf.cast(lb * ub < 0.0, tf.int32)
        all_but_first_dim = np.arange(len(is_unstable.shape))[1:]
        result = tf.reduce_sum(is_unstable, all_but_first_dim)
        return result

    def _compute_bounds_n_layers(n, lbs, ubs, Ws, biases):
        assert n == len(lbs)
        assert n == len(ubs)
        assert n == len(Ws)
        assert n == len(biases)

        # Current layer
        lb = lbs[-1]
        ub = ubs[-1]
        W = Ws[-1]
        b = biases[-1]

        # Base case
        if n == 1:
            if len(W.shape) == 2:
                naive_ia_bounds = _interval_arithmetic(lb, ub, W, b)
            else:
                naive_ia_bounds = _interval_arithmetic_all_batch(lb, ub, W, b)
            return naive_ia_bounds

        # Recursive case
        W_prev = Ws[-2]
        b_prev = biases[-2]

        # Compute W_A and W_NA
        out_dim = W.shape[-1].value
        active_mask_unexpanded = tf.cast(tf.greater(lb, 0), dtype=tf.float32)
        active_mask = tf.tile(tf.expand_dims(active_mask_unexpanded, 2), [1, 1, out_dim]) # This should be B x y x p
        nonactive_mask = 1.0 - active_mask
        W_A = tf.multiply(W, active_mask) # B x y x p
        W_NA = tf.multiply(W, nonactive_mask) # B x y x p

        # Compute bounds from previous layer
        if len(lb.shape) == 2:
            prev_layer_bounds = _interval_arithmetic_all_batch(lb, ub, W_NA, b)
        else:
            prev_layer_bounds = _interval_arithmetic_batch(lb, ub, W_NA, b)

        # Compute new products
        W_prod = tf.einsum('my,byp->bmp', W_prev, W_A) # b x m x p
        b_prod = tf.einsum('y,byp->bp', b_prev, W_A) # b x p

        lbs_new = lbs[:-1]
        ubs_new = ubs[:-1]
        Ws_new = Ws[:-2] + [W_prod]
        biases_new = biases[:-2] + [b_prod]

        deeper_bounds = _compute_bounds_n_layers(n-1, lbs_new, ubs_new, Ws_new, biases_new)
        return (deeper_bounds[0] + prev_layer_bounds[0], deeper_bounds[1] + prev_layer_bounds[1])

    # Assumes shapes of Bxm, Bxm, mxn, n
    def _interval_arithmetic(lb, ub, W, b):
        W_max = tf.maximum(W, 0.0)
        W_min = tf.minimum(W, 0.0)
        new_lb = tf.matmul(lb, W_max) + tf.matmul(ub, W_min) + b
        new_ub = tf.matmul(ub, W_max) + tf.matmul(lb, W_min) + b
        return new_lb, new_ub

    # Assumes shapes of m, m, Bxmxn, n
    def _interval_arithmetic_batch(lb, ub, W, b):
        W_max = tf.maximum(W, 0.0)
        W_min = tf.minimum(W, 0.0)
        new_lb = tf.einsum("m,bmn->bn", lb, W_max) + tf.einsum("m,bmn->bn", ub, W_min) + b
        new_ub = tf.einsum("m,bmn->bn", ub, W_max) + tf.einsum("m,bmn->bn", lb, W_min) + b
        return new_lb, new_ub

    # Assumes shapes of Bxm, Bxm, Bxmxn, Bxn
    def _interval_arithmetic_all_batch(lb, ub, W, b):
        W_max = tf.maximum(W, 0.0)
        W_min = tf.minimum(W, 0.0)
        new_lb = tf.einsum("bm,bmn->bn", lb, W_max) + tf.einsum("bm,bmn->bn", ub, W_min) + b
        new_ub = tf.einsum("bm,bmn->bn", ub, W_max) + tf.einsum("bm,bmn->bn", lb, W_min) + b
        return new_lb, new_ub

    def _convert_conv_to_fc(conv_layer):
        kernel, bias = conv_layer.kernel, conv_layer.bias
        input_shape = conv_layer.input_shape
        output_shape = conv_layer.output_shape
        strides = conv_layer.strides
        if len(strides) == 2:
            strides = [1, strides[0], strides[1], 1]
        padding = conv_layer.padding.upper()
        # Only support standard 2D convolutions
        assert(len(input_shape) == 4 and len(output_shape) == 4)
        assert(len(strides) == 4 and strides[0] == strides[3] == 1)
        assert(math.ceil(input_shape[1] / strides[1]) == output_shape[1] and
               math.ceil(input_shape[2] / strides[2]) == output_shape[2])
        ident = tf.eye(input_shape[1] * input_shape[2] * input_shape[3])
        ident_reshaped = tf.reshape(ident, [-1, input_shape[1], input_shape[2], input_shape[3]])
        kernel_fc = tf.reshape(
            tf.nn.conv2d(ident_reshaped, kernel, strides=strides, padding=padding),
            [-1, output_shape[1] * output_shape[2] * output_shape[3]]
        )
        bias_fc = tf.reshape(
            tf.expand_dims(tf.ones([output_shape[1], output_shape[2]]), 2) * bias,
            [output_shape[1] * output_shape[2] * output_shape[3]]
        )
        return kernel_fc, bias_fc

    ############################################################################

    if not model.built:
        raise ValueError("Model must be compiled before computing RS bounds")

    if not isinstance(model, Sequential):
        raise ValueError("Improved RS bounds only implemented for sequential models")

    total_rs_loss = tf.zeros(1)
    total_num_unstable = tf.zeros(1, dtype=tf.int32)

    lower_bounds = []
    upper_bounds = []
    weights_fc = []
    biases_fc = []

    # Use ._layers as Sequential models hide the input, and exclude output layer
    for i, l in enumerate(model._layers[:-1]):
        if i == 0:
            # Must be input
            input_shape = l.input_shape
            flattened_shape = [-1, input_shape[1] * input_shape[2] * input_shape[3]]
            lower_bounds.append(tf.maximum(tf.reshape(l.output, flattened_shape) - epsilon, 0))
            upper_bounds.append(tf.minimum(tf.reshape(l.output, flattened_shape) + epsilon, 1))
        elif isinstance(l, Activation):
            if l.activation == activations.get("relu"):
                rs_loss = _rs_loss(lower_bounds[-1], upper_bounds[-1])
                l.add_loss(rs_loss)
                total_rs_loss += rs_loss
                total_num_unstable += _num_unstable(lower_bounds[-1], upper_bounds[-1])
            # Assuming monotonic increasing
            lower_bounds[-1] = l(lower_bounds[-1])
            upper_bounds[-1] = l(upper_bounds[-1])
        elif isinstance(l, (AveragePooling2D, MaxPooling2D)):
            lower_bounds[-1] = l(lower_bounds[-1])
            upper_bounds[-1] = l(upper_bounds[-1])
        elif isinstance(l, Conv2D):
            assert(l.activation == activations.get(None))
            W, b = _convert_conv_to_fc(l)
            weights_fc.append(W)
            biases_fc.append(b)
            lb, ub = _compute_bounds_n_layers(len(lower_bounds), lower_bounds,
                                              upper_bounds, weights_fc, biases_fc)
            lower_bounds.append(lb)
            upper_bounds.append(ub)
        elif isinstance(l, Dense):
            assert(l.activation == activations.get(None))
            W, b = l.kernel, l.bias
            weights_fc.append(W)
            biases_fc.append(b)
            lb, ub = _compute_bounds_n_layers(len(lower_bounds), lower_bounds,
                                              upper_bounds, weights_fc, biases_fc)
            lower_bounds.append(lb)
            upper_bounds.append(ub)
        elif isinstance(l, Flatten):
            # Bounds are already flattened as we keep them all fully connected
            pass
        elif isinstance(l, ReLU):
            rs_loss = _rs_loss(lower_bounds[-1], upper_bounds[-1])
            l.add_loss(rs_loss)
            total_rs_loss += rs_loss
            total_num_unstable += _num_unstable(lower_bounds[-1], upper_bounds[-1])
            # Assuming monotonic increasing
            lower_bounds[-1] = l(lower_bounds[-1])
            upper_bounds[-1] = l(upper_bounds[-1])
        else:
            raise ValueError("Unfamiliar layer " + l.name + ", can't compute bounds")

    return total_rs_loss, total_num_unstable
