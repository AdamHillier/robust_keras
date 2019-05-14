import numpy as np

import keras.backend as K
from keras.models import Model, load_model, clone_model
from keras.optimizers import Adam
from keras.layers import Activation, Add, AveragePooling2D, BatchNormalization,\
                         Conv2D, Dense, Flatten, MaxPooling2D, Lambda, ReLU, Softmax
from keras import activations
from keras.engine.input_layer import Input, InputLayer
from keras.datasets import cifar10, mnist, fashion_mnist

from pgd_attack import AdversarialExampleGenerator
from svm_model import SVMModel

from tqdm import trange
import scipy.io as sio

from pathlib import Path
import json

def evaluate(model, x, y):
    print("Evaluating model")
    [loss, acc] = model.evaluate(x, y, batch_size=64)[:2]
    print("    Validation accuracy:", acc)
    print("    Validation loss:", loss)

def prune_small_weights(model, tolerance):
    print("Pruning weights smaller than", tolerance)
    weights = model.get_weights()
    count = 0
    for w in weights:
        count += np.count_nonzero(w)
    print("    Initial non-zero weights:", count)
    count = 0
    total = 0
    for w in weights:
        w[np.where(np.abs(w) < tolerance)] = 0
        count += np.count_nonzero(w != 0)
        total += len(w.flatten())
    print("    Remaining non-zero weights:", count)
    print("    Remaining non-zero weights proportion:,", count / total)
    model.set_weights(weights)
    return model

def prune_relus(model, relu_prune_threshold, x_train, y_train, epsilon):
    print("Pruning relus with threshold", relu_prune_threshold)

    num_training_examples = len(x_train)

    def get_ops(adv):
        num_to_remove = int(num_training_examples * relu_prune_threshold)
        assert(num_to_remove <= num_training_examples / 2 + 1)
        linear_relus = adv >= (num_training_examples - num_to_remove)
        zero_relus = adv <= num_to_remove
        ops = np.zeros(adv.shape)
        ops[linear_relus] = 1
        ops[zero_relus] = -1
        return ops

    model.xent_loss = model.total_loss
    for loss in model.losses:
        model.xent_loss -= loss

    adv_train_generator = AdversarialExampleGenerator(model, x_train, y_train,
                                                      batch_size=64,
                                                      epsilon=epsilon,
                                                      k=10,
                                                      a=epsilon / 5,
                                                      incremental=False)

    # Clone to avoid side-effects for the layers of the original model.
    # Requires re-compile.
    old_model = model
    model = clone_model(old_model)
    model.set_weights(old_model.get_weights())
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=Adam(lr=1e-4),
                  metrics=["accuracy"])

    relu_layers = [l for l in model._layers if isinstance(l, ReLU) or
            (isinstance(l, Activation) and l.activation == activations.get("relu"))]

    input_tensors = [model.inputs[0],
                     model.sample_weights[0],
                     model.targets[0],
                     K.learning_phase()]
    output_tensors = []
    for l in relu_layers:
        output_tensors.append(K.sum(K.cast(l.input > 0, "int32"), axis=0))

    get_relu_counts = K.function(inputs=input_tensors, outputs=output_tensors)

    total_relu_counts = []
    for t in output_tensors:
        total_relu_counts.append(np.zeros(t.shape))

    for i in trange(len(adv_train_generator)):
        x_adv, y_adv, _ = adv_train_generator.__getitem__(i)
        relu_counts = get_relu_counts([x_adv, np.ones(len(x_adv)), y_adv, 0])
        for i, rc in enumerate(relu_counts):
            total_relu_counts[i] += rc

    initial = 0
    final = 0
    layer_ops = []
    for counts in total_relu_counts:
        initial += np.count_nonzero(counts)
        ops = get_ops(counts)
        layer_ops.append(ops)
        final += np.count_nonzero(ops == 0)

    print("    Initial number of ReLUs:", initial)
    print("    Remaining number of ReLUs:", final)

    # Input image dimensions
    input_shape = x_train.shape[1:]

    new_outputs = [Input(shape=input_shape)]

    for layer in model._layers[1:]:
        input_layers = []
        for n in layer._inbound_nodes:
            input_layers.extend(n.inbound_layers)
        inputs = [new_outputs[model._layers.index(l)] for l in input_layers]

        if layer in relu_layers:
            assert(len(input_layers) == 1)
            index = relu_layers.index(layer)
            ops = layer_ops[index]

            linear_mask = K.expand_dims(K.constant(ops == 1), axis=0)
            relu_mask = K.expand_dims(K.constant(ops == 0), axis=0)
            new_outputs.append(
                Lambda(lambda x: (linear_mask * x) + (relu_mask * Activation("relu")(x)))(inputs[0])
            )
        else:
            if len(inputs) == 1:
                inputs = inputs[0]
            new_outputs.append(layer(inputs))

    print("    Compiling masked model")

    masked_model = Model(inputs=new_outputs[0], outputs=new_outputs[-1])
        
    masked_model.compile(loss="sparse_categorical_crossentropy",
                      optimizer=Adam(1e-4),
                      metrics=["accuracy"])

    return masked_model, layer_ops

def add_svm_distances(model, x_train, y_train, num_classes=10):
    print("Adding SVM distances to final output")

    if isinstance(model.layers[-1], Softmax):
        logit_layer = model.layers[-2]
    else:
        logit_layer = model.layers[-1]

    # Generate image features
    input_tensor = model.input
    output_tensor = logit_layer.get_input_at(-1)
    image_features_model = Model(inputs=input_tensor, outputs=output_tensor)
    image_features = image_features_model.predict(x_train)

    # Train SVM model
    svm_model = SVMModel(image_features, y_train)

    # Merge weights
    [W_logits, b_logits] = logit_layer.get_weights()
    W_svm, b_svm = svm_model.distances_linear_map(
        image_feature_dim=image_features.shape[-1],
        num_classes=num_classes
    )
    modified_W = np.concatenate((W_logits, W_svm), axis=1)
    modified_b = np.concatenate((b_logits, b_svm))

    modified_logit_layer = Dense(
        num_classes + int(num_classes * (num_classes - 1) / 2),
        name="combined_logits_and_svm_dists"
    )
    output = modified_logit_layer(logit_layer.get_input_at(-1))
    modified_logit_layer.set_weights([modified_W, modified_b])

    # Even if there was originally a final Softmax, it's now irrelevant
    return Model(input=model.input, output=output)

def process_model(model, dataset, normalise, weight_prune, relu_prune, do_eval,
                  add_svm_dists, output_path, epsilon, weight_prune_threshold=1e-3,
                  relu_prune_threshold=0.05, validation_size=5000):
    if dataset == "CIFAR10":
        (x_train, y_train), _ = cifar10.load_data()
        y_train = y_train.reshape(-1)
        num_classes = 10
    elif dataset == "MNIST":
        (x_train, y_train), _ = mnist.load_data()
        x_train = np.expand_dims(x_train, axis=-1)
        num_classes = 10
    elif dataset == "FASHION_MNIST":
        (x_train, y_train), _ = fashion_mnist.load_data()
        x_train = np.expand_dims(x_train, axis=-1)
        num_classes = 10
    else:
        raise ValueError("Unrecognised dataset")

    # Leave aside a validation set
    x_valid = x_train[-validation_size:]
    y_valid = y_train[-validation_size:]
    x_train = x_train[:-validation_size]
    y_train = y_train[:-validation_size]

    # Normalize data
    x_train = x_train.astype("float32") / 255
    x_valid = x_valid.astype("float32") / 255

    if normalise:
        mean = x_train.mean(axis=(0, 1, 2))
        std = x_train.std(axis=(0, 1, 2)) + 1e-8
        print("Normalising channels with values", mean, std)
        input_shape = x_train.shape[1:]
        normalise_input = Input(shape=input_shape, name="input_for_normalise")
        normalise_layer = Lambda(lambda x: (x - K.constant(mean)) / K.constant(std))
        normalise_layer.mean = mean
        normalise_layer.std = std
        output = normalise_layer(normalise_input)
        for l in model._layers[1:]:
            output = l(output)
        model = Model(inputs=normalise_input, outputs=output)
        model.compile(loss="sparse_categorical_crossentropy",
                      optimizer=Adam(lr=1e-4),
                      metrics=["accuracy"])

    if do_eval:
        evaluate(model, x_valid, y_valid)
    if weight_prune:
        prune_small_weights(model, weight_prune_threshold)
        if do_eval:
            evaluate(model, x_valid, y_valid)
    if relu_prune:
        relu_pruned_model, layer_ops = prune_relus(model, relu_prune_threshold,
                                                   x_train, y_train, epsilon)
        if do_eval:
            evaluate(relu_pruned_model, x_valid, y_valid)
    if add_svm_dists:
        model = add_svm_distances(model, x_train, y_train, num_classes=num_classes)

    # Save processed model

    layer_config = []
    weights_to_save = {}

    relu_layers = [l for l in model._layers if isinstance(l, ReLU) or
            (isinstance(l, Activation) and l.activation == activations.get("relu"))]

    for i, layer in enumerate(model._layers):
        input_layers = []
        for n in layer._inbound_nodes:
            input_layers.extend(n.inbound_layers)
        # input_indices = [model._layers.index(l) for l in input_layers]
        input_indices = [model._layers.index(input_layers[-1])] if len(input_layers) > 0 else []

        if isinstance(layer, InputLayer):
            layer_config.append({"type": "Input"})

        elif isinstance(layer, Add):
            assert(len(input_indices) == 2)
            layer_config.append({"type": "Add"})

        elif isinstance(layer, AveragePooling2D):
            assert(len(input_indices) == 1)
            layer_config.append({
                "type": "AveragePool",
                "pool_size": layer.pool_size
            })

        elif isinstance(layer, BatchNormalization):
            assert(len(input_indices) == 1)
            layer_id = "Normalization_{}".format(i)
            [gamma, beta, mm, mv] = layer.get_weights()
            eps = layer.epsilon
            weights_to_save[layer_id + "/mean"] = mm
            weights_to_save[layer_id + "/std"] = np.sqrt(mv + eps)
            weights_to_save[layer_id + "/gamma"] = gamma
            weights_to_save[layer_id + "/beta"] = beta
            layer_config.append({"type": "Normalization"})

        elif isinstance(layer, Conv2D):
            assert(len(input_indices) == 1)
            layer_id = "Conv2D_{}".format(i)
            [weights, bias] = layer.get_weights()
            weights_to_save[layer_id + "/weight"] = weights
            weights_to_save[layer_id + "/bias"] = bias
            layer_config.append({
                "type": "Conv2D",
                "weight_shape": weights.shape,
                "stride": layer.strides,
                "padding": layer.padding
            })

        elif isinstance(layer, Dense):
            assert(len(input_indices) == 1)
            layer_id = "FullyConnected_{}".format(i)
            [weights, bias] = layer.get_weights()
            weights_to_save[layer_id + "/weight"] = weights
            weights_to_save[layer_id + "/bias"] = bias
            layer_config.append({
                "type": "FullyConnected",
                "weight_shape": weights.shape
            })

        elif isinstance(layer, Flatten):
            assert(len(input_indices) == 1)
            layer_config.append({"type": "Flatten"})

        elif isinstance(layer, MaxPooling2D):
            assert(len(input_indices) == 1)
            layer_config.append({
                "type": "MaxPool",
                "pool_size": layer.pool_size
            })

        elif isinstance(layer, Lambda) and layer.mean is not None and layer.std is not None:
            assert(len(input_indices) == 1)
            layer_id = "Normalization_{}".format(i)
            mean, std = layer.mean, layer.std
            weights_to_save[layer_id + "/mean"] = mean
            weights_to_save[layer_id + "/std"] = std
            weights_to_save[layer_id + "/gamma"] = np.ones_like(mean)
            weights_to_save[layer_id + "/beta"] = np.zeros_like(mean)
            layer_config.append({"type": "Normalization"})

        elif layer in relu_layers:
            assert(len(input_indices) == 1)
            if relu_prune:
                layer_type = "MaskedRelu"
                layer_id = "MaskedRelu_{}".format(i)
                index = relu_layers.index(layer)
                ops = layer_ops[index]
                weights_to_save[layer_id + "/mask"] = layer_ops[index]
            else:
                layer_type = "Relu"
            layer_config.append({"type": layer_type})

        elif isinstance(layer, Softmax) and i == len(model._layers) - 1:
            pass

        else:
            raise ValueError("Unsupported layer")

        layer_config[-1]["input_indices"] = input_indices

    output_config_path = str(output_path) + "__config.json"
    output_weights_path = str(output_path) + "__weights.mat"

    with open(str(output_config_path), "w") as outfile:
        json.dump(layer_config, outfile)
    sio.savemat(str(output_weights_path), weights_to_save)

    print("Saved processed model in:")
    print("    " + str(output_config_path))
    print("    " + str(output_weights_path))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    def add_bool_arg(parser, name, default=True):
        group = parser.add_mutually_exclusive_group(required=False)
        group.add_argument("--" + name, dest=name, action="store_true")
        group.add_argument("--no_" + name, dest=name, action="store_false")
        parser.set_defaults(**{name:default})

    parser.add_argument("model_path", type=Path)
    parser.add_argument("dataset", choices=["CIFAR10", "MNIST", "FASHION_MNIST"])
    parser.add_argument("epsilon", type=float)
    parser.add_argument("--validation_size", type=int, default=5000)
    add_bool_arg(parser, "normalise", default=False)
    add_bool_arg(parser, "weight_prune")
    parser.add_argument("--weight_prune_threshold", type=float, default=1e-3)
    add_bool_arg(parser, "relu_prune")
    parser.add_argument("--relu_prune_threshold", type=float, default=0.05)
    add_bool_arg(parser, "do_eval")
    add_bool_arg(parser, "add_svm_distances", default=False)
    parser.add_argument("--output_folder", type=Path, default=Path("processed_models"))
    parser.add_argument("--set_gpu", type=int)

    config = parser.parse_args()

    if not config.model_path.is_file():
        raise ValueError("The model was not found")

    # Restrict GPU memory usage
    if config.set_gpu is not None:
        import tensorflow as tf
        from keras.backend.tensorflow_backend import set_session
        conf = tf.ConfigProto()
        conf.gpu_options.allow_growth = True
        conf.gpu_options.visible_device_list = str(config.set_gpu)
        sess = tf.Session(config=conf)
        set_session(sess)

    # Some models had custom objects, which aren't needed now but the model
    # won't load unless something is there.
    def empty_func(*args):
        return K.zeros(1)
    custom_objects = {
        "rs_loss": empty_func,
        "rs_loss_metric": empty_func,
        "num_unstable": empty_func,
        "num_unstable_metric": empty_func,
        "loss": "sparse_categorical_crossentropy",
        "robust_acc": empty_func
    }
    model = load_model(str(config.model_path), custom_objects)

    # Extract filename, excluding extension
    input_name = config.model_path.with_suffix("").name
    output_path = config.output_folder / Path(*config.model_path.parts[1:-1]) / input_name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    process_model(model, config.dataset, config.normalise, config.weight_prune,
                  config.relu_prune, config.do_eval, config.add_svm_distances,
                  output_path, config.epsilon,
                  weight_prune_threshold=config.weight_prune_threshold,
                  relu_prune_threshold=config.relu_prune_threshold,
                  validation_size=config.validation_size)
