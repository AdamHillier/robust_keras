import numpy as np

import keras.backend as K
from keras.models import Model, clone_model
from keras.layers import Activation, Add, AveragePooling2D, BatchNormalization,\
                         Conv2D, Dense, Flatten, ReLU, Lambda
from keras import activations
from keras.engine.input_layer import Input, InputLayer
from keras.optimizers import Adam
from keras.datasets import cifar10

from tqdm import trange
import scipy.io as sio

from pathlib import Path
import json

from resnet_model import ResnetModel
from pgd_attack import AdversarialExampleGenerator

import argparse

parser = argparse.ArgumentParser(
    description="Pass in post-processing options. Type -h for details")
parser.add_argument("--model", dest="model",
                    help="specify which saved model to load")
parser.add_argument("--no_weight_prune", dest="weight_prune",
                    action="store_false", help="use this flag to turn off weight pruning")
parser.set_defaults(weight_prune=True)
parser.add_argument("--weight_thresh", dest="weight_thresh", default=1e-3,
                    help="set pruning threshold for small weights (default 1e-3)")
parser.add_argument("--no_relu_prune", dest="relu_prune",
                    action="store_false", help="use this flag to turn off relu pruning")
parser.set_defaults(relu_prune=True)
parser.add_argument("--relu_prune_frac", dest="relu_prune_frac",
                    default=0.05, help="set pruning threshold for relus (default 0.1)")
parser.add_argument("--do_eval", dest="do_eval", action="store_true",
                    help="use this flag to evaluate test accuracy, PGD adversarial accuracy, and ReLU stability after each post-processing step")
parser.set_defaults(do_eval=False)

args = parser.parse_args()

model_path = Path(args.model)
weight_prune = args.weight_prune
relu_prune = args.relu_prune
do_eval = args.do_eval
relu_prune_frac = float(args.relu_prune_frac)
weight_thresh = float(args.weight_thresh)
# Extract filename, excluding extension
output_name = model_path.with_suffix("").name
output_folder = Path("processed_models")

if not model_path.is_file():
    raise ValueError("The model was not found")

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

num_training_examples = x_train.shape[0]
num_valid_examples = config["num_validation_samples"]
valid_batch_size = config["batch_size"]

model = ResnetModel(input_shape=input_shape,
                    depth=config["depth"],
                    l1_coef=config["l1_coef"])

model.compile(loss="sparse_categorical_crossentropy",
              optimizer=Adam(lr=1e-4),
              metrics=["accuracy"])

model.xent_loss = model.total_loss
for loss in model.losses:
    model.xent_loss -= loss

model.load_weights(model_path)

adv_train_generator = AdversarialExampleGenerator(model, x_train, y_train,
                                                  config["batch_size"],
                                                  epsilon=config["epsilon"],
                                                  k=10,
                                                  a=config["epsilon"] / 5.0,
                                                  incremental=False)

adv_valid_generator = AdversarialExampleGenerator(model, x_valid, y_valid,
                                                  config["batch_size"],
                                                  epsilon=config["epsilon"],
                                                  k=20,
                                                  a=config["epsilon"] / 10.0,
                                                  incremental=False)

def evaluate(model):
    print("Evaluating model")
    [loss, acc] = model.evaluate(x_valid, y_valid, batch_size=config["batch_size"])
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

def get_ops(adv, relu_prune_frac):
    num_to_remove = int(num_training_examples * relu_prune_frac)
    assert(num_to_remove <= num_training_examples / 2 + 1)
    linear_relus = adv >= (num_training_examples - num_to_remove)
    zero_relus = adv <= num_to_remove
    ops = np.zeros(adv.shape)
    ops[linear_relus] = 1
    ops[zero_relus] = -1
    return ops

def prune_relus(model, relu_prune_frac=0.1):
    print("Pruning relus with threshold", relu_prune_frac)

    # Clone to avoid side-effects, requires re-compile
    model = clone_model(model)
    model.compile(loss="sparse_categorical_crossentropy",
              optimizer=Adam(lr=1e-4),
              metrics=["accuracy"])
    
    relu = activations.get("relu")
    relu_layers = [l for l in model.layers if isinstance(l, Activation) and l.activation == relu]

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
        ops = get_ops(counts, relu_prune_frac)
        layer_ops.append(ops)
        final += np.count_nonzero(ops == 0)

    print("    Initial number of ReLUs:", initial)
    print("    Remaining number of ReLUs:", final)

    relu = activations.get("relu")
    relu_layers = [l for l in model.layers if isinstance(l, Activation) and l.activation == relu]

    new_outputs = [Input(shape=input_shape)]

    for layer in model.layers[1:]:
        input_layers = []
        for n in layer._inbound_nodes:
            input_layers.extend(n.inbound_layers)
        inputs = [new_outputs[model.layers.index(l)] for l in input_layers]

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

if do_eval:
    evaluate(model)

if weight_prune:
    prune_small_weights(model, weight_thresh)
    if do_eval:
        evaluate(model)

if relu_prune:
    relu_pruned_model, layer_ops = prune_relus(model, relu_prune_frac)
    if do_eval:
        evaluate(relu_pruned_model)

########################
# Save processed model #
########################

layer_config = []
weights_to_save = {}

relu_layers = [l for l in model.layers if isinstance(l, Activation) and l.activation == activations.get("relu")]

for i, layer in enumerate(model.layers):
    input_layers = []
    for n in layer._inbound_nodes:
        input_layers.extend(n.inbound_layers)
    input_indices = [model.layers.index(l) for l in input_layers]
    
    if isinstance(layer, InputLayer):
        layer_config.append({"type": "Input"})

    elif isinstance(layer, AveragePooling2D):
        assert(len(input_indices) == 1)
        layer_config.append({
            "type": "AveragePool",
            "inputs": input_indices,
            "pool_size": layer.pool_size
        })

    elif isinstance(layer, BatchNormalization):
        assert(len(input_indices) == 1)
        layer_id = "Normalization_{}".format(i)
        
        if len(layer.weights) == 0:
            raise ValueError("Encounted BatchNormalization without weights; redundant layer")
        # All variables will be of the same shape; pick any
        shape = layer.weights[0].shape
        
        if layer.center:
            means = -K.eval(layer.beta)
        else:
            means = np.ones(shape)
        if layer.scale:
            std_deviations = 1 / K.eval(layer.gamma)
        else:
            std_deviations = np.ones(shape)

        weights_to_save[layer_id + "/means"] = means
        weights_to_save[layer_id + "/std_deviations"] = std_deviations
        layer_config.append({"type": "Normalization", "inputs": input_indices})

    elif isinstance(layer, Conv2D):
        assert(len(input_indices) == 1)
        layer_id = "Conv2D_{}".format(i)
        [weights, bias] = layer.get_weights()
        weights_to_save[layer_id + "/weight"] = weights
        weights_to_save[layer_id + "/bias"] = bias
        layer_config.append({
            "type": "Conv2D",
            "inputs": input_indices,
            "weight_shape": weights.shape
        })

    elif isinstance(layer, Dense):
        assert(len(input_indices) == 1)
        layer_id = "FullyConnected_{}".format(i)
        [weights, bias] = layer.get_weights()
        weights_to_save[layer_id + "/weight"] = weights
        weights_to_save[layer_id + "/bias"] = bias
        layer_config.append({
            "type": "FullyConnected",
            "inputs": input_indices,
            "weight_shape": weights.shape
        })

    elif isinstance(layer, Flatten):
        assert(len(input_indices) == 1)
        layer_config.append({"type": "Flatten", "inputs": input_indices})

    elif isinstance(layer, Add):
        assert(len(input_indices) == 2)
        layer_config.append({"type": "Add", "inputs": input_indices})

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
            layer_id = "Relu_{}".format(i)
        layer_config.append({"type": layer_type, "inputs": input_indices})

    else:
        raise Exception("Unfamiliar layer")

if not output_folder.exists():
    output_folder.mkdir()

output_config_path = output_folder / "{}__config.json".format(output_name)
output_weights_path = output_folder / "{}__weights.mat".format(output_name)

with open(str(output_config_path), "w") as outfile:
     json.dump(layer_config, outfile)
sio.savemat(str(output_weights_path), weights_to_save)

print("Saved processed model in:")
print("    " + str(output_config_path))
print("    " + str(output_weights_path))
