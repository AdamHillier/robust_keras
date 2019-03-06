using MIPVerify
using Gurobi
using Memento
using JSON
using MAT

config_array = JSON.parsefile("processed_models/weights_200_0.58__config.json")
weights_dict = "processed_models/weights_200_0.58__weights.mat" |> matread

layers = Layer[]
inputs = Dict{Int, Array{Int, 1}}()

for (i, layer) in enumerate(config_array)
    layer_inputs = layer["input_indices"]
    layer_type = layer["type"]

    @assert (
        layer["type"] == "Input" && length(layer_inputs) == 0 ||
        layer["type"] == "Add" && length(layer_inputs) == 2 ||
        length(layer_inputs) == 1
    )
    if layer_type != "Input"
        inputs[i - 1] = layer_inputs
    end

    if layer_type == "Input"
        # No explicit layer. We will need to be more intelligent later if we
        # want to support multiple inputs. A side-benefit of doing this is that
        # axing the explicit input layer will make 0-indexed Python indices
        # up with 1-indexed Julia indices.
    elseif layer_type == "Add"
        push!(layers, Add())
    elseif layer_type == "AveragePool"
        pool_size = layer["pool_size"]
        push!(layers, AveragePool((1, pool_size[1], pool_size[2], 1)))
    elseif layer_type == "Conv2D"
        layer_id = "Conv2D_$(i-1)"
        weight_shape = layer["weight_shape"]
        @assert size(weight_shape) == (4,)
        expected_shape = (weight_shape[1], weight_shape[2], weight_shape[3], weight_shape[4])
        push!(layers,
              get_conv_params(weights_dict, layer_id, expected_shape;
                              # Only equal stride (single int) is supported
                              expected_stride = layer["stride"][1]))
    elseif layer_type == "Flatten"
        push!(layers, Flatten(4))
    elseif layer_type == "FullyConnected"
        layer_id = "FullyConnected_$(i-1)"
        weight_shape = layer["weight_shape"]
        @assert size(weight_shape) == (2,)
        push!(layers, get_matrix_params(weights_dict, layer_id,
                                        (weight_shape[1], weight_shape[2])))
    elseif layer_type == "MaskedRelu"
        layer_id = "MaskedRelu_$(i-1)"
        push!(layers, MaskedReLU(weights_dict["$layer_id/mask"],
                                 interval_arithmetic))
    elseif layer_type == "Normalization"
        layer_id = "Normalization_$(i-1)"
        push!(layers, Normalize(weights_dict["$layer_id/mean"],
                                weights_dict["$layer_id/std"],
                                weights_dict["$layer_id/gamma"],
                                weights_dict["$layer_id/beta"]))
    elseif layer_type == "Relu"
        push!(layers, ReLU())
    else
        error("Unfamiliar layer type encountered: $layer_type")
    end
end

net = Graph(layers, inputs, "Graph network")
println(net)

cifar10 = read_datasets("CIFAR10", 45000, 5000)
f = frac_correct(net, cifar10.train, 1)
println("Fraction correct: $(f)")
