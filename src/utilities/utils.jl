# Utility function which takes kwargs for a neuron constructor n_constr and construct a layer
#   out of those neuron types.
#
# This function enables us to abstract away creation of a layer to specifying element-wise
#   parameters and the layer parameters itself (W, number of neurons). We cannot broadcast
#   over keyword arguments, so we have to write another function to handle the broadcasting.

function batch_layer_construction(n_constr, W, N_neurons; recurrent = false, kwargs...)
    # Order the kwargs to pass to the constructor
    arg_keys = keys(kwargs)
    ordered_args = [kwargs[k] for k in arg_keys]

    # A function which maps positional arguments to kwargs, so we can broadcast to it
    function bcast_function(_garbage, args...)
        arg_tuple = [arg_keys[j] => args[j] for j in 1:length(args)]
        return n_constr(;arg_tuple...)
    end

    # Broadcast over the ordered arguments which are potentially arrays, and include a zeros(...)
    #   vector so that we always return the correct number of neurons.
    neurons = bcast_function.(zeros(N_neurons), ordered_args...)
    return Layer(neurons, W)
end

function layer_constructor(n_constr, N_neurons, N_layers, input_layers, init_dist = Normal(0,1))
    neurons = [n_constr() for _ in 1:N_neurons]
    W = BlockArray(zeros(N_neurons, N_neurons*(N_layers+1)), [N_neurons], fill(N_neurons, N_layers+1))
    for input_layer in input_layers
        # TODO: We should not assume N_inputs == N_layer_neurons
        # TODO: We should not assume we connect inputs directly to neurons with unit weights
        if input_layer == 0
            W[Block(1,input_layer+1)] = Matrix(1I,N_neurons,N_neurons)
        else
            # space for more creative bifurcations depending on init_dist here
            W[Block(1,input_layer+1)] = rand!(init_dist, zeros(N_neurons,N_neurons))
        end
    end
    return Layer(neurons, W, input_layers)
end

function network_constructor(N_layers, N_neurons; n_constr = nnsim.Izh, connections = [], init_dist = Normal(0,1))
    if connections == []
        connections = [[i] for i in 0:(N_layers-1)]
    end

    layers = Vector{AbstractLayer}(undef,length(connections))
    for i in 1:length(connections)
        layers[i] = layer_constructor(n_constr, N_neurons, N_layers, connections[i], init_dist)
    end
    return Network(layers)
end
