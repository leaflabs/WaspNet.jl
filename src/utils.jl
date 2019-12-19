# Utility function which takes kwargs for a neuron constructor n_constr and construct a layer
#   out of those neuron types.
#
# This function enables us to abstract away creation of a layer to specifying element-wise
#   parameters and the layer parameters itself (W, number of neurons). We cannot broadcast
#   over keyword arguments, so we have to write another function to handle the broadcasting.
function Batch_Layer_Construction(n_constr, W, N_neurons; kwargs...)
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

    return Layer(neurons, W, N_neurons)
end
