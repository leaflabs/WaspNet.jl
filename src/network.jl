# A Network comprises layers and handles passing inputs between layers. Furthemore,
#   it also tracks the states of all of the neurons at each time step.
"""
    mutable struct Network<:AbstractNetwork

Contains constituent `Layer`s, orchestrates the movement of signals between `Layer`s, and handles first-layer input.

# Fields
- `layers::Array{AbstractLayer,1}`: Array of `Layer`s ordered from 1 to N for N layers
- `N_in::Int`: Number of input dimensions to the first `Layer`
- `N_out::Int`: Number of output dimensions from the final `Layer`
- `neur_states::Matrix`: TODO: This will hopefully be deprecated soon 
- `neur_outputs::Matrix`: TODO: This will hopefully be deprecated soon 
- `state_size::Int`: TODO: This will hopefully be deprecated soon; it's used for pre-allocating `neur_states`
- `t`: TODO: This will hopefully be deprecated soon; internal time parameter for simulations
- `prev_outputs::Vector`: Vector of vectors sized to hold the output from each `Layer` 
"""
mutable struct Network<:AbstractNetwork
    layers::Array{AbstractLayer, 1}  # Array of layers in order from input to output
    N_in::Int                          # Number of input dimensions
    N_out::Int                         # Number of output dimensions
    neur_states::Matrix                # The states of each neuron for each time step
    neur_outputs::Matrix               # The outputs of each neuron for each time step
    state_size::Int
    t                                  # Internal time parameter

    prev_outputs::Vector               # Vector to pre-allocate storage for feeding into layers

    function Network(layers, N_in, N_out, neur_states, neur_outputs, state_size, t)
        neurons_per_layer = [length(l.neurons) for l in layers]
        output_sizes = vcat([N_in], neurons_per_layer) # number of signals passed out of each layer incl. input

        N_layers = length(layers)
        N_neurons = sum(neurons_per_layer)

        prev_outputs = [zeros(N_in), [zeros(j) for j in neurons_per_layer]...]

        net_layers = []
        for (i,l) in enumerate(layers)
            N_layer_neurons = length(l.neurons)
            # TODO: maybe convert this to PseudoBlockArray
            W = BlockArray(zeros(N_layer_neurons, N_neurons + N_in), [N_layer_neurons], output_sizes)
            conns = Array{Int,1}()
            if isempty(l.conns)
                push!(conns, i-1)
                setblock!(W, l.W, 1, i)
                new_layer = Layer(l.neurons, W, conns, l.N_neurons, l.input, l.output)
            else
                conns = copy(l.conns)
                W = copy(l.W)
                new_layer = deepcopy(l)#_field_update(l, [:conns, :W], [conns, W])
            end
            push!(net_layers, new_layer)
        end

        return new(net_layers, N_in, N_out, neur_states, neur_outputs, state_size, 0., prev_outputs)
    end
end

# Very general constructor which connects layers (including FF with undefined conns) and 
#   arranges the block arrays in each layer. Each layer in the Network is a deepcopy of 
#   the layers fed in, aside from the fields which are explicitly changed.
function Network(layers, N_in::Int) 
    neurons_per_layer = [length(l.neurons) for l in layers]
    
    N_layers = length(layers)
    N_neurons = sum(neurons_per_layer)
    N_out = neurons_per_layer[end]

    state_size = sum([length(get_neuron_states(l)) for l in layers])
    neur_states = zeros(state_size, 1)
    neur_outputs = zeros(N_neurons, 1)

    return Network(layers, N_in, N_out, neur_states, neur_outputs, state_size, 0.)
end

# Constructor for the Network which simply takes as input the layers in order from
#   first to last.
function Network(layers::Array{<:AbstractLayer, 1})
    in_layer = layers[1]
    N_in = 0
    if isa(in_layer.W, Matrix)
        N_in = size(in_layer.W)[2]
    elseif isa(in_layer.W, AbstractBlockArray)
        N_in = size(in_layer.W[Block(1,1)])[2]
    else
        error("Layer weights should be either a Matrix or BlockArray, given a $(typeof(in_layer.W))")
    end

    return Network(layers, N_in)
end

# Evolve the entire Network a duration `dt` starting from time `t` according to the
#   input `input`
function update!(network::Network, input, dt, t)
    first_idx = 1
    last_idx = network.N_in
    network.prev_outputs[1] .= input
    for (i,l) in enumerate(network.layers)
        network.prev_outputs[i+1] .= l.output
    end

    for i in 1:length(network.layers)
        update!(network.layers[i],network.prev_outputs,dt,t)
    end
end

# Reset the Network to its initial state.
function reset!(network::Network)
    network.neur_states = Array{Any, 2}(undef, network.state_size, 0)
    network.neur_outputs = Array{Any, 2}(undef, get_neuron_count(network), 0)
    network.t = 0.
    reset!.(network.layers)
    return nothing
end



# Count the number of neurons in the `Network`.
function get_neuron_count(network::Network)
    return sum(map((x)->x.N_neurons, network.layers))
end

# Get the state of each `Neuron` in the `Network` in a single array at the
#   current internal time step.
function get_neuron_states(network::Network)
    return vcat([get_neuron_states(l) for l in network.layers]...)
end

function get_neuron_outputs(network::Network)
    return vcat([get_neuron_outputs(l) for l in network.layers]...)
end
