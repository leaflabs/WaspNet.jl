"""
    mutable struct Network<:AbstractNetwork

Contains constituent `Layer`s, orchestrates the movement of signals between `Layer`s, and handles first-layer input.

# Fields
- `layers::Array{AbstractLayer,1}`: Array of `Layer`s ordered from 1 to N for N layers
- `N_in::Int`: Number of input dimensions to the first `Layer`
- `prev_outputs::Vector`: Vector of vectors sized to hold the output from each `Layer` 
"""
mutable struct Network<:AbstractNetwork
    layers::Array{AbstractLayer, 1}
    N_in::Int

    prev_outputs

    function Network(layers, N_in, prev_outputs)
        neurons_per_layer = [length(l.neurons) for l in layers]
        output_sizes = vcat([N_in], neurons_per_layer) # number of signals passed out of each layer incl. input

        N_layers = length(layers)
        N_neurons = sum(neurons_per_layer)

        net_layers = []
        for (i,l) in enumerate(layers)
            N_layer_neurons = neurons_per_layer[i]
            
            W = BlockArray(zeros(N_layer_neurons, N_neurons + N_in), [N_layer_neurons], output_sizes)
            conns = Array{Int,1}()
            if isempty(l.conns)
                push!(conns, i-1)
                setblock!(W, l.W, 1, i)
                new_layer = Layer(l.neurons, W, conns, l.N_neurons, l.input, l.output)
            else
                conns = copy(l.conns)
                W = copy(l.W)
                new_layer = deepcopy(l)
            end
            push!(net_layers, new_layer)
        end

        return new(net_layers, N_in, prev_outputs)
    end
end

"""
    function Network(layers, N_in::Int) 

Given an array of `Layer`s and the dimensionality of the input to the network, make a new `Network` which is a copy of each `Layer` with weights converted to `BlockArray` format.

The output dimensionality is in 
""" 
function Network(layers, N_in::Int) 
    neurons_per_layer = [length(l.neurons) for l in layers]
    prev_outputs = [zeros(N_in), [zeros(j) for j in neurons_per_layer]...]

    return Network(layers, N_in, prev_outputs)
end

"""
    function Network(layers::Array{L, 1}) where L <: AbstractLayer

Given an array of `Layer`s, constructs the `Network` resulting from connecting the `Layer`s with their specified `conn`s. 

The input dimensionality is inferred from the size of the weight matrices for the first `Layer` in the `layers` array.
"""
function Network(layers::Array{L, 1}) where L <: AbstractLayer
    in_layer = layers[1]
    N_in = 0
    if isa(in_layer.W, AbstractBlockArray)
        N_in = size(in_layer.W[Block(1,1)])[2]
    elseif isa(in_layer.W, AbstractArray{<:Number,2})
        N_in = size(in_layer.W)[2]
    else
        error("Layer weights should be a subtype of AbstractArray{<:Number,2}, given a $(typeof(in_layer.W))")
    end

    return Network(layers, N_in)
end

function update!(network::Network, input, dt, t)
    copy!(network.prev_outputs[1], input)
    for (i,l) in enumerate(network.layers)
        copy!(network.prev_outputs[i+1], l.output)
    end

    for i in 1:length(network.layers)
        update!(network.layers[i],network.prev_outputs,dt,t)
    end
end

function reset!(network::Network)
    reset!.(network.layers)
    return nothing
end

function get_neuron_count(network::Network)
    return sum(map((x)->x.N_neurons, network.layers))
end

function get_neuron_states(network::Network)
    return vcat([get_neuron_states(l) for l in network.layers]...)
end

function get_neuron_outputs(network::Network)
    return vcat([get_neuron_outputs(l) for l in network.layers]...)
end