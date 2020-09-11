"""
    mutable struct Network<:AbstractNetwork

Contains constituent `Layer`s, orchestrates the movement of signals between `Layer`s, and handles first-layer input.

# Fields
- `layers::Array{AbstractLayer,1}`: Array of `Layer`s ordered from 1 to N for N layers
- `prev_outputs::Vector`: Vector of vectors sized to hold the output from each `Layer` 
- `prev_events::Array{Bool, 1}: Vector of bools indicating which layers have spiked
"""
mutable struct Network<:AbstractNetwork
    layers::Array{AbstractLayer, 1}

    prev_outputs::Vector
    prev_events::Array{Bool, 1}

    function Network(layers)
        # Computer the number of signals passed out of each layer, where an input layer has 0 neurons
        neurons_per_layer = [num_neurons(l) for l in layers]
        output_sizes = vcat(neurons_per_layer)  

        N_layers = length(layers)
        N_neurons = sum(neurons_per_layer)

        net_layers = Array{AbstractLayer,1}()
        for (i,l) in enumerate(layers)
            if isa(l, Layer)
                conns = Array{Int,1}()
                if isempty(l.conns)
                    push!(conns, i-1)
                    new_layer = Layer(l.neurons, l.W, conns, l.N_neurons, l.state_size, l.input, l.output)
                else
                    conns = copy(l.conns)
                    new_layer = deepcopy(l)
                end
                push!(net_layers, new_layer)
            elseif isa(l, InputMatrixLayer)
                push!(net_layers, deepcopy(l))
            end
        end

        prev_outputs = [zeros(num_neurons(l)) for l in net_layers]

        return new(net_layers, prev_outputs, zeros(Bool, N_layers))
    end
end

function update!(net::Network, du, u, t)
    @inbounds for (j,l) in enumerate(net.layers)
        update!(l, du.x[j], u.x[j], t)
    end
end

function event(net::Network, u, t)
    @inbounds for (j,l) in enumerate(net.layers)
        net.prev_events[j] = event(l, u.x[j], t)
    end
    return any(net.prev_events)
end

function aff_element!(net::Network, u, input, t)
    @inbounds for (j,l) in enumerate(net.layers)
        copy!(net.prev_outputs[j], get_output(l))
    end
    @inbounds for (j,l) in enumerate(net.layers)
        aff_element!(l, u.x[j], net.prev_outputs, t)
    end
    for j in 1:length(net.prev_outputs)
        fill!(net.prev_outputs[j], 0)
    end
    fill!(net.prev_events, false)
end