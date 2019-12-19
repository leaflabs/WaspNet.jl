mutable struct Network<:AbstractNetwork
    layers::Array{<:AbstractLayer, 1}  # Array of layers in order from input to output
    N_in::Int                          # Number of input dimensions
    N_out::Int                         # Number of output dimensions
    neur_states::Matrix                # The outputs of each neuron for each time step
    t                                  # Internal time parameter
    track_flag::Bool                   # Flag to track the states of all neurons
end

function Network(layers::Array{<:AbstractLayer, 1}, track_flag = false)
    N_in = size(layers[1].W)[2] # Number of dimensions in the input space
    N_out = size(layers[end].W)[1] # Number of output dimensions
    N_neurons = sum(map(l -> l.N_neurons, layers))

    return Network(layers, N_in, N_out, zeros(N_neurons, 1), 0.0, track_flag)
end

function update!(network::Network, input, dt, t)
    in_vec = input
    out_vec = foldl(
        (prev,layer)-> update!(layer, prev, dt, t), network.layers, init=input
        )
    return out_vec
end

function reset!(network::Network)
    reset!.(network.layers)
    network.state = zeros(N_neurons, 1)
    return nothing
end

function simulate!(network::Network, input, dt, t_total)
    t_steps = 0:dt:t_total
    N_steps = length(t_steps)
    network.neur_states = zeros(get_neuron_count(network), N_steps)
    for (i,t) in zip(1:N_steps,t_steps)
        network.neur_states[:,i] = update!(network, input, dt, t)
    end
    return network.neur_states
end


function get_neuron_count(network::Network)
    return sum(map((x)->x.N_neurons, network.layers))
end

function get_neuron_states(network::Network)
    return vcat([get_neuron_states(l) for l in network.layers]...)
end
