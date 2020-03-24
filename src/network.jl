
# A Network comprises layers and handles passing inputs between layers. Furthemore,
#   it also tracks the states of all of the neurons at each time step.
mutable struct Network<:AbstractNetwork
    layers::Array{AbstractLayer, 1}  # Array of layers in order from input to output
    N_in::Int                          # Number of input dimensions
    N_out::Int                         # Number of output dimensions
    neur_states::Matrix                # The states of each neuron for each time step
    neur_outputs::Matrix               # The outputs of each neuron for each time step
    state_size::Int
    t                                  # Internal time parameter

    function Network(layers, N_in, N_out, neur_states, neur_outputs, state_size, t)
        neurons_per_layer = [length(l.neurons) for l in layers]
        output_sizes = vcat([N_in], neurons_per_layer) # number of signals passed out of each layer incl. input
    
        N_layers = length(layers)
        N_neurons = sum(neurons_per_layer)

        net_layers = []
        for (i,l) in enumerate(layers)
            N_layer_neurons = length(l.neurons)
            # TODO: maybe convert this to PseudoBlockArray
            W = BlockArray(zeros(N_layer_neurons, N_neurons + N_in), [N_layer_neurons], output_sizes)
            conns = []
            if isempty(l.conns)
                push!(conns, i-1)
                setblock!(W, l.W, 1, i)
            else
                W = copy(l.W)
            end
            new_layer = deepcopy_field_update(l, [:conns, :W], [conns, W])
            push!(net_layers, new_layer)
        end

        return new(net_layers, N_in, N_out, neur_states, neur_outputs, state_size, 0.)
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
    elseif isa(in_layer.W, AbstractBlockArray )
        N_in = size(in_layer.W[Block(1,1)])[2]
    else
        error("Layer weights should be either a Matrix or BlockArray, given a $(typeof(in_layer.W))")
    end

    return Network(layers, N_in)
end

# Evolve the entire Network a duration `dt` starting from time `t` according to the
#   input `input`
function update!(network::Network, input, dt, t)
    prev_out = vcat([input],[l.output for l in network.layers])
    for i in 1:length(network.layers)
        update!(network.layers[i],prev_out,dt,t)
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

# Simulate the network from `t0` to `tf` with a time step of `dt` with an input to
#   the first layer of `input`
function simulate!(network::Network, input, dt, tf, t0 = 0.; track_flag = false)
    # t_steps are time points where we evaluate the input function
    # There are ((tf-t0)/dt)+1 time steps, including t0 as the first time step
    # Thus, we evolve the network assuming it is evaluated at the initial time step for
    # every time step, meaning the network ends with t=tf but the last evaluation of input
    # happens at time t=tf-dt
    t_steps = t0:dt:(tf-dt)
    N_steps = length(t_steps)

    # The +1 here is to ensure that we get the initial state at t=t0 in the outputs
    network.neur_outputs = Array{Any, 2}(undef, get_neuron_count(network), N_steps+1) 
    network.neur_outputs[:, 1] = get_neuron_outputs(network) 
    if track_flag
        network.neur_states = Array{Any, 2}(undef, network.state_size, N_steps+1)
        network.neur_states[:,1] .= get_neuron_states(network)
    end

    for (i,t) in zip(1:N_steps,t_steps)
        update!(network, input(t), dt, t)
        # The +1 here is the offset from including the initial values in the outputs
        network.neur_outputs[:, i+1] = get_neuron_outputs(network)
        if track_flag
            network.neur_states[:,i+1] .= get_neuron_states(network)
        end
        network.t += dt
    end

    if track_flag
        return network.neur_outputs, network.neur_states
    else
        return network.neur_outputs
    end
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
