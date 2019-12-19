# Layer, a collection of neurons of the same type being driven by some input vector
struct Layer{L<:AbstractNeuron}<:AbstractLayer
    neurons::Array{L,1}
    W # TODO: Make type union with sparse matrices
    N_neurons
end

# TODO: A constructor for Layer which allows you to specify arguments to the neuron

function update!(l::Layer, input, dt, t)
    input_to_neurons = l.W*input
    out_state = update!.(l.neurons, input_to_neurons, dt, t)

    return out_state
end

function reset!(l::Layer)
    reset!.(l.neurons)
end

function get_neuron_states(l::Layer)
    return vcat([n.state for n in l.neurons]...)
end