# Layer, a collection of neurons being driven by the same input vector
struct Layer{L<:AbstractNeuron}<:AbstractNetwork
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