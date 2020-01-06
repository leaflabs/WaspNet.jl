# Layer, a collection of neurons of the same type being driven by some input vector
struct Layer{L<:AbstractNeuron}<:AbstractLayer
    neurons::Array{L,1}
    W # TODO: Make type union with sparse matrices, if sparse matrices end up efficient
    N_neurons
end

# Evolve all of the neurons in the layer a duration `dt` starting at the time `t`
#   subject to an input from the previous layer `input`.
function update!(l::Layer, input, dt, t)
    input_to_neurons = l.W*input
    out_state = update!.(l.neurons, input_to_neurons, dt, t)

    return out_state
end

function reset!(l::Layer)
    reset!.(l.neurons)
end

# Get the state of each neuron in this layer
function get_neuron_states(l::Layer)
    return vcat([n.state for n in l.neurons]...)
end