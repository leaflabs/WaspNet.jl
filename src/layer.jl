# Layer, a collection of neurons of the same type being driven by some input vector
struct Layer{L<:AbstractNeuron,F<:Real}<:AbstractLayer
    neurons::Array{L,1}
    output::Array{F,1}
    W # TODO: Make type union with sparse matrices, if sparse matrices end up efficient
    N_neurons
end

# Evolve all of the neurons in the layer a duration `dt` starting at the time `t`
#   subject to an input from the previous layer `input`.
function update!(l::Layer, input, dt, t)
    if any(input != 0)
        l.output .= update!.(l.neurons, l.W*input, dt, t) # this returns retval, either 0 or 1
    else
        l.output .= update!.(l.neurons, 0, dt, t)
    end

    return l.output
end

function reset!(l::AbstractLayer)
    reset!.(l.neurons)
end

# Get the state of each neuron in this layer
function get_neuron_states(l::AbstractLayer)
    return vcat([n.state for n in l.neurons]...)
end

# Get the output of each neuron at the current time in the layer
function get_neuron_outputs(l::AbstractLayer)
    return l.output
end

mutable struct Recurrent_Layer{L<:AbstractNeuron,F<:Real}<:AbstractLayer
    neurons::Array{L,1}
    output::Array{F,1}
    prev_output::Array{F,1}
    W # input weights
    W_re # recurrent weights
    N_neurons
end

function update!(l::Recurrent_Layer, input, dt, t)
    if any(input != 0)
        l.prev_output = l.output # I think this is what's causing issues
        l.output .= update!.(l.neurons, l.W*input + l.W_re*l.prev_output, dt, t)
    else
        l.prev_output = l.output
        l.output .= update!.(l.neurons, l.W_re*l.prev_output, dt, t)
    end
end
