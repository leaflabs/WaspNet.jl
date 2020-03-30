# Layer, a collection of neurons of the same type being driven by some input vector
struct Layer{L<:AbstractNeuron, F<:Real, M<:Union{Matrix,AbstractBlockArray}}<:AbstractLayer
    neurons::Array{L,1}
    output::Array{F,1}
    conns::Array{Int,1}
    W::M # TODO: Make type union with sparse matrices, if sparse matrices end up efficient
    N_neurons

    input::Array{F,1}

    # Default constructor, nonparametric
    function Layer(
        neurons::Array{L,1}, output::Array{F,1}, conns, W::M, N_neurons, input
        ) where {L <: AbstractNeuron, F <: Real, M <: Union{Matrix,AbstractBlockArray}}

        return Layer{L, F, M}(neurons, output, conns, W, N_neurons, input)
    end

    # Default constructor, parametric
    function Layer{L,F,M}(
        neurons::Array{L,1}, output::Array{F,1}, conns, W::M, N_neurons, input
        ) where {L <: AbstractNeuron, F <: Real, M <: Union{Matrix, AbstractBlockArray}}

        return new{L, F, M}(neurons, output, conns, W, N_neurons, input)
    end
end

# Cover the case where conns isn't specified (defaults to feed-forward), this only
#   support matrix-types (not block arrays) for W.
function Layer(neurons, output, W::Matrix, N_neurons)
    return Layer(neurons, output, Array{Int}(undef, 0), W, N_neurons)
end

# Cover the case where `conns` is specified, since we might not be feed-forward we 
#   permit both matrices and block arrays for W.
function Layer(
    neurons::Array{L,1}, output::Array{F,1}, conns, W::M, N_neurons
    ) where {L <: AbstractNeuron, F <: Real, M<:Union{Matrix,AbstractBlockArray}}

    input = zeros(length(neurons))
    return Layer(neurons, output, conns, W, N_neurons, input) 
end






# Evolve all of the neurons in the layer a duration `dt` starting at the time `t`
#   subject to an input from the previous layer `input`. 
#   Assumes a BlockArray W
function update!(l::Layer{L,F,M}, input, dt, t) where {L,F, M<:AbstractBlockArray}
    # nonzero_inputs = [!(all(input[i+1] .== 0)) for i in l.conns]
    l.input .= 0 # reset the input vector to the layer

    for conn in l.conns
        nonzero_input = !(all(input[conn+1] .== 0))
        if nonzero_input
            l.input .+= l.W[Block(1, conn+1)]*input[conn+1]
        end
    end

    l.output .= update!.(l.neurons, l.input, dt, t)
    return l.output
end

# Evolve all of the neurons in the layer a duration `dt` starting at the time `t`
#   subject to an input from the previous layer `input`. 
#   Assumes a Matrix W
function update!(l::Layer{L,F,M}, input, dt, t) where {L, F, M<:Matrix}
    for conn in l.conns
        nonzero_input = !(all(input[conn+1] .== 0))
        if nonzero_input
            l.input .= l.W*input[conn+1]
        end
    end

    l.output .= update!.(l.neurons, l.input, dt, t)
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
