# Layer, a collection of neurons of the same type being driven by some input vector
struct Layer{L<:AbstractNeuron, F<:Real, M<:Union{Matrix,AbstractBlockArray}}<:AbstractLayer
    neurons::Array{L,1}
    W::M # TODO: Make type union with sparse matrices, if sparse matrices end up efficient
    conns::Array{Int,1}

    N_neurons::Int

    input::Array{F,1}
    output::Array{F,1}

    # Default constructor, nonparametric. Mainly used for external calling to compute types
    function Layer(
        neurons::Array{L,1}, W::M, conns,  N_neurons, input::Array{F,1}, output::Array{F,1}
        ) where {L <: AbstractNeuron, F <: Real, M <: Union{Matrix,AbstractBlockArray}}

        return Layer{L, F, M}(neurons, W, conns, N_neurons, input, output)
    end

    # Default constructor, parametric. All Layers use this constructor eventually
    function Layer{L,F,M}(
        neurons::Array{L,1}, W::M, conns, N_neurons, input::Array{F,1}, output::Array{F,1} 
        ) where {L <: AbstractNeuron, F <: Real, M <: Union{Matrix, AbstractBlockArray}}

        if isempty(conns)
            conns = [0]
        end
        return new{L, F, M}(neurons, W, conns, N_neurons, input, output)
    end
end

# Cover the case where conns isn't specified (defaults to feed-forward), this only
#   support matrix-types (not block arrays) for W.
function Layer(neurons, W::Matrix)
    conns = Array{Int}(undef, 0)
    N_neurons = length(neurons)
    input = zeros(N_neurons)
    output = zeros(N_neurons)

    return Layer(neurons, W, conns, N_neurons, input, output)
end

# Cover the case where `conns` is specified, since we might not be feed-forward we 
#   permit both matrices and block arrays for W.
function Layer( neurons::Array{L,1}, W::M, conns
    ) where {L <: AbstractNeuron, F <: Real, M<:Union{Matrix,AbstractBlockArray}}

    N_neurons = length(neurons)
    input = zeros(N_neurons)
    output = zeros(N_neurons)
    return Layer(neurons, W, conns, N_neurons, input, output) 
end

# Evolve all of the neurons in the layer a duration `dt` starting at the time `t`
#   subject to an input from the previous layer `input`. 
#   Assumes a BlockArray W
function update!(l::Layer{L,F,M}, input, dt, t) where {L,F, M<:AbstractBlockArray}
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
