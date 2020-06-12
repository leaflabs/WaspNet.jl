"""
    Layer{L<:AbstractNeuron, F<:Real, M<:Union{Matrix,AbstractBlockArray}}<:AbstractLayer

Track a population of neurons of one `AbstractNeuron` type, the other `Layer`s those neurons are 
connected to, and the incoming weights. 
"""
struct Layer{L<:AbstractNeuron, F<:Real, M<:Union{Matrix,AbstractBlockArray}}<:AbstractLayer
    neurons::Array{L,1}
    W::M 
    conns::Array{Int, 1}
    N_neurons::Int
    input::Array{F,1}
    output::Array{F,1}

    # Default constructor, parametric. All Layers use this constructor eventually
    function Layer{L,F,M}(
        neurons::Array{L,1}, W::M, conns::Array{Int,1}, N_neurons::Int, input::Array{F,1}, output::Array{F,1} 
        ) where {L <: AbstractNeuron, F <: Real, M <: Union{Matrix, AbstractBlockArray}}

        return new{L, F, M}(neurons, W, conns, N_neurons, input, output)
    end
end

"""
    Layer(neurons, W, conns, N_neurons, input, output)

Default non-parametric constructor for `Layer`s for pre-processing inputs and computing parametric types.

# Arguments
- `neurons::Array{L,1}`: an array of neurons for the `Layer`
- `W<:Union{Matrix,AbstractBlockArray}`: either a Matrix or BlockArray containing weights for inputs from incoming layers
- `conns`: either [] or Array{Int,1} indicating which `Layer`s in the `Network` are connected as inputs to this `Layer`
- `input::Array{<:Real,1}`: a pre-allocated array of zeros for staging inputs to the layer
- `output::Array{<:Real,1}`: a pre-allocated array for staging outputs from this layer
"""
function Layer(
    neurons::Array{L,1}, W::M, conns,  N_neurons, input::Array{F,1}, output::Array{F,1}
    ) where {L <: AbstractNeuron, F <: Real, M <: Union{Matrix,AbstractBlockArray}}

    if isempty(conns)
        conns = Array{Int,1}()
    end

    return Layer{L, F, M}(neurons, W, conns, N_neurons, input, output)
end

"""
    Layer(neurons, W::Matrix)

Constructs a `Layer` with constituent `neurons` which accept a feed-forward input with a matrix of weights `W` 
"""
function Layer(neurons, W::Matrix)
    conns = Array{Int}(undef, 0)
    N_neurons = length(neurons)
    input = zeros(N_neurons)
    output = zeros(N_neurons)

    return Layer(neurons, W, conns, N_neurons, input, output)
end

"""
    Layer( neurons::Array{L,1}, W::M, conns) where {L <: AbstractNeuron, F <: Real, M<:Union{Matrix,AbstractBlockArray}}

Constructs a `Layer` with constituent `neurons` which accept inputs from the `Layer`s denoted by `conns` (input 1 is the `Network` 
input) and a `BlockArray` of weights if `length(conns) > 1` or a Matrix of weights otherwise.
"""
function Layer( neurons::Array{L,1}, W::M, conns
    ) where {L <: AbstractNeuron, F <: Real, M<:Union{Matrix,AbstractBlockArray}}

    N_neurons = length(neurons)
    input = zeros(N_neurons)
    output = zeros(N_neurons)
    if size(W)[1] !== N_neurons
        error("Weight Matrix and Neuron Count must be the same")
    end
    return Layer(neurons, W, conns, N_neurons, input, output) 
end

"""
    function update!(l::Layer{L,F,M}, input, dt, t) where {L,F, M<:AbstractBlockArray}

Evolve the state of all of the neurons in the `Layer` a duration `dt`, starting from time `t`, subject
to a set of inputs from all `Network` layers in `input`. 

Not all arrays within `input` are used; we iterate over `l.conn` to select the appropriate inputs to this `Layer`, and the corresponding
`Block`s from `l.W` are used to calculate the net `Layer` input.

# Arguments
- `l::Layer{L,F,M}`: the `Layer` to be evolved
- `input`: an `Array` of `Array`s of output values from other `Layers` potentially being input to `l`
- `dt`: the time step to evolve the `Layer`
- `t`: the time at the start of the current time step
"""
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

"""
    function update!(l::Layer{L,F,M}, input, dt, t) where {L,F, M<:Matrix}

Evolve the state of all of the neurons in the `Layer` a duration `dt`, starting from time `t`, subject
to a set of inputs from all `Network` layers in `input`. 

# Arguments
- `l::Layer{L,F,M}`: the `Layer` to be evolved
- `input`: an `Array` of `Array`s of output values from other `Layers` potentially being input to `l`
- `dt`: the time step to evolve the `Layer`
- `t`: the time at the start of the current time step
"""
function update!(l::Layer{L,F,M}, input, dt, t) where {L, F, M<:Matrix}
    if isempty(l.conns) && !(all(input[1] .== 0))
        l.input .= l.W*input[1]
    elseif !isempty(l.conns)
        conn = l.conns[1] # should only have one connection
        nonzero_input = !(all(input[conn+1] .== 0))
        if nonzero_input
            l.input .= l.W*input[conn+1]
        end
    end

    l.output .= update!.(l.neurons, l.input, dt, t)
    return l.output
end

"""
    reset!(l::AbstractLayer)

Reset all of the neurons in `l` to the state defined by their `reset!` function.
"""
function reset!(l::AbstractLayer)
    reset!.(l.neurons)
end

"""
    get_neuron_states(l::AbstractLayer)

Return the current state of `l`'s constituent neurons
"""
function get_neuron_states(l::AbstractLayer)
    return vcat([n.state for n in l.neurons]...)
end

"""
    get_neuron_outputs(l::AbstractLayer)

Return the current output of `l`'s constituent neurons 
"""
function get_neuron_outputs(l::AbstractLayer)
    return l.output
end
