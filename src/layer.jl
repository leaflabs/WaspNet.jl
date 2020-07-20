"""
    Layer{
        L<:AbstractNeuron, N<:Number, A<:AbstractArray{N,1}, M<:Union{AbstractArray{N,2}, Array{AbstractArray{N,2},1}
        }<:AbstractLayer

Track a population of neurons of one `AbstractNeuron` type, the other `Layer`s those neurons are connected to, and the incoming weights. 

# Fields
- `neurons::Array{L,1}`: an array of neurons for the `Layer`
- `W<:Union{Matrix,AbstractBlockArray}`: either a Matrix or BlockArray containing weights for inputs from incoming layers
- `conns`: either `[]` or `Array{Int,1}` indicating which `Layer`s in the `Network` are connected as inputs to this `Layer`
- `input::Array{N,1}`: a pre-allocated array of zeros for staging inputs to the layer
- `output::Array{N,1}`: a pre-allocated array for staging outputs from this layer
"""
struct Layer{
    L<:AbstractNeuron, N<:Number, A<:AbstractArray{N,1}, M<:Union{AbstractArray{N,2}, Array{<:AbstractArray{N,2},1}} 
    }<:AbstractLayer
    neurons::Array{L,1}
    W::M 
    conns::Array{Int,1}
    N_neurons::Int
    input::A
    output::A

    # Default constructor, parametric. All Layers use this constructor eventually
    function Layer{L,N,A,M}(
        neurons::Array{L,1}, W::M, conns::Array{Int,1}, N_neurons::Int, input::Array{N,1}, output::Array{N,1} 
        ) where {L<:AbstractNeuron, N<:Number, A<:AbstractArray{N,1}, M<:Union{AbstractArray{N,2}, Array{<:AbstractArray{N,2},1}}}

        return new{L,N,A,M}(neurons, W, conns, N_neurons, input, output)
    end
end

"""
    Layer(neurons, W, conns, N_neurons, input, output)

Default non-parametric constructor for `Layer`s for pre-processing inputs and computing parametric types.
"""
function Layer(
    neurons::Array{L,1}, W::M, conns::Array{J,1}, N_neurons::Int, input::A, output::A
    ) where {J, L<:AbstractNeuron, N<:Number, A<:AbstractArray{N,1}, M<:Union{AbstractArray{N,2}, Array{<:AbstractArray{N,2},1}}}

    if isempty(conns)
        conns = Array{Int,1}()
    end
    return Layer{L,N,A,M}(neurons, W, conns, N_neurons, input, output)
end

"""
    Layer(neurons, W[, conns = Array{Int,1}()])

Constructs a `Layer` with constituent `neurons` which accept inputs from the `Layer`s denoted by `conns` (input 1 is the `Network` input) and either a `BlockArray` of weights if `length(conns) > 1` or a Matrix of weights otherwise.
"""
function Layer(neurons, W, conns = Array{Int, 1}(undef, 0))

    N_neurons = length(neurons)
    input = zeros(N_neurons)
    output = zeros(N_neurons)
    return Layer(neurons, W, conns, N_neurons, input, output) 
end

function Layer(neurons, W::AbstractBlockArray)
    error("`Layer(neurons, W)` construction not available for `AbstractBlockArray`s; feed-forward only")
end

"""
    function update!(l::Layer, input, dt, t)

Evolve the state of all of the neurons in the `Layer` a duration `dt`, starting from time `t`, subject
to a set of inputs from all `Network` layers in `input`.

This (default) method assumes a feed-forward, non-BlockArray representation for `l.W`

# Arguments
- `l::Layer`: the `Layer` to be evolved
- `input`: an `Array` of `Array`s of output values from other `Layers` potentially being input to `l`
- `dt`: the time step to evolve the `Layer`
- `t`: the time at the start of the current time step
"""
function update!(l::Layer{L,N,A,M}, input, dt, t) where {L,N,A,M<:AbstractArray{N,2}}
    if isempty(l.conns) 
        mul!(l.input, l.W, input[1])
    elseif !isempty(l.conns)
        conn = l.conns[1] # should only have one connection
        mul!(l.input, l.W, input[conn+1])
    end

    for j in 1:length(l.neurons)
        (l.output[j], l.neurons[j]) = update!(l.neurons[j], l.input[j], dt, t)
    end
    return l.output
end

"""
    function update!(l::Layer{L,N,A,M}, input, dt, t)

Evolve the state of all of the neurons in the `Layer` a duration `dt`, starting from time `t`, subject to a set of inputs from all `Network` layers in `input`. 

Not all arrays within `input` are used; we iterate over `l.conn` to select the appropriate inputs to this `Layer`, and the corresponding `Block`s from `l.W` are used to calculate the net `Layer` input.
"""
function update!(l::Layer{L,N,A,M}, input, dt, t) where {L,N,A,M<:AbstractBlockArray}
    for conn in l.conns
        mul!(l.input, l.W[Block(1, conn+1)], input[conn+1], 1, 1)
    end
    for j in 1:length(l.neurons)
        (l.output[j], l.neurons[j]) = update!(l.neurons[j], l.input[j], dt, t)
    end
    return l.output
end

"""
    function update!(l::Layer{L,N,A,M}, input, dt, t) where {L,N,A, M<:AbstractArray{T,1}}

Evolve the state of all of the neurons in the `Layer` a duration `dt`, starting from time `t`, subject to a set of inputs from all `Network` layers in `input`. 

Not all arrays within `input` are used; we iterate over `l.conn` to select the appropriate inputs to this `Layer`, and the corresponding `Block`s from `l.W` are used to calculate the net `Layer` input.
"""
function update!(l::Layer{L,N,A,M}, input, dt, t) where {L,N,A,M<:AbstractArray{<:AbstractArray,1}}
    for (conn,W) in zip(l.conns, l.W)
        mul!(l.input, W, input[conn+1], 1, 1)
    end
    for j in 1:length(l.neurons)
        (l.output[j], l.neurons[j]) = update!(l.neurons[j], l.input[j], dt, t)
    end
    return l.output
end

"""
    reset!(l::AbstractLayer)

Reset all of the neurons in `l` to the state defined by their `reset!` function.
"""
function reset!(l::AbstractLayer)
    l.neurons .= reset.(l.neurons)
end

"""
    get_neuron_states(l::AbstractLayer)

Return the current state of `l`'s constituent neurons
"""
function get_neuron_states(l::AbstractLayer)
    return vcat([get_neuron_states(n) for n in l.neurons]...)
end

"""
    get_neuron_outputs(l::AbstractLayer)

Return the current output of `l`'s constituent neurons 
"""
function get_neuron_outputs(l::AbstractLayer)
    return l.output
end

"""
    get_neuron_count(l::AbstractLayer)

Return the number of neurons in the given `Layer`
"""
function get_neuron_count(l::AbstractLayer)
    return length(l.neurons)
end
