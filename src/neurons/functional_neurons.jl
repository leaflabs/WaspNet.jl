"""
    struct Functional{T<:Number, F<:Function}<:AbstractNeuron 

A neuron type which applies some scalar function to its input and returns that value as both its state and output.

# Fields
- `func::F`: A scalar function to apply to all inputs
- `state::T`: The last value computed by this neuron's function
"""
@with_kw struct Functional{T<:Number, F<:Function}<:AbstractNeuron 
    func::F
    state::T = 0.
end

function update!(neuron::Functional, input_update, dt, t)
    state = neuron.func(input_update)
    return (state, Functional(neuron.func, state))
end

function reset(neuron::Functional)
    return Functional(neuron.func, 0)
end

function get_neuron_outputs(n::Functional{T,F}) where {T,F}
    return neuron.state
end

function Functional(f::F; state::T = 0.) where {T<:Number, F<:Function}
    return Functional{T,F}(f, state)
end 

"""
    function ReLU()

A special `Functional` neuron with `ReLU` activation.
"""
function ReLU(; state = 0., kwargs...)
    return Functional((x) -> max(0, x), state=state)
end

"""
    function sigmoid()

A special `Functional` neuron with `sigmoid` activation.
"""
function sigmoid(; state = 0., kwargs...)
    return Functional((x) -> 1. /(1. +exp(-x)), state = state)
end

"""
    function tanh()

A special `Functional` neuron with `tanh` activation.
"""
function tanh(; state = 0., kwargs...)
    return Functional((x) -> Base.tanh(x), state = state)
end

"""
    function identity()

A special `Functional` neuron with identity activation; primarily used for testing.
"""
function identity(; state = 0.,  kwargs...)
    return Functional((x) -> x, state = state)
end