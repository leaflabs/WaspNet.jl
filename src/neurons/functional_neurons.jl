"""
    struct Functional{F,G<:Function}<:AbstractNeuron 

A neuron type which applies some scalar function to its input and returns that value as both its state and output.

# Fields
- `func::G`: A scalar function to apply to all inputs
- `state::Array{F,1}`: The last value computed by this neuron's function
"""
@with_kw struct Functional{F,G<:Function}<:AbstractNeuron 
    func::G
    state::Array{F,1} = [0.]
end

function update!(neuron::Functional, input_update, dt, t)
    neuron.state[1] = neuron.func(input_update)
end

function reset!(neuron::Functional)
    neuron.state[1] = 0
end

"""
    function ReLU()

A special `Functional` neuron with `ReLU` activation.
"""
function ReLU(; state = [0.], kwargs...)
    return Functional(func = (x) -> max(0, x), state = state, kwargs...)
end

"""
    function sigmoid()

A special `Functional` neuron with `sigmoid` activation.
"""
function sigmoid(; state = [0.], kwargs...)
    return Functional(func = (x) -> 1. /(1. +exp(-x)), state = state, kwargs...)
end

"""
    function tanh()

A special `Functional` neuron with `tanh` activation.
"""
function tanh(; state = [0.], kwargs...)
    return Functional(func = (x) -> Base.tanh(x), state = state, kwargs...)
end

"""
    function identity()

A special `Functional` neuron with identity activation; primarily used for testing.
"""
function identity(; state = [0.],  kwargs...)
    return Functional(func = (x) -> x, state = state, kwargs...)
end