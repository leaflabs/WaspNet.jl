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

function ReLU(; state = [0.], kwargs...)
    return Functional(func = (x) -> max(0, x), state = state, kwargs...)
end

function sigmoid(; state = [0.], kwargs...)
    return Functional(func = (x) -> 1. /(1. +exp(-x)), state = state, kwargs...)
end

function tanh(; state = [0.], kwargs...)
    return Functional(func = (x) -> Base.tanh(x), state = state, kwargs...)
end

function identity(; state = [0.],  kwargs...)
    return Functional(func = (x) -> x, state = state, kwargs...)
end