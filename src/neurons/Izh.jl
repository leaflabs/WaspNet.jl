"""
    struct Izh{T<:Number}<:AbstractNeuronn

Contains the vector of paramters [a, b, c, d, I, θ] necessary to simulate an Izhikevich neuron as well as the current state of the neuron.

The @with_kw macro is used to produce a constructor which accepts keyword arguments for all values. This neuron struct is immutable, therefor we store the state of the neuron in an `Array` such that its values can change while the parameters remain static. This represents a minimal example for an `AbstractNeuron` implementation to build it into a `Layer`.

# Fields
- `a::T`-`d::T`: Neuron parameters as described at https://www.izhikevich.org/publications/spikes.htm
- `I::T`: Background current (mA)
- `θ::T`: Threshold potential (mV)
- `v0::T`: Reset voltage (mV)
- `u0::T`: Reset recovery variable value
- `state::T`: Vector holding the current (v,u) state of the neuron
- `output::T`: Vector holding the current output of the neuron
"""
@with_kw struct Izh{T<:Number}<:AbstractNeuron
    a::T = 0.02      
    b::T = 0.2
    c::T = -65.
    d::T = 8.
    I::T = 25.       
    θ::T = 30.       

    v0::T = -65.     # Reset voltage (mV)
end

"""
    update(neuron::Izh, input_update, dt, t)

Evolves the given `Neuron` subject to an input of `input_update` a time duration `dt` starting from time `t` according to the equations defined in the Izhikevich paper https://www.izhikevich.org/publications/spikes.htm

We use an Euler update for solving the set of differential equations for its computational efficiency and simplicity of implementation.
"""

function update(neuron::Izh, u, t)
    dv = 1000*(0.04 * u[1]^2 + 5*u[1] + 140 - u[2] + neuron.I)
    du = 1000*(neuron.a)*(neuron.b*u[1]-u[2])
    return (dv, du)
end

function aff_neuron!(neuron::Izh, u, input, t)
    u[1] += input;
end

function event(neuron::Izh, u, t)
    spike = u[1] > neuron.θ
    if spike 
        return (spike, 1)
    else
        return (spike, 0)
    end
end

function reset(neuron::Izh, u)
    return (neuron.v0, u[2] + neuron.d)
end
state_size(neuron::Izh) = 2