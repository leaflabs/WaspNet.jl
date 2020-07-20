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
    u0::T = 0.       # Reset state variable
    v::T = -65.      # Membrane potential (mV) 
    u::T = 0.        # state variable 
    output::T = 0.
end

"""
    update!(neuron::Izh, input_update, dt, t)

Evolves the given `Neuron` subject to an input of `input_update` a time duration `dt` starting from time `t` according to the equations defined in the Izhikevich paper https://www.izhikevich.org/publications/spikes.htm

We use an Euler update for solving the set of differential equations for its computational efficiency and simplicity of implementation.
"""
function update!(neuron::Izh, input_update, dt, t)
    dt *= 1000. # convert seconds to milliseconds for the Izh model
    output = 0.
    # If an impulse came in, add it
    v = neuron.v + input_update
    u = neuron.u

    # Euler method update
    dv = dt*(
      0.04 * v^2 + 5*v + 140 - u + neuron.I
      )
    du = dt*(neuron.a)*(neuron.b*v-u)
    v += dv
    u += du

    # Check for thresholding
    if v >= neuron.θ
        v = neuron.v0
        u = u + neuron.d
        output = 1.
    end

    return (output, Izh(neuron.a, neuron.b, neuron.c, neuron.d, neuron.I, neuron.θ, neuron.v0, neuron.u0, v, u, output))
end

"""
    reset!(neuron::Izh)

Resets the state of the Izhikevich neuron to its initial values given by `v0`, `u0`
"""
function reset(neuron::Izh)
    return Izh(
        neuron.a, neuron.b, neuron.c, neuron.d, neuron.I, neuron.θ, neuron.v0, neuron.u0, neuron.v, neuron.u, 0.
        )
end

function get_neuron_states(neuron::Izh)
    return (neuron.v, neuron.u)
end