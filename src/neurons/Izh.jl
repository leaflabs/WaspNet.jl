"""
    struct Izh{F}<:AbstractNeuron

Contains the vector of paramters [a, b, c, d, I, θ] necessary to simulate an Izhikevich neuron as well as the current state of the neuron.

The @with_kw macro is used to produce a constructor which accepts keyword arguments for all values. This neuron struct is immutable, therefor we store the state of the neuron in an `Array` such that its values can change while the parameters remain static. This represents a minimal example for an `AbstractNeuron` implementation to build it into a `Layer`.

# Fields
- `a::F`-`d::F`: Neuron parameters as described at https://www.izhikevich.org/publications/spikes.htm
- `I::F`: Background current (mA)
- `θ::F`: Threshold potential (mV)
- `v0::F`: Reset voltage (mV)
- `u0::F`: Reset recovery variable value
- `state::Array{F,1}`: Vector holding the current (v,u) state of the neuron
"""
@with_kw struct Izh{F}<:AbstractNeuron
    a::F = 0.02      
    b::F = 0.2
    c::F = -65.
    d::F = 8.
    I::F = 25.       
    θ::F = 30.       

    v0::F = -65.     # Reset voltage (mV)
    u0::F = 0.       # Reset state variable
    state::Array{F,1} = [-65., 0.]      # Membrane potential (mV) and state variable
end

"""
    update!(neuron::Izh, input_update, dt, t)

Evolves the given `Neuron` subject to an input of `input_update` a time duration `dt` starting from time `t` according to the equations defined in the Izhikevich paper https://www.izhikevich.org/publications/spikes.htm

We use an Euler update for solving the set of differential equations for its computational efficiency and simplicity of implementation.
"""
function update!(neuron::Izh, input_update, dt, t)
    dt *= 1000. # convert seconds to milliseconds for the Izh model
    retval = 0
    # If an impulse came in, add it
    neuron.state[1] += input_update

    # Euler method update
    u1 = dt*(
      0.04 * neuron.state[1]^2 + 5*neuron.state[1] + 140 - neuron.state[2] + neuron.I
      )
    u2 = dt*(neuron.a)*(neuron.b*neuron.state[1]-neuron.state[2])
    neuron.state[1] += u1
    neuron.state[2] += u2

    # Check for thresholding
    if neuron.state[1] >= neuron.θ
        neuron.state[1] = neuron.v0
        neuron.state[2] = neuron.state[2] + neuron.d
        retval = 1
    end

    return retval
end

"""
    reset!(neuron::Izh)

Resets the state of the Izhikevich neuron to its initial values given by `v0`, `u0`
"""
function reset!(neuron::Izh)
    neuron.state[1] = neuron.v0
    neuron.state[2] = neuron.u0
end