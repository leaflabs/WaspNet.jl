# Izhikevich Neuron, using Euler method to ensure uniform time step
@with_kw struct Izh{F}<:AbstractNeuron
    a::F = 0.02      # a-d are model parameters
    b::F = 0.2
    c::F = -65.
    d::F = 8.
    I::F = 25.       # Background current injection (mA)
    θ::F = 30.       # Threshold potential (mV)

    v0::F = -65.     # Reset voltage (mV)
    u0::F = 0.       # Reset state variable
    state::Array{F,1} = [-65., 0.]      # Membrane potential (mV) and state variable
end

function update!(neuron::Izh, input_update, dt, t)
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
        neuron.state .= [ neuron.v0, neuron.state[2] + neuron.d]
        retval = 1
    end

    return retval
end

function reset!(neuron::Izh)
    neuron.state .= [neuron.v0, neuron.u0]
end