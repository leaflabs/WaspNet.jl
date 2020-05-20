# LIF Neuron
@with_kw struct LIF{F}<:AbstractNeuron
    τ::F = 8.         # Time Constant (ms)
    R::F = 10.E3      # "Resistor" (kOhms)
    θ::F = 30.      # Threshold voltage (mV)
    I::F = 40.      # Background current injection (mA)

    v0::F = -55.     # Reset voltage (mV)
    state::Array{F,1} = [-55.]     # Membrane potential (mV)
end


# LIF time evolution step, using Euler method to ensure uniform time steps
function update!(neuron::LIF, input_update, dt, t)
    retval = 0
    # If an impulse came in, add it
    neuron.state[1] += input_update

    # Euler method update
    neuron.state[1] += (dt/neuron.τ) * (-neuron.state[1] + neuron.R*neuron.I)

    # Check for thresholding
    if neuron.state[1] >= neuron.θ
        neuron.state[1] = neuron.v0
        retval = 1 # Binary output
    end

    return retval
end

function reset!(neuron::LIF)
    neuron.state .= neuron.v0
end