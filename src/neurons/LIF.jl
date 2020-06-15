"""
    struct LIF{F}<:AbstractNeuron

Contains the necessary parameters for describing a Leaky Integrate-and-Fire (LIF) neuron as well as the current membrane potential of the neuron.

# Fields
- `τ::F`: Neuron time constant (ms)
- `R::F`: Neuronal model resistor (kOhms)
- `θ::F`: Threshold voltage (mV)
- `I::F`: Background current injection (mV)
- `v0::F`: Reset voltage (mV)
- `state::Array{F,1}`: Current membrane potential (mV)
"""
@with_kw struct LIF{F}<:AbstractNeuron
    τ::F = 8.         
    R::F = 10.E3      
    θ::F = 30.      
    I::F = 40.      

    v0::F = -55.     
    state::Array{F,1} = [-55.]     
end


# LIF time evolution step, using Euler method to ensure uniform time steps
"""
    update!(neuron::LIF, input_update, dt, t)

Evolve and `LIF` neuron subject to a membrane potential step of size `input_update` a time duration `dt` starting from time `t`
"""
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