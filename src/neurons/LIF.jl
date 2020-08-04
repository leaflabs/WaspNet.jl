"""
    LIF{T<:Number}<:AbstractNeuron 

Contains the necessary parameters for describing a Leaky Integrate-and-Fire (LIF) neuron as well as the current membrane potential of the neuron.

# Fields
- `τ::T`: Neuron time constant (ms)
- `R::T`: Neuronal model resistor (kOhms)
- `θ::T`: Threshold voltage (mV)
- `I::T`: Background current injection (mV)
- `v0::T`: Reset voltage (mV)
- `state::T`: Current membrane potential (mV)
"""
@with_kw struct LIF{T<:Number}<:AbstractNeuron 
    τ::T = 4.         
    R::T = 6.      
    θ::T = 30.      
    I::T = 40.      

    v0::T = -55.     
    state::T = -55.
    output::T = 0.     
end

"""
    update!(neuron::LIF, input_update, dt, t)

Evolve and `LIF` neuron subject to a membrane potential step of size `input_update` a time duration `dt` starting from time `t`
"""
function update(neuron::LIF, input_update, dt, t)
    output = 0.
    # If an impulse came in, add it
    state = neuron.state + input_update

    # Euler method update
    state += (dt/neuron.τ) * (-state + neuron.R*neuron.I)

    # Check for thresholding
    if state >= neuron.θ
        state = neuron.v0
        output = 1. # Binary output
    end

    return (output, LIF(neuron.τ, neuron.R, neuron.θ, neuron.I, neuron.v0, state, output))
end

function reset(neuron::LIF)
    return LIF(neuron.τ, neuron.R, neuron.θ, neuron.I, neuron.v0, neuron.v0, 0.)
end