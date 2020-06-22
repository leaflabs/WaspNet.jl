"""
    LIF{T<:Number,A<:AbstractArray{T, 1}}<:AbstractNeuron 

Contains the necessary parameters for describing a Leaky Integrate-and-Fire (LIF) neuron as well as the current membrane potential of the neuron.

# Fields
- `τ::T`: Neuron time constant (ms)
- `R::T`: Neuronal model resistor (kOhms)
- `θ::T`: Threshold voltage (mV)
- `I::T`: Background current injection (mV)
- `v0::T`: Reset voltage (mV)
- `state::A`: Current membrane potential (mV)
"""
@with_kw struct LIF{T<:Number,A<:AbstractArray{T, 1}}<:AbstractNeuron 
    τ::T = 8.         
    R::T = 10.E3      
    θ::T = 30.      
    I::T = 40.      

    v0::T = -55.     
    state::A = [-55.]
    output::A = [0.]     
end

"""
    update!(neuron::LIF, input_update, dt, t)

Evolve and `LIF` neuron subject to a membrane potential step of size `input_update` a time duration `dt` starting from time `t`
"""
function update!(neuron::LIF, input_update, dt, t)
    neuron.output[1] = 0
    # If an impulse came in, add it
    neuron.state[1] += input_update

    # Euler method update
    neuron.state[1] += (dt/neuron.τ) * (-neuron.state[1] + neuron.R*neuron.I)

    # Check for thresholding
    if neuron.state[1] >= neuron.θ
        neuron.state[1] = neuron.v0
        neuron.output[1] = 1 # Binary output
    end

    return neuron.output[1] 
end

function reset!(neuron::LIF)
    neuron.state .= neuron.v0
end