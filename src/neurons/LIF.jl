"""
    LIF{T<:Number}<:AbstractNeuron 

Contains the necessary parameters for describing a Leaky Integrate-and-Fire (LIF) neuron.

# Fields
- `τ::T`: Neuron time constant (ms)
- `R::T`: Neuronal model resistor (kOhms)
- `θ::T`: Threshold voltage (mV)
- `I::T`: Background current injection (mV)
- `v0::T`: Reset voltage (mV)
"""
@with_kw struct LIF{T<:Number}<:AbstractNeuron 
    τ::T = 4.         
    R::T = 6.      
    θ::T = 30.      
    I::T = 40.      

    v0::T = -55.     
end

"""
    update(neuron::LIF, input_update, dt, t)

Evolve and `LIF` neuron subject to a membrane potential step of size `input_update` a time duration `dt` starting from time `t`
"""
function update(neuron::LIF, u, t)
    dv = (-u[1] + neuron.R*neuron.I) / neuron.τ
    return (dv, )
end
function event(neuron::LIF, u, t)
    u[1] > neuron.θ
end

function reset(neuron::LIF)
    return (neuron.v0, )
end

function state_size(neuron::LIF)
    return 1
end