# LIF Neuron
@with_kw mutable struct LIF{F}<:AbstractNeuron
    τ::F = 5.         # Time Constant (ms)
    R::F = 10.E3      # "Resistor" (kOhms)
    θ::F = 30.      # Threshold voltage (mV)
    v0::F = -55.     # Reset voltage (mV)
    I::F = 40.      # Background current injection (mA)
    v::F = -55.     # Membrane potential (mV)
end

function update!(neuron::LIF, input_update, dt, t) 
    retval = 0
    # If an impulse came in, add it
    neuron.v += input_update

    # Euler method updates to potential in 2 steps for numerical stability
    println("pre:", neuron.v)
    neuron.v += (dt/2 * 1/neuron.τ) * (-neuron.v + neuron.R*neuron.I)
    neuron.v += (dt/2 * 1/neuron.τ) * (-neuron.v + neuron.R*neuron.I)
    println("post:", neuron.v)

    # Check for thresholding
    if neuron.v >= neuron.θ
        neuron.v = neuron.v0
        retval = 1 # Binary output
    end

    return retval
end

function reset!(neuron::LIF)
    neuron.v = neuron.v0
end

# QIF Neuron
# @with_kw struct QIF<:AbstractNeuron
#     c
# end

# Izhikevich Neuron
@with_kw struct Izh{F}<:AbstractNeuron
   a::F = 0.02  
   b::F = 0.2
   c::F = -65.
   d::F = 8.
   I::F = 25.       # Background current injection (mA)
   v0::F = -65.     # Reset voltage (mV)
   θ::F = 30.       # Threshold potential (mV)

   v::F = -65.      # Membrane potential (mV)
   u::F = 0.        # Recovery variable
end

function update!(neuron::Izh, input_update, dt, t)
    retval = 0

    # Euler method updates to potential in 2 steps for numerical stability
    neuron.v += (dt/2)*(0.05 * neuronv.v^2 + 5*neuron.v + 140 - neuron.u + neuron.I)
    neuron.u += (dt/2)*(neuron.a)*(neuron.b*neuron.v-neuron.u)

    neuron.v += (dt/2)*(0.05 * neuronv.v^2 + 5*neuron.v + 140 - neuron.u + neuron.I)
    neuron.u += (dt/2)*(neuron.a)*(neuron.b*neuron.v-neuron.u)

    # Check for thresholding
    if neuron.v >= neuron.θ
        neuron.v = neuron.v0
        neuron.u += d
        retval = 1
    end

    return retval
end
