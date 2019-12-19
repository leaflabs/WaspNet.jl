

# LIF Neuron
@with_kw struct LIF{F}<:AbstractNeuron
    τ::F = 8.         # Time Constant (ms)
    R::F = 10.E6      # "Resistor" (kOhms)
    θ::F = 30.      # Threshold voltage (mV)
    I::F = 40.      # Background current injection (mA)

    v0::Array{F,1} = [-55.]     # Reset voltage (mV)
    state::Array{F,1} = [-55.]     # Membrane potential (mV)
end

function update!(neuron::LIF, input_update, dt, t) 
    retval = 0
    # If an impulse came in, add it
    neuron.state .+= input_update

    # Euler method update
    neuron.state .+= (dt/neuron.τ) * (-neuron.state[1] + neuron.R*neuron.I)

    # Check for thresholding
    if neuron.state[1] >= neuron.θ
        neuron.state .= neuron.v0
        retval = 1 # Binary output
    end

    return retval
end

function reset!(neuron::LIF)
    neuron.state .= neuron.v0
end


# Izhikevich Neuron
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
    neuron.state .+= input_update

    # Euler method update
    neuron.state .+= [
        dt*(0.05 * neuron.state[1]^2 + 5*neuron.state[1] + 140 - neuron.state[2] + neuron.I),
        dt*(neuron.a)*(neuron.b*neuron.state[1]-neuron.state[2])
        ]

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