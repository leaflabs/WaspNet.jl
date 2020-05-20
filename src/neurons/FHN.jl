# FitzHugh-Nagumo Neuron, using Euler method to ensure uniform time step
@with_kw struct FHN{F}<:AbstractNeuron
   a::F = 0.8      # a-d are model parameters
   b::F = 0.7
   tau::F = 12.5    # Time Constant
   I::F = 25.       # Background current injection (mA)
   θ::F = 30.       # Threshold potential (mV)

   v0::F = -65.     # Reset voltage (mV)
   w0::F = 0.       # Reset state variable
   state::Array{F,1} = [-65., 0.]      # Membrane potential (mV) and state variable
end

function update!(neuron::FHN, input_update, dt, t)
    retval = 0
    # If an impulse came in, add it
    neuron.state[1] += input_update

    # Euler method update
    neuron.state .+= [
        dt*(neuron.state[1] - (neuron.state[1]^3)/3 - neuron.state[2] + neuron.I),
        dt*(neuron.state[1] + neuron.a - neuron.state[2]*neuron.b)/(neuron.tau)
        ]

    # Commented out for now because I don't think this model requires manual variable resets, the spikes should just be built into the phase space of the model, but I am not sure
    # TODO make sure checking for spikes is unnecessary for FHN model
    # Check for thresholding
    # if neuron.state[1] >= neuron.θ
    #     neuron.state .= [ neuron.v0, neuron.state[2]]
    #     retval = 1
    # end
    retval = 0

    return retval
end

function reset!(neuron::FHN)
    neuron.state .= [neuron.v0, neuron.w0]
end