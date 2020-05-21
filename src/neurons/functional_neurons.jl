##################################################################
# ReLU unit
##################################################################
@with_kw struct ReLU{F}<:AbstractNeuron
    v0::F = 0.     # Reset voltage (mV)
    state::Array{F,1} = [0.]     # Membrane potential (mV)
end

# ReLU update function.  ReLU is memoryless, so dt and t are not used.
# They are left as parameters for the sake of interoperability with existing function calls
function update!(neuron::ReLU, input_update, dt, t)
    # take maximum of 0 and the input, as per the definition
    neuron.state[1] = max(0, input_update)
    return neuron.state[1]
end

function reset!(neuron::ReLU)
    neuron.state .= [neuron.v0]
end

##################################################################
# tanh unit
##################################################################
@with_kw struct tanh{F}<:AbstractNeuron
    v0::F = 0.     # Reset voltage (mV)
    state::Array{F,1} = [0.]     # Membrane potential (mV)
end

# tanh update function.  tanh is memoryless, so dt and t are not used.
# They are left as parameters for the sake of interoperability with existing function calls
function update!(neuron::tanh, input_update, dt, t)
    # specify Base for the sake of avoiding namespace collisions
    neuron.state[1] = Base.tanh(input_update)
    return neuron.state[1]
end

function reset!(neuron::tanh)
    neuron.state .= [neuron.v0]
end

##################################################################
# sigmoid unit
##################################################################
@with_kw struct sigmoid{F}<:AbstractNeuron
    v0::F = 0.     # Reset voltage (mV)
    state::Array{F,1} = [0.]     # Membrane potential (mV)
end

# sigmoid update function.  sigmoid is memoryless, so dt and t are not used.
# They are left as parameters for the sake of interoperability with existing function calls
function update!(neuron::sigmoid, input_update, dt, t)
    neuron.state[1] = 1. / (1. + exp(-input_update))
    return neuron.state[1]
end

function reset!(neuron::sigmoid)
    neuron.state .= [neuron.v0]
end

##################################################################
# Identity unit (testing)
##################################################################
@with_kw struct identity{F}<:AbstractNeuron
    state::Array{F,1} = [0.]    # Not *really* a stateful neuron, just passes its input forward
end

# Passes through its input value as an output
function update!(neuron::identity, input_update, dt, t)
    neuron.state[1] = input_update
    return neuron.state[1]
end

# Only here because it's necessary for WaspNet
function reset!(neuron::identity)
    neuron.state .= [0.]
end