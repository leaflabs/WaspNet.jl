# For overloading purposes
function update! end

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
###
# Hodgkin-Huxley Neuron, using Euler method to ensure uniform time step
# TODO: Getting this one to work is slightly more complicated because the differential equations governing the time evolution of n, m, and h, the gating variables, involve functions of the voltage that I haven't found approximate expressions for
@with_kw struct HH{F}<:AbstractNeuron
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

alpha_n(v) = 0.01*(10-v)/(exp((10-v)/10)-1)
alpha_m(v) = 0.1*(25-v)/(exp((25-v)/10)-1)
alpha_h(v) = 0.07*exp(-v/20)

function update!(neuron::HH, input_update, dt, t)
    retval = 0
    # If an impulse came in, add it
    neuron.state[1] += input_update

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

function reset!(neuron::HH)
    neuron.state .= [neuron.v0, neuron.u0]
end

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

# Morris-Lecar neuron model, using Euler method to ensure uniform time step
@with_kw struct ML{F}<:AbstractNeuron
   v1::F = -1.2      # a-d are model parameters
   v2::F = 18.
   v3::F = 2.
   v4::F = 30.
   gca::F = 4.4
   gk::F = 8.0
   gl::F = 2.0
   vk::F = -84.
   vl::F = -60.
   vca::F = 120.
   phi::F = 0.04
   I::F = 25.       # Background current injection (mA)
   θ::F = 30.       # Threshold potential (mV)

   v0::F = -65.     # Reset voltage (mV)
   u0::F = 0.       # Reset state variable
   state::Array{F,1} = [-65., 0.]      # Membrane potential (mV) and state variable
end
###
# These functions describe the gating variables and time constant as a function of voltage.  Externally defining them greatly simplifies the description of the time evolution of the state parameters.  Not sure if this is the most elegant way to define them but there you have it
m(neuron::ML, v) = 0.5*(1 + tanh((v-neuron.v1)/neuron.v2))
n(neuron::ML, v) = 0.5*(1 + tanh((v-neuron.v3)/neuron.v4))
tau(neuron::ML, v) = (cosh((v-neuron.v3)/(2*neuron.v4)))^-1

function update!(neuron::ML, input_update, dt, t)
    retval = 0
    # If an impulse came in, add it
    neuron.state[1] += input_update

    # Euler method update
    neuron.state .+= [
        dt*(-neuron.gca*m(neuron, neuron.state[1])*(neuron.state[1]-neuron.vca) - neuron.gk*neuron.state[2]*(neuron.state[1] - neuron.vk) - neuron.gl*(neuron.state[1]-neuron.vl) + neuron.I),
        dt*(neuron.phi*(n(neuron,neuron.state[1])-neuron.state[2])/tau(neuron,neuron.state[1]))
        ]

    # Check for thresholding
    # if neuron.state[1] >= neuron.θ
    #     neuron.state .= [ neuron.v0, neuron.state[2] + neuron.d]
    #     retval = 1
    # end

    return retval
end

function reset!(neuron::ML)
    neuron.state .= [neuron.v0, neuron.u0]
end

# ReLU unit
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

# tanh unit
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

# sigmoid unit
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
