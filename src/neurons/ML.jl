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