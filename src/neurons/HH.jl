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
    dt *= 100
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