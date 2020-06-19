include("neurons/LIF.jl")
include("neurons/Izh.jl")
include("neurons/HH.jl")
include("neurons/FHN.jl")
include("neurons/ML.jl")
include("neurons/functional_neurons.jl")

function get_neuron_outputs(n::AbstractNeuron)
    return n.output
end

function get_neuron_states(n::AbstractNeuron)
    return n.state
end

function get_neuron_count(n::AbstractNeuron)
    return 1
end