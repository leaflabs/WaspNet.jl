# Abstract type for all nnsim.jl types.
abstract type NetworkElement end

# Abstract type for neurons, meant to be implemented by user.
abstract type AbstractNeuron <: NetworkElement end

# Abstract type for layers
abstract type AbstractLayer <: NetworkElement end

# Abstract type for networks
abstract type AbstractNetwork <: NetworkElement end
