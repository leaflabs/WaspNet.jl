# Abstract type for all nnsim.jl types.
abstract type AbstractNetwork end

# Abstract type for neurons, meant to be implemented by user.
abstract type AbstractNeuron <: AbstractNetwork end