# Abstract type for all nnsim.jl types.
abstract type NetworkElement end

"""
	abstract type AbstractNeuron <: NetworkElement

Contains the relevant parameters and states for simulating a neuronal model. The `state` of the neuron
must be stored in a field named `state`, but otherwise there are no restrictions. 

For help, see the `nnsim.Izh` docstring and example.
"""	
abstract type AbstractNeuron <: NetworkElement end

# Abstract type for layers
abstract type AbstractLayer <: NetworkElement end

# Abstract type for networks
abstract type AbstractNetwork <: NetworkElement end
