# Abstract type for all nnsim.jl types.
abstract type WaspnetElement end

"""
	abstract type AbstractNeuron <: WaspNetElement

Contains the relevant parameters and states for simulating a neuronal model. The `state` of the neuron
must be stored in a field named `state`, but otherwise there are no restrictions. 

For help, see the `nnsim.Izh` docstring and example.
"""	
abstract type AbstractNeuron <: WaspnetElement end

# Abstract type for layers
abstract type AbstractLayer <: WaspnetElement end

# Abstract type for networks
abstract type AbstractNetwork <: WaspnetElement end

# Abstract type for simulations
abstract type AbstractSimulation <: WaspnetElement end