abstract type WaspnetElement end

"""
	abstract type AbstractNeuron <: WaspNetElement

Contains the relevant parameters and states for simulating a specific neuronal model. 

The `state` of the neuron must be stored in a field named `state` and the output stored in `output`, but otherwise there are no restrictions. These requirements may be relaxed by writing new `nnsim.get_neuron_states` and `nnsim.get_neuron_outputs` methods.

For help, see the `nnsim.Izh` docstring and example.
"""	
abstract type AbstractNeuron <: WaspnetElement end

abstract type AbstractLayer <: WaspnetElement end

abstract type AbstractNetwork <: WaspnetElement end

abstract type AbstractSimulation <: WaspnetElement end