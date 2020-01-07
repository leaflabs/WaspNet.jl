# nnsim
`nnsim` is a Julia package for fixed-time-step simulations of both artificial and spiking neural networks (ANNs and SNNs). 

This package is meant primarily for exploring SNN or hybrid architectures and experimenting with new neuron models. `Network` and `Layer` abstractions are provided, and an `AbstractNeuron` type is defined for users to construct their own neuron models. Utilizing multiple dispatch ensures that new neuron types slot easily into the existing framework.

## Structure of `nnsim`

`nnsim` provides concise but robust abstractions which allow construction of spiking neural networks. These abstractions separate the behavior of neural networks into three distinct levels, in increasing order: the neuron, the layer, and the network. An example implementation of an Izhikevich neuron is provided in [this notebook](/notebooks/nnsim_tour.ipynb) (requires IJulia).

### Neurons

Neurons (subtypes of `AbstractNeuron`) control the behavior of the individual neuronal units comprising the network. A neuron processes an input signal and then outputs some signal. Most neurons will possess a concept of _time_, particularly spiking neural networks which are governed by differential equations. Two example spiking neuron models are implemented in [`neurons.jl`](/src/neurons.jl): the [Leaky Integrate & Fire](https://neuronaldynamics.epfl.ch/online/Ch1.S3.html) and the [Izhikevich](https://www.izhikevich.org/publications/spikes.htm) models. 

A neuron model comprises a struct which is a subtype of `AbstractNeuron` and an implementation of model-specific `update!` and `reset!` methods. The struct should hold any parameters relevant to the model as well as a `state` variable which completely describes the state of the neuron.

`update!` should evolve a specified neuron subject to some `input` a total duration of `dt` units of time at a time `t`. For efficiency, the supplied examples use an Euler method update for the neuronal states. `reset!` resets the state of the neuron to its initial values to begin a new simulation run.

### Layers

A `Layer` is a collection of neurons and their associated input weight matrix, `W`. To construct a new `Layer`, an array whose elements are some subtype of `AbstractNeuron` must be supplied along with the weight matrix. 

When calling `update!` on a `Layer`, the appropriate neuron inputs are computed and the neurons are evolved in time. Inputs to the `Layer` are multiplied by `W` to produce the weighted input to each neuron in the layer. Each neuron is then evolved in accordance with its `update!` function, and the neuron outputs are returned to be fed into the following `Layer`.

In the future, training or learning are intended take place at the layer level, with e.g. backpropagated gradients or STDP learning rules coded to operate as a function of neuron states.

### Networks

A `Network` collects `Layer`s and orders them and tracks the states of the constituent neurons over the course of a simulation. `Network`s are constructed by specifying the `Layer`s which make up the `Network` in order from first to final.

`Network`s primarily function as a structure for tracking the state of the simulation. While a `Network` can be updated similarly to its subcomponenents, a network is better used for doing multiple-time-step simulations. These simulations are executed through a `simulate!` function which simulates the network for a specified duration of time.

## Future Work

* Implement a training/learning function. Maybe this needs to be a new type we provide to `Layer`s or `Network`s as well
* More neuron examples as they become relevant. LIF and Izhikevich are good starting points, but probably best to write them ourselves
* Implement recurrent networks, this seems nontrivial but I haven't thought enough to figure out why/how it will be hard
* Show that we can do ANNs in this as well (it just means time is irrelevant, at least until we try some RNN work)
* More utilitarian functions to construct networks. Standard layer weight initializers, network-level constructors, etc


