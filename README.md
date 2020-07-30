# WaspNet.jl
`WaspNet.jl` is a Julia package for fixed-time-step simulations of primarily spiking neural networks (SNNs). 

`WaspNet.jl` is intended for exploring SNN or hybrid architectures and experimenting with new neuron models. `Network` and `Layer` abstractions are provided, and an `AbstractNeuron` type is defined for users to construct their own neuron models. Utilizing multiple dispatch ensures that new neuron types slot easily into the existing framework.

# Installation

To install `WaspNet.jl`, from the REPL go into `Pkg` mode by pressing `]` and then `add WaspNet`.

Alternatively, run `using Pkg; Pkg.add("WaspNet")`

# Introduction

An example script is provided in the documentation which can be built with `docs/make.jl`. This script showcases the development of a new neuron type, isntantiation and simulation of that neuron, and then building and simulating a network of these neurons. 

# Naming

`WaspNet.jl` is named after [Megaphragma mymaripenne](https://en.wikipedia.org/wiki/Megaphragma_mymaripenne), the third-smallest known extant insect and a microscopically sized wasp. *M. mymaripenne* are known for possessing a very low number of neurons (roughtly 7,000) yet they are still capabable of exhibiting higher order behavior.

# Acknowledgements

This material is based upon work supported by the Defence Advanced Research Projects Agency (DARPA) under Agreement No. HR00111990036; DARPA-PA-18-02-03 Microscale Bio-mimetic Robust Artificial Intelligence Networks (AIE; μBRAIN)
