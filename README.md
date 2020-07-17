# WaspNet.jl
`WaspNet.jl` is a Julia package for fixed-time-step simulations of primarily spiking neural networks (SNNs). 

`WaspNet.jl` is intended for exploring SNN or hybrid architectures and experimenting with new neuron models. `Network` and `Layer` abstractions are provided, and an `AbstractNeuron` type is defined for users to construct their own neuron models. Utilizing multiple dispatch ensures that new neuron types slot easily into the existing framework.

# Naming

`WaspNet.jl` is named after [Megaphragma mymaripenne](https://en.wikipedia.org/wiki/Megaphragma_mymaripenne), the third-smallest known extant insect and a microscopically sized wasp. *M. mymaripenne* are known for possessing a very low number of neurons (roughtly 7,000) yet they are still capabable of exhibiting higher order behavior.

# Acknowledgements

This material is based upon work supported by the Defence Advanced Research Projects Agency (DARPA) under Agreement No. HR00111990036; DARPA-PA-18-02-03 Microscale Bio-mimetic Robust Artificial Intelligence Networks (AIE; μBRAIN)