module WaspNet

# Used for concise constructors with @with_kw macro
using Parameters
using BlockArrays
using Distributions
using Random
using LinearAlgebra

include("types.jl")
include("defs.jl")
include("neurons.jl")
include("layer.jl")
include("network.jl")
include("simulate.jl")
include("utils.jl")

export WaspnetElement, AbstractNetwork, AbstractNeuron, AbstractLayer
export Layer, Network, SimulationResult
export update!, reset!, simulate!

export batch_layer_construction, network_constructor, layer_constructor, feed_forward_network

end # module
