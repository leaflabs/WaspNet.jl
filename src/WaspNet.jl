module WaspNet

# Used for concise constructors with @with_kw macro
using BlockArrays
using Distributions
using LinearAlgebra
using Parameters
using Random

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
export update

export batch_layer_construction, network_constructor, layer_constructor, feed_forward_network

end # module
