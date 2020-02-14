module nnsim

# Used for concise constructors with @with_kw macro
using Parameters
using BlockArrays
using Distributions
using Random

include("types.jl")
include("neurons.jl")
include("layer.jl")
include("network.jl")
include("utils.jl")

export NetworkElement, AbstractNetwork, AbstractNeuron, AbstractLayer
export Layer, Network
export update!, reset!, simulate!

export batch_layer_construction, network_constructor, layer_constructor, feed_forward_network

end # module
