module WaspNet

# Used for concise constructors with @with_kw macro
using BlockArrays
using Distributions
using LinearAlgebra
using Parameters
using Random
using RecursiveArrayTools

using OrdinaryDiffEq

include("types.jl")

include("defs.jl")
include("layer.jl")
include("input_layer.jl")
include("network.jl")
include("neurons.jl")
include("simulate.jl")
include("utils.jl")

export WaspnetElement, AbstractNetwork, AbstractNeuron, AbstractLayer
export Layer, Network, SimulationResult
export update!, update, simulate!
export aff_element!, event

export construct_problem

export batch_layer_construction, network_constructor, layer_constructor, feed_forward_network

export getPoissonST

end # module
