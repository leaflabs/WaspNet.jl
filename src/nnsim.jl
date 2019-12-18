module nnsim

using SparseArrays
using Parameters

include("types.jl")
include("neurons.jl")
include("layer.jl")
include("network.jl")

export NetworkElement, AbstractNetwork, AbstractNeuron, AbstractLayer
export Layer, Network
export update!, reset!

end # module
