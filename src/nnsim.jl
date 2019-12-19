module nnsim

using Parameters

include("types.jl")
include("neurons.jl")
include("layer.jl")
include("network.jl")

include("utils.jl")

export NetworkElement, AbstractNetwork, AbstractNeuron, AbstractLayer
export Layer, Network
export update!, reset!

export Batch_Layer_Construction

end # module
