module nnsim

using SparseArrays
using Parameters

include("types.jl")
include("neurons.jl")
include("layer.jl")

export AbstractNetwork, AbstractNeuron, Layer
export update!

end # module
