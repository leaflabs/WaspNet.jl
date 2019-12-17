# Layer
mutable struct Layer{L<:AbstractNeuron}<:AbstractNetwork
    W::Matrix{AbstractFloat}
    N_neurons::Int
    N_inputs::Int
    neurons::Array{L,1}
end



