mutable struct InputMatrixLayer{M<:AbstractArray{<:Number, 2}, T<:AbstractArray{<:Number,1}}<:AbstractLayer
    data::M
    times::T
    idx::Int
    N_neurons::Int
end

function InputMatrixLayer(data, times)
    @assert (length(times) == size(data)[2]) "Time duration and length of input data must match"
    return InputMatrixLayer(data, times, 1, size(data)[1])
end


function update!(l::InputMatrixLayer, du, u, t)
    return nothing
end

function event(l::InputMatrixLayer, u, t)
    return false
end

function aff_element!(l::InputMatrixLayer, u, input, t)
    return nothing
end

num_neurons(l::InputMatrixLayer) = l.N_neurons

get_output(l::InputMatrixLayer) = @view l.data[:,l.idx]