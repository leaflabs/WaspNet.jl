# Layer, a collection of neurons of the same type being driven by some input vector
struct Layer{L<:AbstractNeuron,F<:Real}<:AbstractLayer
    neurons::Array{L,1}
    output::Array{F,1}
    conns::Array{Int,1}
    W # TODO: Make type union with sparse matrices, if sparse matrices end up efficient
    N_neurons

    input::Array{F,1}
    block_array::Bool

    function Layer(neurons::Array{L,1}, output::Array{F,1}, conns, W, N_neurons) where {L <: AbstractNeuron, F <: Real}
        input = zeros(length(neurons))
        if isa(W, AbstractBlockArray)
            block_array = true
        elseif isa(W, Matrix)
            block_array = false 
        else
            error("Expected Layer Weights to be BlockArray or Matrix, got $(typeof(W))")
        end

        return new{L, F}(neurons, output, conns, W, N_neurons, input, block_array) 
    end

    # Default constructor, nonparametric
    function Layer(neurons::Array{L,1}, output::Array{F,1}, conns, W, N_neurons, input, block_array
        ) where {L <: AbstractNeuron, F <: Real}
        return Layer{L,F}(neurons, output, conns, W, N_neurons, input, block_array)
    end

    # Default constructor, parametric
    function Layer{L,F}(
        neurons::Array{L,1}, output::Array{F,1}, conns, W, N_neurons, input, block_array
        ) where {L <: AbstractNeuron, F <: Real}
        return new{L, F}(neurons, output, conns, W, N_neurons, input, block_array)
    end
end

# Cover the case where conns isn't specified (defaults to feed-forward)
function Layer(neurons, output, W::Array{<:Any,2}, N_neurons)
    return Layer(neurons, output, Array{Int}(undef, 0), BlockArray(W), N_neurons)
end

# function Layer(neurons, output, conns, W::Array{<:Real,2}, N_neurons)
#     return Layer(neurons, output, conns, BlockArray(W), N_neurons)
# end

# Evolve all of the neurons in the layer a duration `dt` starting at the time `t`
#   subject to an input from the previous layer `input`.
function update!(l::Layer, input, dt, t)
    nonzero_inputs = [!(all(input[i+1] .== 0)) for i in l.conns]
    l.input .= 0 # reset the input vector to the layer

    if any(nonzero_inputs)
        if l.block_array # layer uses block-array weight storage
            for conn in l.conns
                if nonzero_inputs[conn+1]
                    l.input .+= l.W[Block(1, conn+1)]*input[conn+1]
                end
            end
        else
            l.input .= l.W*input[l.conns[1]+1]
        end
    end

    l.output .= update!.(l.neurons, l.input, dt, t)
    # if none([all(l_out .== 0) for l_out in input]))
    #     trans_inp = zeros(l.N_neurons) # pre-allocate all zeros array
    #     for i in l.conns # loop over non-zero incoming connections and summate signals
    #         trans_inp += (l.W[Block(1,i+1)]*input[i+1])
    #     end
    #     l.output .= update!.(l.neurons, trans_inp, dt, t) # TODO: change input radically
    # else
    #     l.output .= update!.(l.neurons, 0, dt, t)
    # end

    return l.output
end

function reset!(l::AbstractLayer)
    reset!.(l.neurons)
end

# Get the state of each neuron in this layer
function get_neuron_states(l::AbstractLayer)
    return vcat([n.state for n in l.neurons]...)
end

# Get the output of each neuron at the current time in the layer
function get_neuron_outputs(l::AbstractLayer)
    return l.output
end
