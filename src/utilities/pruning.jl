# Prunes a given Layer l which is located at index l_idx in the layers and neurons arrays
# length(layers) = length(neurons), each entry in the latter corresponds to neurons from 
# the respective layer in the former.
function prune(l::Layer{L,N,A,M}, l_idx::Int, layers::Array{Int,1}, neurons::Array{Array{Int,1},1}
    ) where {L,N,A, M<:Matrix}
    new_neurons = l.neurons
    new_W = l.W
    new_conns = l.conns

    if l_idx in layers
        n_idx = findfirst(x -> x == l_idx, layers)
        new_neurons = l.neurons[setdiff(1:length(new_neurons), neurons[n_idx])]
        new_W = delete_entries(new_W, neurons[n_idx], axis = 1)
    end

    if (isempty(l.conns) || l.conns[1] == 0)
        # Pass
    elseif l.conns[1] in layers
        l_idx_extern = findfirst(x -> x == l.conns[1], layers)
        new_W = delete_entries(new_W, neurons[l_idx_extern], axis = 2)
    end
    return Layer(new_neurons, new_W, new_conns)
end

function prune(l::Layer{L,N,A,M}, l_idx::Int, layers::Array{Int,1}, neurons::Array{Array{Int,1},1}
    ) where {L,N,A, M<:BlockArray}
    new_neurons = l.neurons
    new_W = l.W
    new_conns = l.conns

    if l_idx in layers
        n_idx = findfirst(x -> x == l_idx, layers)
        new_neurons = l.neurons[setdiff(1:length(new_neurons), neurons[n_idx])]
        new_W = delete_entries(new_W, neurons[n_idx], axis = 1)
    end

    if isempty(l.conns)
    else 
        n_deleted = 0
        cols_to_delete = Array{Int,1}()
        for (i,c) in enumerate(l.conns)
            if c in layers
                l_idx_extern = findfirst(x -> x == c, layers)
                neurons_in_block = neurons[l_idx_extern]
                neurons_in_BA = neurons_in_block .+ blockfirsts(axes(l.W,2))[i] .- 1
                push!(cols_to_delete, neurons_in_BA...)
            end
        end
        new_W = delete_entries(new_W, cols_to_delete, axis = 2)
    end
    return Layer(new_neurons, new_W, new_conns)
end

function prune(net::Network, layers::Array{Int,1}, neurons::Array{Array{Int, 1}, 1})
    old_layers = net.layers
    new_layers = Array{AbstractLayer, 1}()
    for (l_idx,l) in enumerate(old_layers)
        push!(new_layers, prune(l, l_idx, layers, neurons))
    end
    new_net = Network(new_layers, net.N_in)
    return new_net
end

# Removes the entries along the specified axis from the given BlockArray, maintaining the
#   sizes of all unaffected blocks and contracting the sizes of the affected blocks.
# e.g. removing the 3rd row from a 5x5 matrix
function delete_entries(W::BlockArray, entries::Array{Int,1}; axis::Int = 1)
    # Delete the rows from the BlockArray and store this in a new array
    # First we get all the axes of the BlockArray, then remove the proper entries by index
    remaining_entries = Array{Any}(collect(axes(W) ))
    remaining_entries[axis] = setdiff(remaining_entries[axis], entries) 
    new_array = Array(W[remaining_entries...])

    # Now we re-calculate the block sizes
    n_dims = length(size(W)) # Number of dimension
    n_items = size(W)[axis] # Number of items along the specified axis

    block_lengths = collect(blocklengths.(axes(W))) # Lengths of blocks along each axis

    item_sizes = block_lengths[axis] # block sizes along the axis of interest
    item_idxs = [cumsum(item_sizes)...] # get the corresponding block indices from the block sizes

    new_item_sizes = Vector{Int}()
    idx_0 = 1
    for idx_f in item_idxs
        # Count how many of the remaining entries are in the current idx range for this block
        n_in_block = count([n in idx_0:idx_f for n in remaining_entries[axis]])
        push!(new_item_sizes, n_in_block) 
        idx_0 = idx_f + 1
    end
    block_lengths[axis] = new_item_sizes

    return BlockArray(new_array, block_lengths...)
end

function delete_entries(W::Matrix, entries::Array{Int, 1}; axis::Int = 1)
    remaining_entries = Array{Any}(collect(axes(W) ))
    remaining_entries[axis] = setdiff(remaining_entries[axis], entries) 
    return W[remaining_entries...]

end

