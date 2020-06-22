"""
    function prune(el::WaspnetElement, layers, neurons[, l_idx])

Given an element `el` along with indices for target `Neuron`s, constructs new `Layer`s and `Network`s with all references to those neurons removed by deleting rows and columns from the proper weight matrices in each `Layer`.

`layers` should be an array of indices relative to the `Network` it is being pruned in; `neurons` should be an array of arrays of indices where the entries in each inner array are indices of neurons within the respective `Layer` from `layers`.

# Arguments
-`el::WaspnetElement`: The element to prune neurons from, either a `Network` or `Layer`
-`layers`: A list of indices for which `Layer`s we're removing neurons from the `Network` where it resides
-`neurons`: A list of lists of neurons to remove in the respective entries from `layers`.
-`l_idx`: If `prune` is called on a `Layer`, `l_idx` denotes the index of the that `Layer` if it were to appear in the list `layers`
"""
function prune(el::WaspnetElement, layers, neurons) end

function prune(l::Layer{L,N,A,M}, layers, neurons, l_idx) where {L,N,A,M}
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

function prune(l::Layer{L,N,A,M}, layers, neurons, l_idx) where {L,N,A, M<:BlockArray}
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

function prune(net::Network, layers, neurons)
    old_layers = net.layers
    new_layers = Array{AbstractLayer, 1}()
    for (l_idx,l) in enumerate(old_layers)
        push!(new_layers, prune(l, layers, neurons, l_idx))
    end
    new_net = Network(new_layers, net.N_in)
    return new_net
end

"""
    function delete_entries(W, entries; axis::Int = 1)

Given an `AbstractArray`, deletes the specified `entries` (e.g. rows or columns) along the given axis; used for pruning weight matrices. 

As an example, `delete_entries(W, [3,4]; axis = 2)` would delete columns 3 and 4 from `W` and return the modified `W`.
"""
function delete_entries(W, entries; axis = 1) end

function delete_entries(W::AbstractArray{T,N}, entries; axis = 1) where {T<:Number,N}
    # Get all the axes, then remove the proper entries by index
    remaining_entries = Array{Any}(collect(axes(W) ))
    remaining_entries[axis] = setdiff(remaining_entries[axis], entries) 
    return W[remaining_entries...]
end

function delete_entries(W::AbstractBlockArray, entries; axis::Int = 1)
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
        #       There's probably an algorithm to do this efficiently, but I don't think that's necessary rn
        n_in_block = count([n in idx_0:idx_f for n in remaining_entries[axis]])
        push!(new_item_sizes, n_in_block) 
        idx_0 = idx_f + 1
    end
    block_lengths[axis] = new_item_sizes

    return BlockArray(new_array, block_lengths...)
end