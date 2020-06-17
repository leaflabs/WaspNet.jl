@testset "Pruning" begin 

    @testset "delete_entries" begin 
        # Block Array tests

        ba = BlockArray(reshape(collect(1:25), (5,5)), [2, 2, 1], [2, 3])
        new_ba = nnsim.delete_entries(ba, [3], axis=1)
        @test begin
            all(size(new_ba) .== (4,5))
        end
        @test begin
            all(Array(new_ba) .== reshape(collect(1:25), (5,5))[[1,2,4,5],:])
        end

        new_ba = nnsim.delete_entries(new_ba, [3], axis=2)
        @test begin
            all(Array(new_ba) .== reshape(collect(1:25), (5,5))[[1,2,4,5],[1,2,4,5]])
        end
        @test begin
            all(blocklengths(axes(new_ba,1)) .== [2,1,1]) && all(blocklengths(axes(new_ba,2)) .== [2,2])
        end

        # Matrix tests
        W = reshape(collect(1:25), (5,5))
        new_W = nnsim.delete_entries(W, [3], axis=1)
        @test begin
            all(size(new_W) .== (4,5))
        end
        @test begin
            all(Array(new_W) .== reshape(collect(1:25), (5,5))[[1,2,4,5],:])
        end

        new_W = nnsim.delete_entries(new_W, [3], axis=2)
        @test begin
            all(Array(new_W) .== reshape(collect(1:25), (5,5))[[1,2,4,5],[1,2,4,5]])
        end
    end

    @testset "Layer Pruning" begin
        # Matrix W
        neurons = [nnsim.Izh() for _ in 1:5]
        W = Array{Float64,2}(reshape(collect(1:25), (5,5)))
        conns = [3]
        L = Layer(neurons, W, conns)

        l_idx = 1
        layers = [1,2,3,4]
        prune_neurons = [[2,4], [1,2], [1,3,5], [7, 18]]

        L2 = nnsim.prune(L, l_idx, layers, prune_neurons)
        @test begin
            length(L2.neurons) == 3
        end
        @test begin
            all(L2.W .== W[[1,3,5],[2,4]])
        end

        # BlockArray W
        neurons = [nnsim.Izh() for _ in 1:5]
        W = BlockArray(Array{Float64,2}(reshape(collect(1:25), (5,5))), [5], [2, 2, 1])
        conns = [2,3,4]
        L = Layer(neurons, W, conns)

        l_idx = 1
        layers = [1,2,3]
        prune_neurons = [[2,4], [1], [2]]
        L2 = nnsim.prune(L, l_idx, layers, prune_neurons)
        @test begin
            length(L2.neurons) == 3    
        end
        @test begin
            all(size(L2.W) .== (3,3))
        end
        @test begin
            all(Array(L2.W) .== reshape(collect(1:25), (5,5))[[1,3,5],[2,3,5]] )
        end
        @test begin
            all(blocklengths(axes(L2.W,1)).== [3]) && all(blocklengths(axes(L2.W,2)).== [1,1,1])
        end

    end

    @testset "Network Pruning" begin
        N1 = [nnsim.LIF() for _ in 1:5]
        W10 = reshape(collect(1:20)*1., (5,4))
        W11 = reshape(collect(1:25)*1., (5,5))
        W12 = reshape(collect(1:30)*1., (5,6))
        W1 = BlockArray(hcat(W10, W11, W12), [5],[4,5,6])
        L1 = Layer(N1, W1, [0, 1, 2])

        N2 = [nnsim.LIF() for _ in 1:6]
        W20 = reshape(collect(1:24)*1., (6,4))
        W21 = reshape(collect(1:30)*1., (6,5))
        W22 = reshape(collect(1:36)*1., (6,6))
        W2 = BlockArray(hcat(W20, W21, W22), [6], [4,5,6])
        L2 = Layer(N2, W2, [0,1,2])

        net = Network([L1, L2], 4)
        prune_layers = [1,2]
        prune_neurons = [[3], [2, 4]]
        pruned = nnsim.prune(net, prune_layers, prune_neurons)
        
        @test begin
            all(size(pruned.layers[1].W) .== (4,12)) && all(size(pruned.layers[2].W) .== (4,12))
        end
        @test begin 
            all(pruned.layers[1].W .== W1[[1,2,4,5],[collect(1:6)..., 8,9,10,12,14,15]]) &&
                all(pruned.layers[2].W .== W2[[1,3,5,6],[collect(1:6)..., 8,9,10,12,14,15]])
        end
    end

end