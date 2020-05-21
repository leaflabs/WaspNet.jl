@testset "Matrix Layer" begin
        lif1 = nnsim.LIF()
        lif2 = nnsim.LIF()
        W = zeros(2,3)
        W[1,1] = 1.
        W[1,2] = 2.
        W[2,2] = 1.

        L = Layer([lif1, lif2], W)

        @test isa(L.W, Matrix)
        @test isempty(L.conns)
        @test all(L.input .== zeros(2))
        @test all(L.output .== zeros(2))

        update!(L, [[0., 0., 0.]], 0.001, 0.)
        @test lif1.state[1] == lif2.state[1]    # Both neurons are initialized to same state
        s10 = lif1.state[1]
        s20 = lif2.state[1]

        update!(L, [[2., 3., 0.]], 0., 0.)
        @test begin                             # Inputs are routed correctly
            ( (lif2.state[1] == s20 + 3.) &&
                (lif1.state[1] == s10 + 8.) )
        end

        update!(L, [[0., 0., 9.]], 0., 0.)
        @test begin                             # Inputs are routed correctly
            ( (lif1.state[1] == s10 + 8.) &&
                (lif2.state[1] == s20 + 3.) )
        end

        @test begin                             # All neurons reset
            reset!(L)
            ( (lif1.state[1] == -55.) &&
                (lif2.state[1] == -55.) )
        end
    end

    @testset "BlockArray Layer" begin
        N0 = 2
        N1 = 3
        N2 = 4
        neurons = [nnsim.LIF() for _ in 1:N1]
        W = BlockArray(zeros(N1, N0+N1+N2), [N1], [N0, N1, N2])
        W[Block(1,1)] .= randn(N1,N0)
        W[Block(1,3)] .= randn(N1,N2)
        conns = [0,2]

        layer = Layer(neurons, W, conns)
        v0 = layer.neurons[1].state[1]

        @test all(size(layer.W) .== (N1, N0+N1+N2))
        @test all(layer.conns .== [0,2])

        # Testing that update works at all
        @test begin
            update!(layer, [zeros(N0), zeros(N1), zeros(N2)], 0., 0.)
            all([all(n.state .== [v0]) for n in layer.neurons])
        end

        # Testing that we don't let input from the middle layer affect the state
        @test begin
            update!(layer, [zeros(N0), ones(N1), zeros(N2)], 0., 0.)
            all([all(n.state .== [v0]) for n in layer.neurons])
        end

        # Testing that input from the first + third layers affect the state
        @test begin
            update!(layer, [ones(N0), zeros(N1), zeros(N2)], 0., 0.)
            all([all(n.state .!= [v0]) for n in layer.neurons])
        end
        reset!(layer)
        @test begin
            update!(layer, [zeros(N0), zeros(N1), ones(N2)], 0., 0.)
            all([all(n.state .!= [v0]) for n in layer.neurons])
        end
    end
