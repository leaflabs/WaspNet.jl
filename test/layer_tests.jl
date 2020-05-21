@testset "Matrix Layer" begin
    n1 = nnsim.identity()
    n2 = nnsim.identity()
    W = zeros(2,3)
    W[1,1] = 1.
    W[1,2] = 2.
    W[2,2] = 1.
    W[2,3] = 2.

    L = Layer([n1, n2], W)

    # Checking initializations are correct
    @test isa(L.W, Matrix)
    @test isempty(L.conns)
    @test all(L.input .== zeros(2)) 
    @test all(L.output .== zeros(2))
    @test n1.state[1] == n2.state[1]    

    # Inputs are routed correctly
    @test begin                             
        update!(L, [[2., 3., 4.]], 0., 0.)
        all(nnsim.get_neuron_states(L) .== [8., 11.])
    end

    # All neurons reset
    @test begin                             
        reset!(L)
        all(nnsim.get_neuron_states(L) .== [0., 0.])
    end
end

@testset "BlockArray Layer" begin
    N0 = 2
    N1 = 3
    N2 = 4
    neurons = [nnsim.identity() for _ in 1:N1]
    W = BlockArray(zeros(N1, N0+N1+N2), [N1], [N0, N1, N2])
    W[Block(1,1)] .= randn(N1, N0)
    W[Block(1,3)] .= randn(N1, N2)
    conns = [0,2]

    layer = Layer(neurons, W, conns)

    @test all(size(layer.W) .== (N1, N0+N1+N2))
    @test all(layer.conns .== [0,2])

    # Testing that update works at all
    @test begin
        update!(layer, [zeros(N0), zeros(N1), zeros(N2)], 0., 0.)
        all([all(n.state .== [0]) for n in layer.neurons])
    end

    # Testing that input from the first + third layers affect the state
    # Have to test approximate equality with ≈, presumbly from FP errors
    @test begin
        update!(layer, [ones(N0), zeros(N1), zeros(N2)], 0., 0.)
        all(nnsim.get_neuron_states(layer) .≈ sum.(eachrow(W[Block(1,1)])))
    end
    reset!(layer)
    @test begin
        update!(layer, [zeros(N0), zeros(N1), ones(N2)], 0., 0.)
        all(nnsim.get_neuron_states(layer) .≈ sum.(eachrow(W[Block(1,3)])))
    end
    @test begin
        update!(layer, [ones(N0), zeros(N1), ones(N2)], 0., 0.) 
        all(
            nnsim.get_neuron_states(layer) .≈ 
                sum.(eachrow(W[Block(1,3)])) .+ sum.(eachrow(W[Block(1,1)]))
            )
    end

    # Testing that we don't let input from the middle layer affect the state
    @test begin
        update!(layer, [zeros(N0), ones(N1), zeros(N2)], 0., 0.)
        all(nnsim.get_neuron_states(layer) .== [0; 0; 0])
    end
end

