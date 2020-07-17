@testset "Simulation" begin
    N = 32
    N_in = 16
    neurons1 = [WaspNet.identity() for _ in 1:N]
    neurons2 = [WaspNet.identity() for _ in 1:N]
    L1 = Layer(neurons1, randn(N, N_in))
    L2 = Layer(neurons1, randn(N, N))
    net_hom = Network([L1, L2], N_in)

    #######################################################
    # Test simulate! with Matrix inputs
    #######################################################
    reset!(net_hom)
    results = simulate!(net_hom, zeros(N_in, 1000), 0.001, 1.; track_state = true)
    @test begin                         # Check neuron output matrix size
        all(size(results.outputs) .== [2*N, 1001])
    end

    # Check neuron output matrix size
    @test begin                         
        all(size(results.states) .== [2*N, 1001])
    end

    ######################################################
    # Test simulate! with a function input
    ######################################################
    input_fun(t) = zeros(Float64, N_in)
    results = simulate!(net_hom, input_fun, 0.001, 1.; track_state = true)
    # Check neuron output matrix size
    @test begin                         
        all(size(results.outputs) .== [2*N, 1001])
    end

    # Check neuron output matrix size
    @test begin                         
        all(size(results.states) .== [2*N, 1001])
    end

    # Reset the full network
    @test begin                         
        reset!(net_hom)
        all(WaspNet.get_neuron_states(net_hom) .== 0.)
    end
end