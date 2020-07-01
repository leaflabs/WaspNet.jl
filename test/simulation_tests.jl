@testset "Simulation" begin
        #######################################################
        # Test simulate! with Matrix inputs
        #######################################################
        # reset!(net_hom)
        # outputs, states = simulate!(net_hom, zeros(N_in, 1000), 0.001, track_flag = true)
        # @test begin                         # Check neuron output matrix size
        #     all(size(outputs) .== [2*N, 1001])
        # end

        # # Check neuron output matrix size
        # @test begin                         
        #     all(size(states) .== [2*N, 1001])
        # end

        # ######################################################
        # # Test simulate! with a function input
        # ######################################################
        # input_fun(t) = zeros(Float64, N_in)
        # outputs, states = simulate!(net_hom, input_fun, 0.001, 1., track_flag = true)
        # # Check neuron output matrix size
        # @test begin                         
        #     all(size(outputs) .== [2*N, 1001])
        # end

        # # Check neuron output matrix size
        # @test begin                         
        #     all(size(states) .== [2*N, 1001])
        # end

        # # Reset the full network
        # @test begin                         
        #     reset!(net_hom)
        #     all(WaspNet.get_neuron_states(net_hom) .== 0.)
end