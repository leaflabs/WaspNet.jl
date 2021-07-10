using WaspNet

@testset "Networks" begin
    N = 32
    N_in = 16

    @testset "Homogeneous Networks FF" begin
        neurons1 = [WaspNet.identity() for _ in 1:N]
        W1 = randn(N, N_in)
        neurons2 = [WaspNet.identity() for _ in 1:N]
        W2 = randn(N, N)
        L1 = Layer(neurons1, W1)
        L2 = Layer(neurons2, W2)
        net_hom = Network([L1, L2], N_in)

        # Network should change Layer `conns`
        @test begin
            ( all(net_hom.layers[1].conns .== [0]) &&
                all(net_hom.layers[2].conns .== [1]) )
        end

        state0 = L1.neurons[1].state[1]

        # All neurons initialized correctly
        @test begin
            all(WaspNet.get_neuron_states(net_hom) .== 0.)
        end

        # Layers are changed to have correct `conns` given the feed-forward network
        @test begin
            ( all(net_hom.layers[1].conns .== [0])
                && all(net_hom.layers[2].conns .== [1]) )
        end

        # Neuron Outputs function works
        @test begin
            all(WaspNet.get_neuron_outputs(net_hom) .== 0.)
        end

        # Update works, not evolving system/state unchanged
        @test begin
            update!(net_hom, zeros(Float64, N_in), 0, 0)
            all(WaspNet.get_neuron_states(net_hom) .== 0.)
        end

        # Update passes the correct values to network inputs
        @test begin
            update!(net_hom, ones(Float64, N_in), 0, 0)
            all( WaspNet.get_neuron_outputs(net_hom) .≈ vcat(
                    sum.(eachrow(W1)), sum.(eachrow(W2*W1))
                    )
                )
        end
    end
end
