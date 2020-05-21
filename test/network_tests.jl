@testset "Networks" begin
        N = 32
        N_in = 16

        @testset "Homogeneous Networks FF" begin
            neurons1 = [nnsim.LIF() for _ in 1:N]
            W1 = randn(N, N_in)
            neurons2 = [nnsim.LIF() for _ in 1:N]
            W2 = randn(N, N)
            L1 = Layer(neurons1, W1)
            L2 = Layer(neurons2, W2)
            net_hom = Network([L1, L2], N_in)

            @test begin                         # Network should change Layer `conns`
                ( all(net_hom.layers[1].conns .== [0]) &&
                    all(net_hom.layers[2].conns .== [1]) )
            end

            @test begin                         # Layers should now have BlockArrays
                isa(net_hom.layers[1].W, AbstractBlockArray) &&
                    isa(net_hom.layers[2].W, AbstractBlockArray)
            end

            @test begin                         # Layer block arrays should have correct size
                ( all(size(net_hom.layers[1].W) .== [N, N_in + N + N]) &&
                    all(size(net_hom.layers[2].W) .== [N, N_in + N + N]) )
            end

            state0 = L1.neurons[1].state[1]
            v0 = L1.neurons[1].v0

            @test begin                         # All neurons initialized correctly
                all(nnsim.get_neuron_states(net_hom) .== state0)
            end

            @test begin                         # Layers are changed to have correct `conns`
                (all(net_hom.layers[1].conns .== [0]) && all(net_hom.layers[2].conns .== [1]))
            end

            @test begin                         # Update works, not evolving system/state unchanged
                update!(net_hom, zeros(Float64, N_in), 0, 0)
                all(nnsim.get_neuron_states(net_hom) .== state0)
            end

            @test begin                         # Neuron Outputs function works
                all(nnsim.get_neuron_outputs(net_hom) .== 0)
            end

            input_fun(t) = zeros(Float64, N_in)
            outputs, states = simulate!(net_hom, input_fun, 0.001, 1., track_flag = true)
            @test begin                         # Check neuron output matrix size
                all(size(outputs) .== [2*N, 1001])
            end

            @test begin                         # Check neuron output matrix size
                all(size(states) .== [2*N, 1001])
            end

            @test begin                         # Reset the full network
                reset!(net_hom)
                all(nnsim.get_neuron_states(net_hom) .== v0)
            end

            #### Test simulate! but with Matrix inputs)
            reset!(net_hom)
            outputs, states = simulate!(net_hom, zeros(N_in, 1000), 0.001, track_flag = true)
            @test begin                         # Check neuron output matrix size
                all(size(outputs) .== [2*N, 1001])
            end

            @test begin                         # Check neuron output matrix size
                all(size(states) .== [2*N, 1001])
            end

        end

    end