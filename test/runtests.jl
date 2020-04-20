using nnsim, BlockArrays, Test



@testset "nnsim Tests" begin

    @testset "Neurons" begin
        @testset "LIF" begin
            lif = nnsim.LIF()
            @test lif.state[1] == lif.v0        # Constructor works

            @test begin                         # Adding input to state without time evolution
                update!(lif, 1, 0, 0)
                lif.state[1] == lif.v0 + 1.
            end

            @test begin                         # Time evolution changes the state
                update!(lif, 0, 0.001, 0)
                lif.state[1] != lif.v0 + 1
            end

            @test begin                         # Reset works
                reset!(lif)
                lif.state[1] == lif.v0
            end
        end

        @testset "Izhikevich" begin
            izh = nnsim.Izh()
            v0 = izh.v0
            u0 = izh.u0
            @test all(izh.state .== [v0, u0])   # Constructor works

            @test begin                         # Adding input to state without time evolution
                update!(izh, 1, 0, 0)
                all(izh.state .== [v0 + 1., u0])
            end

            @test begin                         # Time evolution changes the state
                update!(izh, 0, 0.001, 0)
                all(izh.state .!= [v0 + 1., u0])
            end

            @test begin
                reset!(izh)
                all(izh.state .== [v0, u0])
            end                                 # Reset works
        end

        @testset "ReLU" begin
            ReLU = nnsim.ReLU()
            @test ReLU.state[1] == ReLU.v0        # Constructor works

            @test begin                         # Adding input to state without time evolution
                update!(ReLU, 1, 0, 0)
                ReLU.state[1] == 1.
            end

            @test begin                         # Reset works
                reset!(ReLU)
                ReLU.state[1] == ReLU.v0
            end
        end

        @testset "tanh" begin
            n_tanh = nnsim.tanh()
            @test n_tanh.state[1] == n_tanh.v0        # Constructor works

            @test begin                         # Adding input to state without time evolution
                update!(n_tanh, 1, 0, 0)
                n_tanh.state[1] == Base.tanh(1.)
            end

            @test begin                         # Reset works
                reset!(n_tanh)
                n_tanh.state[1] == n_tanh.v0
            end
        end

        @testset "sigmoid" begin
            sigmoid = nnsim.sigmoid()
            @test sigmoid.state[1] == sigmoid.v0        # Constructor works

            @test begin                         # Adding input to state without time evolution
                update!(sigmoid, 1, 0, 0)
                sigmoid.state[1] == 1. / (1. + exp(-1))
            end

            @test begin                         # Reset works
                reset!(sigmoid)
                sigmoid.state[1] == sigmoid.v0
            end
        end
    end

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
        end

    end

    @testset "Utility Functions" begin
        @test begin                             # All neurons in layer initialized correctly
            v0 = nnsim.LIF().state[1]
            layer = layer_constructor(nnsim.LIF, 2, 1, [])
            all([n.state[1] for n in layer.neurons] .== v0)
        end

        @test begin                             # Ensure updating neuron does not influence all neurons in layer
            layer = layer_constructor(nnsim.LIF, 2, 1, [])
            update!(layer.neurons[1],1,0,0)
            layer.neurons[1].state[1] != layer.neurons[2].state[1]
        end

        @test begin                             # Function supports passing kwargs
            W = zeros(2,2)
            b_layer = batch_layer_construction(nnsim.LIF, W, 2, v0 = -40.)
            all([n.v0 for n in b_layer.neurons] .== -40.)
        end


    end

end;
