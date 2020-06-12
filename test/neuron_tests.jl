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
        @test ReLU.state[1] == 0.        # Constructor works

        @test begin                         # Adding input to state without time evolution
            update!(ReLU, 1, 0, 0)
            ReLU.state[1] == 1.
        end

        @test begin                         # Reset works
            reset!(ReLU)
            ReLU.state[1] == 0.
        end
    end

    @testset "tanh" begin
        n_tanh = nnsim.tanh()
        @test n_tanh.state[1] == 0.        # Constructor works

        @test begin                         # Adding input to state without time evolution
            update!(n_tanh, 1, 0, 0)
            n_tanh.state[1] == Base.tanh(1.)
        end

        @test begin                         # Reset works
            reset!(n_tanh)
            n_tanh.state[1] == 0. 
        end
    end

    @testset "sigmoid" begin
        sigmoid = nnsim.sigmoid()
        @test sigmoid.state[1] == 0.        # Constructor works

        @test begin                         # Adding input to state without time evolution
            update!(sigmoid, 1, 0, 0)
            sigmoid.state[1] == 1. / (1. + exp(-1))
        end

        @test begin                         # Reset works
            reset!(sigmoid)
            sigmoid.state[1] == 0. 
        end
    end

    @testset "identity neuron" begin
        id = nnsim.identity()
        @test id.state[1] == 0.

        @test begin
            update!(id, 42., 0.001, 0.)
            id.state[1] == 42.
        end

        @test begin
            reset!(id)
            id.state[1] == 0.
        end
    end
end
