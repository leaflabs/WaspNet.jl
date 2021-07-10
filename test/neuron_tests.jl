using WaspNet

@testset "Neurons" begin
    @testset "LIF" begin
        lif = WaspNet.LIF()
        @test lif.state == lif.v0        # Constructor works

        @test begin                         # Adding input to state without time evolution
            (_, lif) = update(lif, 1, 0, 0)
            lif.state == lif.v0 + lif.R / lif.τ
        end

        @test begin                         # Time evolution changes the state
            (_, lif) = update(lif, 0, 0.001, 0)
            lif.state != lif.v0 + lif.R / lif.τ
        end

        @test begin                         # Reset works
            lif = WaspNet.reset(lif)
            lif.state == lif.v0
        end
    end

    @testset "Izhikevich" begin
        izh = WaspNet.Izh()
        v0 = izh.v0
        u0 = izh.u0
        @test all((izh.v, izh.u) .== (v0, u0))   # Constructor works

        @test begin                         # Adding input to state without time evolution
            (_, izh) = update(izh, 1, 0, 0)
            all((izh.v, izh.u) .== (v0 + 1., u0))
        end

        @test begin                         # Time evolution changes the state
            (_, izh) = update(izh, 0, 0.001, 0)
            all((izh.v, izh.u) .!= (v0 + 1., u0))
        end

        @test begin
            izh = WaspNet.reset(izh)
            all((izh.v, izh.u) .== (v0, u0))
        end                                 # Reset works
    end

    @testset "ReLU" begin
        ReLU = WaspNet.ReLU()
        @test ReLU.state == 0.        # Constructor works

        @test begin                         # Adding input to state without time evolution
            (_, ReLU) = update(ReLU, 1, 0, 0)
            ReLU.state == 1.
        end

        @test begin                         # Reset works
            ReLU = WaspNet.reset(ReLU)
            ReLU.state == 0.
        end
    end

    @testset "tanh" begin
        n_tanh = WaspNet.tanh()
        @test n_tanh.state == 0.        # Constructor works

        @test begin                         # Adding input to state without time evolution
            (_, n_tanh) = update(n_tanh, 1, 0, 0)
            n_tanh.state == Base.tanh(1.)
        end

        @test begin                         # Reset works
            n_tanh = WaspNet.reset(n_tanh)
            n_tanh.state == 0.
        end
    end

    @testset "sigmoid" begin
        sigmoid = WaspNet.sigmoid()
        @test sigmoid.state == 0.        # Constructor works

        @test begin                         # Adding input to state without time evolution
            (_, sigmoid) = update(sigmoid, 1, 0, 0)
            sigmoid.state == 1. / (1. + exp(-1))
        end

        @test begin                         # Reset works
            sigmoid = WaspNet.reset(sigmoid)
            sigmoid.state == 0.
        end
    end

    @testset "identity neuron" begin
        id = WaspNet.identity()
        @test id.state == 0.

        @test begin
            (_, id) = update(id, 42., 0.001, 0.)
            id.state == 42.
        end

        @test begin
            id = WaspNet.reset(id)
            id.state == 0.
        end
    end
end
