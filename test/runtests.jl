using nnsim, Test



@testset "nnsim Tests" begin

    @testset "Neurons" begin
        lif = nnsim.LIF()
        @test lif.state[1] == -55.

        update!(lif, 1, 0, 0)
        @test lif.state[1] == -54.

        reset!(lif)
        @test lif.state[1] == -55.
    end

    @testset "Layers" begin
        lif1 = nnsim.LIF()
        lif2 = nnsim.LIF()
        W = zeros(2,2)
        W[1,1] = 1.
        W[2,2] = 1.

        L = Layer([lif1, lif2], zeros(2), W, 2)

        update!(L, [0., 0.], 0.001, 0.)
        @test lif1.state[1] == lif2.state[1]
        s10 = lif1.state[1]
        s20 = lif2.state[1]

        update!(L, [2., 3.], 0., 0.)
        @test lif1.state[1] == s10 + 2.
        @test lif2.state[1] == s20 + 3.

        reset!(L)
        @test lif1.state[1] == -55.
        @test lif2.state[1] == -55.
    end

    @testset "Networks" begin

        
    end


end;