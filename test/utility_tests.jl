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
