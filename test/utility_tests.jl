using WaspNet
using Distributions

@testset "Utility Functions" begin
    @test begin                             # All neurons in layer initialized correctly
        v0 = WaspNet.LIF().state[1]
        layer = layer_constructor(WaspNet.LIF, 2, 1, [])
        all([n.state[1] for n in layer.neurons] .== v0)
    end

    @test begin                             # Ensure updating neuron does not influence all neurons in layer
        layer = layer_constructor(WaspNet.LIF, 2, 1, [])
        (_, layer.neurons[1]) = update(layer.neurons[1],1,0,0)
        WaspNet.get_neuron_states(layer.neurons[1]) != WaspNet.get_neuron_states(layer.neurons[2])
    end

    @test begin                             # Function supports passing kwargs
        W = zeros(2,2)
        b_layer = batch_layer_construction(WaspNet.LIF, W, 2, v0 = -40.)
        all([n.v0 for n in b_layer.neurons] .== -40.)
    end

    include("pruning_tests.jl")

    @test begin
        inputs = [1,1]
        p = getPoissonST(inputs, Bernoulli)
        spikes = p(1)
        spikes == [0,0] || spikes == [0,1] || spikes == [1,0] || spikes == [1,1]
    end

    @test begin
        inputs = [0,0,1]
        p = getPoissonST(inputs, Bernoulli)
        spikes = p(1)
        spikes == [0,0,1]
    end
end
