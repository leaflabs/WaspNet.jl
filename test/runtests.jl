using nnsim, BlockArrays, Test

@testset "nnsim Tests" begin

    include("neuron_tests.jl")
    include("layer_tests.jl")
    include("network_tests.jl")
    include("simulation_tests.jl")
    include("utility_tests.jl")


end;
