using Test, SafeTestsets

@time @safetestset "Neurons" begin include("neuron_tests.jl") end
@time @safetestset "Layers" begin include("layer_tests.jl") end
@time @safetestset "Networks" begin include("network_tests.jl") end
@time @safetestset "Simulations" begin include("simulation_tests.jl") end
@time @safetestset "Utils" begin include("utility_tests.jl") end
