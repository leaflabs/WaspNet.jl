using nnsim, Test



@testset "nnsim Tests" begin

    @testset "Neurons" begin
        @testset "LIF" begin
            lif = nnsim.LIF()
            @test lif.state[1] == lif.v0        # Constructor works

            update!(lif, 1, 0, 0)
            @test lif.state[1] == lif.v0 + 1.   # Adding input to state without time evolution

            update!(lif, 0, 0.001, 0)
            @test lif.state[1] != lif.v0 + 1    # Time evolution changes the state

            reset!(lif)
            @test lif.state[1] == lif.v0        # Reset works
        end

        @testset "Izhikevich" begin
            izh = nnsim.Izh()            
            v0 = izh.v0
            u0 = izh.u0
            @test all(izh.state .== [v0, u0])   # Constructor works

            update!(izh, 1, 0, 0)
            @test begin                         # Adding input to state without time evolution
                all(izh.state .== [v0 + 1., u0])    
            end                                 

            update!(izh, 0, 0.001, 0)           
            @test begin                         # Time evolution changes the state
                all(izh.state .!= [v0 + 1., u0])
            end                                 

            reset!(izh)
            @test begin
                all(izh.state .== [v0, u0])      
            end                                 # Reset works
        end
    end

    @testset "Layers" begin
        lif1 = nnsim.LIF()
        lif2 = nnsim.LIF()
        W = zeros(2,2)
        W[1,1] = 1.
        W[1,2] = 2.
        W[2,2] = 1.

        L = Layer([lif1, lif2], zeros(2), [0], W, 2)

        update!(L, [[0., 0.]], 0.001, 0.)
        @test lif1.state[1] == lif2.state[1]    # Both neurons are initialized to same state
        s10 = lif1.state[1]
        s20 = lif2.state[1]

        update!(L, [[2., 3.]], 0., 0.)
        @test lif1.state[1] == s10 + 8.         # Inputs are routed correctly
        @test lif2.state[1] == s20 + 3.         

        reset!(L)
        @test lif1.state[1] == -55.             # All neurons reset
        @test lif2.state[1] == -55.

        b_layer = batch_layer_construction(nnsim.LIF, W, 2)
        v0 = lif1.v0
        @test begin                             # All neurons in layer initialized correctly
            all([n.state[1] for n in b_layer.neurons] .== v0)
        end

        b_layer = batch_layer_construction(nnsim.LIF, W, 2, v0 = -40.)
        @test begin                             # Function supports passing kwargs
            all([n.v0 for n in b_layer.neurons] .== -40.)
        end
    end

    @testset "Networks" begin
        N = 32
        N_in = 16
        
        @testset "Homogeneous Networks" begin 
            L1 = layer_constructor(nnsim.LIF, N, 2, [0])
            L2 = layer_constructor(nnsim.LIF, N, 2, [1])
            net_hom = Network([L1, L2])
            state0 = L1.neurons[1].state[1]
            v0 = L1.neurons[1].v0

            @test begin                         # All neurons initialized correctly
                all(nnsim.get_neuron_states(net_hom) .== state0)
            end

            @test begin                         # Update works, not evolving system/state unchanged
                update!(net_hom, zeros(Float64, 32), 0, 0)
                all(nnsim.get_neuron_states(net_hom) .== state0)
            end

            @test begin                         # Neuron Outputs function works
                all(nnsim.get_neuron_outputs(net_hom) .== 0)
            end

            input_fun(t) = zeros(Float64, 32) 

            outputs, states = simulate!(net_hom, input_fun, 0.001, 1., track_flag = true)                
            @test begin                         # Check neuron output matrix size
                all(size(outputs) .== [2*N, 1001])            
            end

            @test begin                         # Check neuron output matrix size
                all(size(states) .== [2*N, 1001])            
            end

            reset!(net_hom)
            @test begin                         # Reset the full network
                all(nnsim.get_neuron_states(net_hom) .== v0)
            end
        end

    end

end;