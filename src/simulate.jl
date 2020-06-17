"""
    struct SimulationResult{
        NT<:Network, OT<:AbstractArray{<:Number,2}, ST<:AbstractArray{<:Number, 2}, TT<:AbstractArray{<:Real,1}
        }<:AbstractSimulation   

Contains simulation results from simulating a `Network` for a specific length of time with `simulate!`.
"""
struct SimulationResult{
    NT<:Network, OT<:AbstractArray{<:Number,2}, ST<:AbstractArray{<:Number, 2}, TT<:AbstractArray{<:Real,1}
    }<:AbstractSimulation
    network::NT
    neuron_outputs::OT
    neuron_states::ST
    times::TT
end

# Simulate the network from `t0` to `tf` with a time step of `dt` with an input to
#   the first layer of `input`
function simulate!(network::Network, input::Function, dt, tf, t0 = 0.; track_state=false, kwargs...)
    # t_steps are time points where we evaluate the input function
    # There are ((tf-t0)/dt)+1 time steps, including t0 as the first time step
    # Thus, we evolve the network assuming it is evaluated at the initial time step for
    # every time step, meaning the network ends with t=tf but the last evaluation of input
    # happens at time t=tf-dt
    t_steps = t0:dt:(tf-dt)
   
    input_matrix = hcat(input.(t_steps)...) 
    return simulate!(network, input_matrix, dt, t0, track_flag = track_flag)
end

function simulate!(network::Network, input::Matrix, dt, t0 = 0.; track_state=false, kwargs...)
    t = t0
    N_steps = size(input)[2]

    # The +1 here is to ensure that we get the initial state at t=t0 in the outputs
    network.neur_outputs = Array{Any, 2}(undef, get_neuron_count(network), N_steps+1) 
    network.neur_outputs[:, 1] = get_neuron_outputs(network) 
    if track_flag
        network.neur_states = Array{Any, 2}(undef, network.state_size, N_steps+1)
        network.neur_states[:,1] .= get_neuron_states(network)
    end

    for i in 1:N_steps
        update!(network, input[:,i], dt, t)
        # The +1 here is the offset from including the initial values in the outputs
        network.neur_outputs[:, i+1] = get_neuron_outputs(network)
        if track_flag
            network.neur_states[:,i+1] .= get_neuron_states(network)
        end
        t += dt
        network.t += dt
    end

    if track_flag
        return network.neur_outputs, network.neur_states
    else
        return network.neur_outputs
    end 

end