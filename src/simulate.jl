"""
    struct SimulationResult{
        OT<:AbstractArray{<:Number,2}, ST<:AbstractArray{<:Number, 2}, TT<:AbstractArray{<:Real,1}
        }<:AbstractSimulation   

Contains simulation results from simulating a `Network` for a specific length of time with `simulate!`

# Fields
- `outputs::OT`: A `Matrix` containing the output of all simulated neurons at every time step.
- `states::ST`: A `Matrix` containing the state of all simulated neurons at every time step. If states were not tracked, an Nx0 dimensional `Matrix`.
- `times::TT`: An `Array` of times at which the `WaspnetElement` was sampled.
"""
struct SimulationResult{
    OT<:AbstractArray{<:Number,2}, ST<:AbstractArray{<:Number, 2}, TT<:AbstractArray{<:Real,1}, EL<:WaspnetElement
    }<:AbstractSimulation
    outputs::OT
    states::ST
    times::TT 

    function SimulationResult{OT,ST,TT,EL}(outputs::OT, states::ST, times::TT) where {
        OT<:AbstractArray{<:Number,2}, ST<:AbstractArray{<:Number,2}, TT<:AbstractArray{<:Real, 1}, EL<:WaspnetElement
        }
        return new{OT,ST,TT,EL}(outputs, states, times)
    end
end

"""
    SimulationResult(element::EL, times::TT) where {EL<:WaspnetElement,TT<:AbstractArray{<:Real, 1}}

Given a `WaspnetElement` and the times at which to simulate the element, construct the `SimulationResult` instance to store the results of the simulation. 
""" 
function SimulationResult(element::EL, times::TT) where {EL<:WaspnetElement,TT<:AbstractArray{<:Real, 1}}
    cols = length(times) 
    outputs_proto = get_neuron_outputs(element)
    state_proto = get_neuron_states(element)

    n_out = length(outputs_proto)
    n_state = length(state_proto)
    zero_out = zero(outputs_proto[1])
    zero_state = zero(state_proto[1])

    outputs = fill(zero_out, n_out, cols)
    states = fill(zero_state, n_state, cols)
    return SimulationResult(outputs, states, times, element)
end

function SimulationResult(outputs::OT,states::ST,times::TT, element::EL) where {
    OT<:AbstractArray{<:Number,2}, ST<:AbstractArray{<:Number,2}, TT<:AbstractArray{<:Real, 1}, EL<:WaspnetElement
    }
    return SimulationResult{OT,ST,TT,EL}(outputs, states, times)
end

"""
    simulate!(element::WaspnetElement, input::Function, dt, tf, t0 = 0.; track_state=false, kwargs...)

Simulates the supplied `WaspnetElement` subject to a function of time, `input` by sampling `input` at the chosen sample times and returns the relevant `SimulationResult` instance
"""
function simulate!(element::WaspnetElement, input::Function, dt, tf, t0 = 0.; track_state=false, kwargs...)
    t_steps = t0:dt:tf
   
    input_matrix = hcat(input.(t_steps[1:(end-1)])...) 
    return simulate!(element, input_matrix, dt, tf, t0, track_state = track_state, kwargs...)
end

"""
    simulate!(element::WaspnetElement, input::Matrix, dt, tf, t0 = 0.; track_state=false, kwargs...)

Simulates the supplied `WaspnetElement` subject to some pre-sampled `input` where each column is one time step and returns the relevant `SimulationResult` instance
"""
function simulate!(element::WaspnetElement, input::AbstractMatrix, dt, tf, t0 = 0.; track_state=false, kwargs...)
    t_steps = t0:dt:tf
    result = SimulationResult(element, t_steps)

    return simulate!(result, element, input, dt, tf, t0, track_state=track_state, kwargs...) 
end

function simulate!(
    result::SimulationResult, element::WaspnetElement, input::Function, dt, tf, t0 = 0.; track_state=false, kwargs...
    )
    t_steps = t0:dt:tf

    input_matrix = hcat(input.(t_steps[1:(end-1)])...)
    return simulate!(result, element, input_matrix, dt, tf, t0, track_state=track_state, kwargs...)
end

function simulate!(
    result::SimulationResult, element::WaspnetElement, input::AbstractMatrix, dt, tf, t0 = 0.; track_state=false, kwargs...
    )
    t_steps = t0:dt:tf
    t = t0
    N_steps = length(t_steps) - 1

    # The +1 here is to ensure that we get the initial state at t=t0 in the outputs
    copyto!(view(result.outputs,:,1), get_neuron_outputs(element))
    if track_state
        copyto!(view(result.states, :,1), get_neuron_states(element))
    end

    for i in 1:N_steps
        sim_update!(element, input[:,i], dt, t)
        # The +1 here is the offset from including the initial values in the outputs
        copyto!(view(result.outputs, :, i+1), get_neuron_outputs(element))
        if track_state
            copyto!(view(result.states, :, i+1), get_neuron_states(element))
        end
        t += dt
    end
    return result
end

"""
    function sim_update!(ne::WaspnetElement, input_update, dt, t)

Generic function for wrapping calls to `update!` from `simulate!`
"""
function sim_update!(ne::WaspnetElement, input_update, dt, t)
    update!(ne, input_update, dt, t)
end

"""
    function sim_update!(neuron::AbstractNeuron, input_update<:AbstractArray{T,N}, dt, t) where {T<:Number, N}

Wrapper to ensure that if a 1D array is passed to update a neuron, it is converted to a scalar first
"""
function sim_update!(neuron::AbstractNeuron, input_update::AbstractArray{T,N}, dt, t) where {T<:Number, N}
    return update!(neuron, input_update[1], dt, t)
end