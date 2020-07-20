"""
	update!(ne::WaspnetElement, input, dt, t)

Evolve the `WaspnetElement` subject to an input `input` a duration of time `dt` starting from time `t`.
"""
function update!(ne::WaspnetElement, input, dt, t) end

function update!(ne::AbstractNeuron, input, dt, t)
	warning("Neurons should use update, not update! function")
	return update(ne, input, dt, t)
end

"""
	reset!(ne::WaspnetElement)

Reset the `WaspnetElement` to a default state defined by the method.
"""
function reset!(ne::WaspnetElement) end

function reset!(ne::AbstractNeuron, input, dt, t)
	warning("Neurons should use reset, not reset! function")
	return reset(ne, input, dt, t)
end

"""
	reset(n::AbstractNeuron)

Resets the neuron and returns the new struct with original values but the state updated.
"""
function reset(n::AbstractNeuron) end

"""
	simulate!(element<:WaspnetElement, input, dt, tf, t0 = 0.; track_state=false, kwargs...)

Simulates a given `WaspnetElement` element from times `t0` to `tf` with increments of `dt` subject to an input `input`, storing the results in a SimulationResult instance.

Note that there are `floor((tf-t0)/dt)+1` samples returned; the first sample point corresponds to `t0`, then at all time points in increments of `dt` beyond `t0` until the final time `tf` is exceeded.

# Arguments
- `element<:WaspnetElement`: The `WaspnetElement` to simulate, typically a `Neuron`, `Layer`, or `Network`. The element's state is modified by simulation.
- `input`: The input to the simulation, currently either a `Matrix` of sampled values or an explicit function of time to sample.
- `dt`: The time step for simulation
- `tf`: The final time for simulation
- `t0`: The initial time for simulation; defaults to 0.
- `track_state`: A flag indicating whether the state of the element being simulated should be tracked by the `SimulationResult`.
"""
function simulate!(ne::WaspnetElement, input, dt, tf, t0 = 0.; track_state=false, kwargs...) end

"""
	function get_neuron_count(ne::WaspnetElement)

Returns the number of neurons in the given `WaspnetElement`
"""
function get_neuron_count(ne::WaspnetElement) end

"""
	function get_neuron_outputs(ne::WaspnetElement)

Returns the current output of all neurons in the given `WaspnetElement`
"""
function get_neuron_outputs(ne::WaspnetElement) end

"""
	function get_neuron_outputs(ne::WaspnetElement)

Returns the current state of all neurons in the given `WaspnetElement`
"""
function get_neuron_states(ne::WaspnetElement) end