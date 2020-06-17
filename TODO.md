- [X] `Network` constructor method to take FF layers and make FF network without specifying connections
- [X] Generalize `Network` constructor function so all methods call to one primary method
- [ ] Utility to generate networks of varying widths
- [ ] Utility functions should not assume inputs are connected *directly* to neurons

- [ ] HH Neuron Model (including tests)
- [ ] QIF Neuron Model (including tests)
- [ ] FNH Neuron Model (including tests)

- [ ] Tests for all utility functions in multiple use-cases
- [X] Test deepcopy utility function
- [ ] Tests for heterogeneous networks

- [X] Update `Layer` to handle the weight matrices at construction-time for both Arrays + Block Arrays
- [X] Update `Layer` to preallocate space for input vectors to avoid re-allocation at every `update!`
- [X] Organize the `Layer` constructors better; agree on typical use case and write for that
- [ ] Test all `Layer` method

- [X] Feed `Network` with a `Matrix` where each column is input at one time step
- [ ] Tests for `Network` with `Matrix` input
- [X] Fix Izhikevich time scales

- [X] Move neurons to be in their own source folders

====================================================================================================

- [ ] Docstrings
	- [X] Izh Neuron
	- [X] LIF Neuron
	- [X] Functional neurons
		- [X] Functional refactor
	- [X] `Layer`
	- [ ] `Network`
	- [ ] Generic functions for docs (?)
		- [X] Added `defs.jl` for abstract functions for documentation
- [ ] Optimizations
	- [X] Do `Layer` updates in-place
	- [ ] Do `Network` updates in-place
	- [ ] Remove broadcasting in `Layer` and `Network`
	- [ ] Maybe change `Network.prev_outputs` to a `BlockArray`, this Vector-of-Vector business is questionable
- [ ] Simple example
	- [ ] ???
- [ ] Review tests for relevance
- [ ] Re-type to <:AbstractArray{T,dims} 
	- [X] LIF, Izh, Functional
	- [X] Layer
	- [ ] Network

====================================================================================================

- [ ] Move simulation to something outside of the `Network`
	- [ ] Remove `neur_states`, `neur_outputs` from `Network` (leverage the `prev_outputs` fields heavily here)
	- [ ] New `NetworkSimOutput` type which holds the results, manages time, maybe useful for plotting
	- [ ] Make sure all of the `Network` tests are still passing