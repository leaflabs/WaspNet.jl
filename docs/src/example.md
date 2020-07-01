# Example
Here we step through an example to demonstrate typical usage of `WaspNet`. We begin by building a new `LIF Neuron` type which encapsulates the update rules for a neuron. Then, we make two `Layer`s of neurons to communicate back and forth; one `Layer` is strictly feed-forward, the other is a recurrent layer accepting inputs both from itself and from the preceding `Layer`. Finally, we build a `Network` out of these `Layer`s and simulate it to observe the evolution of the `Network` as a whole.
## Getting Started
The following code has been tested in Julia 1.4.2 and executes without errors. Any dependencies will be called out as necessary. 

To start, the only necessary dependency will be `WaspNet` itself, so we start by importing it. Other useful packages will be `Random` and `BlockArrays`

```
using WaspNet
```
## Constructing a New Neuron
The simlpest unit in `WaspNet` is the `Neuron` which translates an input signal from preceding neurons into the evolution of an internal state and ultimately spikes which are sent to neurons down the line.

A concrete `AbstractNeuron` needs to cover 3 things:
 - A new `struct` which is a subtype of `AbstractNeuron`; optionally mutable
 - An `update!` method to implement the dynamics of the neuron
 - A `reset!` method which restores the neuron to its default state.

We will implement the [Leaky Integrate-&-Fire](https://en.wikipedia.org/wiki/Biological_neuron_model#Leaky_integrate-and-fire) neuron model here, but a slightly different implementation is available in `WaspNet/src/neurons/lif.jl` or with `WaspNet.LIF`. 

A concrete `AbstractNeuron` implementation currently must include two specific fields: `state` and `output`. `state` holds the current state of the neuron in a `Vector` and `output` holds the output of the neuron after its last update; for a spiking neuron, update is either a `0` or a `1` to denote whether a spike did or did not occur. Additional fields should be implemented as needed to parameterize the neuron. For performance, our implementation ofthe struct is immutable, so `state` and `output` must be `Array`s in order to have their values change.
```
struct LIF{T<:Number,A<:AbstractArray{T, 1}}<:AbstractNeuron 
    τ::T
    R::T
    θ::T
    I::T

    v0::T
    state::A
    output::A
end

# output

```
Additionally, we need to define how to evolve our neuron given a time step. This is done by adding a method to `WaspNet.update!`,  a function which is global across all `WaspnetElements`. To `update!` a neuron, we provide the `neuron` we need to update, the `input_update` to the neuron, the time duration to evolve `dt`, and the current global time `t`. In the LIF case, the `input_update` is a voltage which must be added to the membrane potential of the neuron resulting from spikes in neurons which feed into the current neuron. `reset!` simply restores the state of the neuron to its some state.

We use an [Euler update](https://en.wikipedia.org/wiki/Euler_method) for the time evolution because of its simplicity of implementation.

Note that both `update!` and `reset!` are defined *within* `WaspNet`; that is, we actually define the methods `WaspNet.update!` and `WaspNet.reset!`. If defined externally, these methods are not visible to other methods from within `WaspNet`.
```
function WaspNet.update!(neuron::LIF, input_update, dt, t) 
    neuron.output[1] = 0 # Reset spikes
    neuron.state[1] += input_update # If an impulse came in, add it

    # Euler method update
    neuron.state[1] += (dt/neuron.τ) * (-neuron.state[1] + neuron.R*neuron.I)

    # Check for spiking
    if neuron.state[1] >= neuron.θ
        neuron.state[1] = neuron.v0
        neuron.output[1] = 1 # Binary output
    end

    return neuron.output[1] 
end

# output

```
Now we want to instantiate our `LIF` neuron, update it a few times to see the state of the neuron change
```
neuronLIF = LIF(8., 10.E2, 30., 40., -55., [-55.], [0.])

println(neuronLIF.state)
# [-55.0]
update!(neuronLIF, 0., 0.001, 0)
println(neuronLIF.state)
# [-49.993125]

reset!(neuronLIF)
println(neuronLIF.state)
# [-55.0]
```
We can also `simulate!` a neuron, chaining together multiple `update!` calls and returning the outputs (spikes) and optionally the internal state of the neuron as well. The following code simulates our `LIF` neuron for 250 ms with a 0.1 ms time step. The input to the neuron is a function of one parameter, `t`, defined by `(t) -> 0.4*exp(-4t)`.
```
LIFsim = simulate!(neuronLIF, (t)->0.4*exp(-4t), 0.0001, 0.250, track_state=true);
```
`LIFsim` is a `SimulationResult` instance with three fields: `LIFsim.outputs`, `LIFsim.states`, and `LIFsim.times`. `times` holds all of the times at which the neuron was simualted, `outputs` holds the output of the neuron at each time step, and `states` hold the state of the neuron at each time step.
## Combining Neurons into a Layer
In `WaspNet`, a collection of neurons is called a `Layer` or a population. A `Layer` is homogeneous insofar as all of the `Neuron`s in a given `Layer` must be of the same type, although their individual parameters may differ. The chief utility of a `Layer` is to handle the computation of the inputs into its constituent `Neuron`s; which is handled through a multiplication of the input spike vector by a corresponding weight matrix, `W`.

The following code constructs a feed-forward `Layer` with `N` `LIF` neurons inside of it with an incoming weight matrix `W` to handle 2 inputs. 
```
N = 8;
neurons = [LIF(8., 10.E2, 30., 40., -55., [-55.], [0.]) for _ in 1:N];
weights = randn(MersenneTwister(13371485), N,2);
layer = Layer(neurons, weights);
```
We can also `update!` a `Layer` by driving it with some input as we did for our `LIF` neuron above. Not that `input` here is actually an `Array{Array{<:Number, 1}, 1}` and not just `Array{<:Number, 1}`. The purpose of this is to handle recurrent or non-feed-forward connections; we will discuss this more in [Constructing Networks from Layers](@ref).
```
reset!(layer)
update!(layer, [[0.5, 0.8]], 0.001, 0)
println(WaspNet.get_neuron_states(layer))
# [-49.541978928637135, -49.60578871324857, ..., -50.84036022383181]
```
And we can `simulate!` with the same syntax as before
```
layersim = simulate!(layer, (t) -> [randn(2)], 0.001, 0.25, track_state=true);
```
Now `layersim.outputs` and `layersim.states` will be of size `NxT` where there are `N` neurons in the `Layer` and `T` time steps.

There are several `Layer` constructors and `update!` methods; for more information, see [Reference](@ref) or type `?Layer` or `?update!` in the REPL.
## Constructing Networks from Layers
Once we have `Layer`s available, we need a way to communicate spikes between them. The `Network` solves exactly that problem: it orchestrates communication of spiking (or output signals in general) between `Layer`s, routing the appropriate outputs between `Layer`s. 

We'll start by constructing a new first `Layer` for our `Network` similar to how we did before with the added parameter of `Nin`, the number of inputs we're feeding into the first `Layer` of the `Network`.
```
Nin = 2
N1 = 3
neurons1 = [LIF(8., 10.E2, 30., 40., -55., [-55.], [0.]) for _ in 1:N1]
weights1 = randn(MersenneTwister(13371485), N1, Nin)
layer1 = Layer(neurons1, weights1);
```
Now we'll make our second `Layer`. This `Layer` is special: it will take feed-forward inputs from the first `Layer`, but also a recurrent connection to itself. This means that we need to specify `W` slightly differently, and we also need to supply a new field, `conns`. To handle non-feed-forward connections in a `K`-layer, `W` must be declared as a `1x(K+1)` `BlockArray`. 

For our case, `K=2`. Thus, the first block in `W` holds the input weights corresponding to the `Network` input, the second block holds the weights for the first `Layer`, and the third block holds the weights for the second `Layer` feeding back into itself. Similarly we must supply `conns`, an array stating which `Layer`s the current `Layer` connects to. Entries in `conns` are indexed such that `0` corresponds to the `Network` input, `1` corresponds to the output of the first `Layer` and so on. 
```
N2 = 4;
neurons2 = [LIF(8., 10.E2, 30., 40., -55., [-55.], [0.]) for _ in 1:N2]

W12 = randn(N2, N1) # connections from layer 1
W22 = 5*randn(N2, N2) # Recurrent connections
row_block_sizes = [N2]
col_block_sizes = [Nin, N1, N2]

weights2 = BlockArray(zeros(N2, Nin+N1+N2), row_block_sizes, col_block_sizes) 
weights2[Block(1,2)] .= W21
weights2[Block(1,3)] .= W22

conns = [1, 2]

layer2 = Layer(neurons2, weights2, conns);
```
To form a `Network`, we specify the constituent `Layer`s
```
mynet = Network([layer1, layer2], Nin)
```
`update!`, `reset!` work just as they did for `Layer`s and `Neuron`s
```
reset!(mynet)
update!(mynet, 0.4*ones(Nin), 0.001, 0)
println(WaspNet.get_neuron_states(mynet))
# [-49.61444931320574, -50.42051080817629, ..., -49.993125]
```
As does `simulate!`
```
reset!(mynet)
netsim = simulate!(mynet, (t) -> 0.4*ones(Nin), 0.001, 1, track_state=true)
```