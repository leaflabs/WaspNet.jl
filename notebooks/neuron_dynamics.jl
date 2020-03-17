# %%
using PyPlot
using Statistics 
using Distributions # Used to generate 
using Revise        # Used to make edits to source, not necessary for running

using nnsim

# %%
#================================================================================
    Generate an array of zeros punctuated by a spike
================================================================================= =#
# %%
function generate_spike_train(amplitude, quiet_length, dt = 0.001)
    quiet_steps = floor(Int, quiet_length/dt) - 1
    return vcat(zeros(quiet_steps), amplitude)
end

# %%
#================================================================================
    Given a binary matrix representing spike events for each neuron,
    compute statistics for the spike trains. Assumes a rectangular (uniform)
    shape for the network    
================================================================================= =#
# %%
function analyze_spike_matrix(spike_mat, layers, dt = 0.001)
    n_neur = size(spike_mat, 1)
    width = n_neur ÷ layers

    stats = Dict{Any,Any}("network" => _analyze_spike_set(spike_mat, dt))

    for l in 1:layers
        neuron_idxs = (l-1)*width .+ (1:width)
        stats[l] = _analyze_spike_set(spike_mat[neuron_idxs, :], dt)
        for n in 1:width
            stats[(l,n)] = _analyze_spike_set(spike_mat[(l-1)*width+n, :], dt)
        end
    end
    return stats
end

# ber = Bernoulli(0.05)
# data = rand(ber, 32, 5001)
# a_dict = analyze_spike_matrix(data, 4)

# println("Network: ", a_dict["network"])
# println()
# for j in 1:4
#     println("Layer $(j): ", a_dict[j])
#     for k in 1:8
#         println("Neuron ($(j),$(k)): ", a_dict[(j,k)])
#     end
#     println()
# end

# %%
function _analyze_spike_set(spikes, dt = 0.001)
    if ndims(spikes) == 1
        spikes = reshape(spikes, 1, length(spikes))
    end
    freqs = mean(spikes, dims = 2)/dt
    _mean = mean(freqs)
    _median = median(freqs)
    _max = maximum(freqs)
    _min = minimum(freqs)
    return (_mean, _median, _min, _max)
end

# ber = Bernoulli(0.2)
# data = rand(ber, 5, 1000)
# println(_analyze_spike_set(data))
# for j in 1:5
#     println(_analyze_spike_set(data[j,:]))
# end

# %%
#================================================================================


Let's see what sort of firing rates we can expect from a standard Izh neuron 
subject to a variety of input spike amplitudes and frequencies


================================================================================= =#

# %%
# Simulate a neuron that has a spike input every quiet_length+1 steps
#   and return the output spike train
# To convert to quiet time, use (quiet_length+1)*dt
function test_neuron!(neuron, amplitude, quiet_length, Tsim, dt = 0.001)
    reset!(neuron)

    sim_times = 0.:dt:Tsim
    sim_length = length(sim_times)

    zeros_and_spike = generate_spike_train(amplitude, quiet_length, dt)
    spike_length = length(zeros_and_spike)

    output = [] 
    for (i,t) in enumerate(sim_times)
        push!(output, update!(neuron, zeros_and_spike[(i-1) % spike_length + 1], dt, t))
    end

    return output
end 

# %%
# Let's sweep over the amplitude and check the spiking frequency with a spike coming in every 30 ms
izh_neuron = nnsim.Izh()

quiet_lengths = [0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05] 
Tsim = 5. # seconds
dt = 0.001

amps = -10:0.5:40

fire_freq = []
for (i,quiet_length) in enumerate(quiet_lengths)
    push!(fire_freq, [])
    for amp in amps
        fire_out = test_neuron!(izh_neuron, amp, quiet_length, Tsim, dt)
        push!(fire_freq[i], mean(fire_out)/dt )
    end
end

# %%
plt.plot(amps, hcat(fire_freq...))
plt.xlabel("Incoming Amplitude (mV)")
plt.ylabel("Spiking Frequency (Hz)")
plt.title("Spiking Frequency at various Amplitudes, Incoming Spike Periods")
plt.legend(["$(round(Int, x*1000)) ms" for x in (quiet_lengths)])

# savefig("/home/buercklin/Documents/Figures/nnsim/neuron_dynamics/spike_freqs.png");
# %%
#================================================================================


Let's look at how firing rates scale as a function of number of neurons
in a linear chain. We'll only have a single neuron in each layer and examine
the spiking rate the the output layer. 


================================================================================= =#

# %%
function test_network!(network, quiet_length, Tsim, dt = 0.001)
    reset!(network)
    sim_times = 0.:dt:Tsim
    sim_length = length(sim_times)

    # Note that we set the amplitude to 1 here b/c the weights handle the amplitude
    zeros_and_spike = generate_spike_train(1, quiet_length, dt)
    spike_length = length(zeros_and_spike)

    output = [] 
    for (i,t) in enumerate(sim_times)
        update!(network, [ zeros_and_spike[(i-1) % spike_length + 1] ], dt, t)
        all_out = nnsim.get_neuron_outputs(network)
        push!(output, all_out)
    end

    return hcat(output...)
end

function construct_linear_network(N_layers, amplitude, neuron = nnsim.Izh)
    dist = Normal(amplitude, 0.) # amplitude-mean, 0 variance
    return network_constructor(N_layers, 1, n_constr = neuron, init_dist = dist)
    # layers = fill(Layer([neuron()], zeros(1), ones(1,1)*amplitude, 1), N_layers)
    # return Network(layers)
end

# %%
max_layers = 5

Tsim = 5. # seconds
# quiet_length = 0.010
quiet_length = 0.005
dt = 0.0001

amps = -10:0.5:40

fire_freqs = []
for amp in -10:0.5:40
    test_net = construct_linear_network(max_layers, amp)
    spikes = test_network!(test_net, quiet_length, Tsim, dt)
    amp_freqs = [mean(spikes[i,:])/dt for i in 1:size(spikes,1)]
    push!(fire_freqs, amp_freqs)
end

# %%
plt.plot(amps, transpose(hcat(fire_freqs...)))
plt.ylim([-2, 32])
plt.xlabel("Incoming Amplitude (mV)")
plt.ylabel("Spiking Frequency (Hz)")
plt.title("Spiking Frequency at various Amplitudes, Layer Level")
plt.legend(["Layer $(x)" for x in 1:max_layers])

# savefig("/home/buercklin/Documents/Figures/nnsim/neuron_dynamics/spike_freqs_by_layer_single_chain.png");

# %%
#================================================================================


Now we'll consider the spiking rates of a network with some width subject to a 
single neuron at the input layer being driven by a constant amplitude with
random (dense) connections between layers.


================================================================================= =#

# %%
# Unlike before, we're driving the input layer with a constant amplitude,
#   not a periodic amplitude, 
function test_wide_network!(network, quiet_length, Tsim, l_width, dt = 0.001)
    reset!(network)
    
    sim_times = 0.:dt:Tsim
    sim_length = length(sim_times)

    # Note that we set the amplitude to 1 here b/c the weights handle the amplitude
    zeros_and_spike = generate_spike_train(1, quiet_length, dt)
    spike_length = length(zeros_and_spike)

    output = [] 
    for (i,t) in enumerate(sim_times)
        inval = zeros_and_spike[(i-1) % spike_length + 1]
        update!(network, vcat(inval, zeros(l_width-1)), dt, t)
        all_out = nnsim.get_neuron_outputs(network)
        push!(output, all_out)
    end

    return hcat(output...)   
end

# %%
function construct_wide_network(N_layers, width, distr, neuron = nnsim.Izh)
    W1 = zeros(width, width)
    W1[1,1] = mean(distr)
    layers = [batch_layer_construction(neuron, W1, width)]
    for j in 1:(N_layers-1)
        push!(layers, batch_layer_construction(neuron, rand(distr, width, width), width))
    end
    return Network(layers)
end

# %%
N_layers = 12 
width = 32 

Tsim = 40. # seconds
quiet_length = 0.01
dt = 0.001

amps = -10:0.5:10
variance = 16

fire_freqs = []
spike_list = []
retval = 1
for amp in amps
    println(amp)
    distr = Normal(amp, variance)
    test_net = construct_wide_network(N_layers, width, distr)

    spikes = test_wide_network!(test_net, quiet_length, Tsim, width, dt)
    output = analyze_spike_matrix(spikes, N_layers)

    push!(spike_list, spikes)
    push!(fire_freqs, [output[j][1] for j in 2:N_layers])
end
# %%
plt.plot(amps, fire_freqs)
plt.legend(["Layer $(j)" for j in 2:N_layers])
plt.xlabel("Mean Spike Amplitude (mV)")
plt.ylabel("Mean Spiking Rate (Hz)")
plt.title("Spiking Rates vs Spike Amplitudes for a $(N_layers)x$(width) network")
# savefig("/home/buercklin/Documents/Figures/nnsim/neuron_dynamics/spike_freqs_by_layer_wide.png");

# %%
layer_cutoffs = hcat([32*j*ones(length(times)) for j in 1:(N_layers-1)]...)
times = collect(0:dt:Tsim)

plt.figure(figsize=(8,10))
plt.subplot(2,1,1)
trial = 30
xs = []
ys = []
for pt in findall(spike_list[trial].>0)
    push!(xs, pt[2]*dt)
    push!(ys, pt[1])
end
plt.scatter(xs, ys)
plt.plot(times, layer_cutoffs, color="r")
plt.xlabel("Time (s)")
plt.ylabel("Neuron");
plt.title("Spiking Events with Amplitude = $(amps[trial]) mV")

plt.subplot(2,1,2)
trial = 40
xs = []
ys = []
for pt in findall(spike_list[trial].>0)
    push!(xs, pt[2]*dt)
    push!(ys, pt[1])
end
plt.scatter(xs, ys)
plt.plot(times, layer_cutoffs, color="r")
plt.xlabel("Time (s)")
plt.ylabel("Neuron");
plt.title("Spiking Events with Amplitude = $(amps[trial]) mV")
savefig("/home/buercklin/Documents/Figures/nnsim/neuron_dynamics/spike_events_wide.png");

# %%
# %%
# %%