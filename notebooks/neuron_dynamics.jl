# %%
using PyPlot
using Statistics 

using nnsim

# %%
#=
    Generate a length of zeros punctuated by a spike
=#
# %%
function generate_spike_train(amplitude, quiet_length, dt = 0.001)
    quiet_steps = floor(Int, quiet_length/dt) - 1
    return vcat(zeros(quiet_steps), amplitude)
end

# %%
#= 
Let's see what sort of firing rates we can expect from a standard Izh neuron 
subject to a variety of input spike amplitudes and frequencies
=#

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

savefig("/home/buercklin/Documents/Figures/nnsim/neuron_dynamics/spike_freqs.png");

# %%
#= 
Let's look at how firing rates scale as a function of number of neurons
in a linear chain. We'll only have a single neuron in each layer and examine
the spiking rate the the output layer. 
=#

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
    layers = [Layer([neuron()], zeros(1), ones(1,1)*amplitude, 1)  for _ in 1:N_layers]
    return Network(layers)
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

savefig("/home/buercklin/Documents/Figures/nnsim/neuron_dynamics/spike_freqs_by_layer_single_chain.png");

# %%
# %%
# %%
# %%
# %%
# %%
# %%

# %%