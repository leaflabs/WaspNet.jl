# %%
using PyPlot
using Statistics 

using nnsim

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

    zeros_and_spike = vcat(zeros(quiet_length), amplitude)
    in_vec = vcat([zeros_and_spike for j in 1:round(Int, sim_length/quiet_length + 1)]...)

    output = [] 
    for (i,t) in enumerate(sim_times)
        push!(output, update!(neuron, in_vec[i], dt, t))
    end

    return output
end 

# %%
# Let's sweep over the amplitude and check the spiking frequency with a spike coming in every 30 ms
izh_neuron = nnsim.Izh()

quiet_lengths = [9, 14, 19, 24, 29, 34, 39, 44, 49] # spike every 29+1 ms
Tsim = 5. # seconds
dt = 0.0001

amps = 1:0.1:40

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
plt.legend(["$x ms" for x in (quiet_lengths.+1)])

.savefig("/home/buercklin/Documents/Figures/nnsim/neuron_dynamics/spike_freqs.png");
# %%
# %%
# %%
# %%


# %%