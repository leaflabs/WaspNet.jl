using nnsim
using Parameters # Not necessary, but highly recommended for automating the generation of constructors for types
using PyPlot
using StatsBase
using LinearAlgebra
using Plots

# Lets make them all uniform
N = 100
a1 = 0.02
b1 = 0.2
c1 = -65.
d1 = 8.
recurrent = true

# Keep random connectivity within the layer for now
W = randn(N,2*N)
# Let's start with just two layers
layer = batch_layer_construction(nnsim.Izh, W, N; a = a1, b = b1, c = c1, d = d1, I = 5*ones(N));
net = Network([layer,layer]);

network_constructor([layer])

net = network_constructor(5,200)
# Simulation Variables
t0 = 0.
dt = 0.001
tf = 1.
inp(t) = 5*ones(200)

figure()
reset!(net)
@time outputs, states = nnsim.simulate!(net, inp, dt, tf, track_flag = true)
PyPlot.plot(t0:dt:tf, transpose(states[1:2:end,2:end]))
xlabel("Time (s)")
ylabel("Membrane Potential (mV)")
title("Membrane Potential of Each Neuron");
gcf()

function test()
    print("yeet")
end
