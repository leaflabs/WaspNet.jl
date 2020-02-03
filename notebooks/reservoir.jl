using SOM
using RDatasets
using MAT
using nnsim
using Parameters # Not necessary, but highly recommended for automating the generation of constructors for types
using PyPlot
using StatsBase
using LinearAlgebra
using Plots

vars = matread("/home/charles/Downloads/IR_dataset_93chemicals.mat")

train = vars["ir_absorbance"]
o = vars["chemical_id"][:,1]
# Normalize the Data
# dt = fit(ZScoreTransform, train)
# train = StatsBase.transform(dt,train)
train = copy(transpose(train))

som = initSOM(train, 10, 10, topol = :rectangular)
som = trainSOM(som, train, 10000) # Temporarily reducing to 1000 rounds so as to reduce time
som = trainSOM(som, train, 10000, r = 3.0)

W_som = som.codes
#
b = train[66,:]
# c = zeros(100)
# for i in 1:100
#     c[i] = norm(b-a[i,:])
#     if c[i] > 20.
#         c[i] = c[i]/2.
#     end
# end
# c = reshape(c,10,10) #.*(-1.)
# heatmap(c, color=:viridis)
#
# gcf()

labels = ["Hexanol" "1-Octanol" "isoValeric Acid"]
Plots.plot(train[:,[43,66,90]], label=labels, xaxis="Wavenumber (cm-1)", yaxis="IR Absorption Spectra ") # Hexanal, 1-Octonal, isoValeric Acid in that order

winners = mapToSOM(som, train[1:93,:])
closeNeurons = winners[1:10,3]
# need to find way to translate fequency of map nodes to currents to inject

# Lets make them all uniform
# One neuron for each node in the SOM
N = 100
a1 = 0.02
b1 = 0.2
c1 = -65.
d1 = 8.
recurrent = true
# I1 = zeros(100)
# for i in closeNeurons
#     I1[i] += 25.
# end

function encoding(W, O)
    current = zeros(100)
    aux = W*O
    act = maximum(aux) ./ aux
    # for i in length(act)
    #     current[i] = act[i]
    # end
    return act
end

I1 = encoding(W_som, b)

# Keep random connectivity within the layer for now
W = randn(N,N)
# Let's start with just two layers
RNN = batch_layer_construction(nnsim.Izh, W, N; recurrent = true, a = a1, b = b1, c = c1, d = d1, I = I1);
net = Network([RNN]);

# Simulation Variables

t0 = 0.
dt = 0.001
tf = 5.

figure()
reset!(net)
outputs, states = nnsim.simulate!(net, zeros(N), dt, tf, track_flag = true)
PyPlot.plot(t0:dt:tf, transpose(states[1:2:end,2:end]))
xlabel("Time (s)")
ylabel("Membrane Potential (mV)")
title("Membrane Potential of Each Neuron");
gcf()
