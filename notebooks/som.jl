using SOM
using RDatasets
using MAT
using nnsim
using Parameters # Not necessary, but highly recommended for automating the generation of constructors for types
using PyPlot

vars = matread("/home/charles/Downloads/IR_dataset_93chemicals.mat")

train = vars["ir_absorbance"]
train = copy(transpose(train))

som = initSOM(train, 10, 10, topol = :rectangular)
som = trainSOM(som, train, 1000) # Temporarily reducing to 1000 rounds so as to reduce time
som = trainSOM(som, train, 1000, r = 3.0)

#plotDensity(som)
#gcf()

winners = mapToSOM(som, train[1:93,:])
closeNeurons = winners[:,3]
# need to find way to translate fequency of map nodes to currents to inject

# Lets make them all uniform
# One neuron for each node in the SOM
N = 100
a1 = 0.02
b1 = 0.2
c1 = -65.
d1 = 8.
I1 = zeros(100)
for i in closeNeurons
    I1[i] += 25.
end

# Keep random connectivity within the layer for now
W = randn(N,N)

# Let's start with just two layers
layer1 = batch_layer_construction(nnsim.Izh, W, N, a = a1, b = b1, c = c1, d = d1, I = I1);
layer2 = batch_layer_construction(nnsim.Izh, W, N, a = a1, b = b1, c = c1, d = d1, I = I1);
layer3 = batch_layer_construction(nnsim.Izh, W, N, a = a1, b = b1, c = c1, d = d1, I = I1);
net = Network([layer1, layer2, layer3]);

outputs, states = nnsim.simulate!(net, 2. * ones(N), .001, 5., track_flag = true)
