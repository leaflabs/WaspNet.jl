"""
    poissonST(l::AbstractVector)

Generates Poisson Spike Trains based on the normalized vector. Each
pseudo-neuron (probability p in vector), fires with probability p at each
timestep of simulation.

# Inputs
- `l`: array of values
"""

function poissonST(l::AbstractVector)
    return poissonST(t) = [Float64(rand(Bernoulli(p), 1)[1]) for p in normalize(l)]
end
