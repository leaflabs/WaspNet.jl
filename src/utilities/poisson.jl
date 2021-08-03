"""
    poissonST(l::AbstractVector, d)

Generates Poisson Spike Trains based on the normalized vector. Each
pseudo-neuron (probability p in vector), fires with probability p at each
timestep of simulation.

# Inputs
- `l`: array of values
- `d`: distribution to generate spike train (ST) from `l`
"""


function getPoissonST(l::AbstractVector, d)
    norm = normalize(l)
    ds = product_distribution(d.(norm))
    sample = rand(ds)
    return (_) -> rand!(ds, sample)
end
