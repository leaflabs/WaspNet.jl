"""
    poissonST(l::AbstractVector)

Generates Poisson Spike Trains based on the normalized vector. Each
pseudo-neuron (probability p in vector), fires with probability p at each
timestep of simulation.

# Inputs
- `l`: array of values
"""

function getPoissonST_old(l::AbstractVector)
    return (_) -> rand.(Bernoulli.(normalize(l)))
end

function getPoissonST(l::AbstractVector, d)
    norm = normalize(l)
    return (_)->rand.(d.(norm))
end
