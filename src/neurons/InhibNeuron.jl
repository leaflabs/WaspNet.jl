struct InhibNeuron{N<:AbstractNeuron} <: AbstractNeuron
    inner_neuron::N
end

function update(neuron::InhibNeuron, input_update, dt, t)
    inner_output, return_neuron = update(neuron.inner_neuron, input_update, dt, t)

    return (-1.0*inner_output, InhibNeuron(return_neuron))
end

function get_neuron_outputs(n::InhibNeuron)
    return get_neuron_outputs(n.inner_neuron)
end

function get_neuron_states(n::InhibNeuron)
    return get_neuron_states(n.inner_neuron)
end

function get_neuron_count(n::InhibNeuron)
    return get_neuron_count(n.inner_neuron)
end
