"""
	update!(ne::NetworkElement, input, dt, t)

Evolve the `NetworkElement` subject to an input `input` a duration of time `dt` starting from time `t`.
"""
function update!(ne::NetworkElement, input, dt, t) end

"""
	reset!(ne::NetworkElement)

Reset the `NetworkElement` to a default state defined by the method.
"""
function reset!(ne::NetworkElement) end