function construct_problem(net::Network, u0, tspan)
    function f!(du, u, p, t)
        update!(net, du, u, t)
    end
    cb = _setup_callbacks(net)
    prob = ODEProblem(f!, u0, tspan, nothing)
    return prob, cb
end

function _setup_callbacks(net::Network, args...; kwargs...)
    aff!(int) = aff_element!(net, int.u, 0., int.t)
    callbacks = []
    for (i,l) in enumerate(net.layers)
        if isa(l, InputMatrixLayer )
            times = l.times
            function aff_net_times!(int)
                net.prev_outputs[i] = @view l.data[:,l.idx]
                net.prev_events[i] = true
                net.layers[i] = InputMatrixLayer(l.data, l.times, l.idx+1)
                aff!(int)
            end
            push!(callbacks, PresetTimeCallback(times, aff_net_times!))
        end
    end
    
    event_cond(u, t, int) = event(net, u, t)
    event_callback = DiscreteCallback(event_cond, aff!, args...; save_positions = (true, true), kwargs...)
    push!(callbacks, event_callback)
    cbset = CallbackSet(callbacks...)
    return cbset
end