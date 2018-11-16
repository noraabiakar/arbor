#!/usr/bin/env julia

include("NaX_Kamt_Krdmt_Channels.jl")

using JSON
using Unitful
using Unitful.DefaultSymbols
using Main.spikeChannels

scale(quantity, unit) = uconvert(NoUnits, quantity/unit)

radius = 22µm/2
area = 4*pi*radius^2

stim = []
push!(stim, Stim(0ms, 100ms, 0.1nA/area))
push!(stim, Stim(10ms, 100ms, 0.1nA/area))
ts, vs = run_spike(100ms, stim=stim, sample_dt=0.025ms)

trace = Dict(
    :name => "membrane voltage",
    :sim => "numeric",
    :model => "soma",
    :units => "mV",
    :data => Dict(
        :time => scale.(ts, 1ms),
        Symbol("soma.mid") => scale.(vs[1], 1mV)
    )
)


trace[:data][Symbol("soma.mid"*string(2))] = scale.(vs[2], 1mV)

println(JSON.json([trace]))

