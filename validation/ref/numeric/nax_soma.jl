#!/usr/bin/env julia

include("NaXChannels_multipleSomas.jl")

using JSON
using Unitful
using Unitful.DefaultSymbols
using Main.naxChannels

scale(quantity, unit) = uconvert(NoUnits, quantity/unit)

radius = 22µm/2
area = 4*pi*radius^2

stim = Stim(10ms, 300ms, 0.1nA/area)
ts, vs = run_nax(300ms, stim=stim, sample_dt=0.025ms)

trace = Dict(
    :name => "membrane voltage",
    :sim => "numeric",
    :model => "soma",
    :units => "mV",
    :data => Dict(
        :time => scale.(ts, 1ms),
        Symbol("soma.mid") => scale.(vs, 1V)
    )
)

println(JSON.json([trace]))

