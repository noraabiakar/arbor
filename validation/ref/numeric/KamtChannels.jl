module kamtChannels

export Stim, run_kamt

using Sundials
using Unitful
using Unitful.DefaultSymbols

struct kamtParam
    ek        # K channel reversal potential

    c_m       # membrane spacific capacitance
    gbar      # Na channel cross-membrane conductivity

    q10       # temperature dependent rate coefficient
              # (= 3^((T-T₀)/10K) with T₀ = 6.3 °C)
    a0m
    vhalfm
    zetam
    gmm

    a0h
    vhalfh
    zetah
    gmh

    sha
    shi

    celsius
    # constructor with default values, corresponding
    # to a resting potential of -65 mV and temperature 35 °C
    kamtParam(;
        # default values from kamt.mod
        ek     =  -90mV,

        c_m    = 0.018F*m^-2,
        gbar   = 0.004S*cm^-2,

        q10    = 3,

        a0m    = 0.04ms^-1,
        vhalfm = -45mV,
        zetam  = 0.1mV^-1,
        gmm    = 0.75,

        a0h    = 0.018ms^-1,
        vhalfh = -70mV,
        zetah  = 0.2mV^-1,
        gmh    = 0.99,

        sha    = 9.9mV,
        shi    = 5.7mV,

        celsius = 35
    ) = new(ek, c_m, gbar, q10, a0m, vhalfm, zetam, gmm, a0h, vhalfh, zetah, gmh, sha, shi, celsius)
end

struct Stim
    t0        # start time of stimulus
    t1        # stop time of stimulus
    i_e       # stimulus current density

    Stim() = new(0s, 0s, 0A/m^2)
    Stim(t0, t1, i_e) = new(t0, t1, i_e)
end

scale(quantity, unit) = uconvert(NoUnits, quantity/unit)

# "m" activation system
function m_lims(v, p)
    qt   = p.q10 ^ ((p.celsius - 24) / 10)
    alpm = exp(p.zetam * (v - p.vhalfm))
    betm = exp(p.zetam * p.gmm * (v - p.vhalfm))

    minf = 1/(1 + exp(-(v - p.sha - 7.6mV) / 14mV))
    mtau = betm / (qt * p.a0m * (1 + alpm))
    return mtau, minf
end

# "h" activation system
function h_lims(v, p)
    qt   = p.q10 ^ ((p.celsius - 24) / 10)
    alph = exp(p.zetah * (v - p.vhalfh))
    beth = exp(p.zetah * p.gmh * (v - p.vhalfh))

    hinf = 1/(1 + exp((v - p.shi + 47.4mV) / 6mV))
    htau = beth / (qt * p.a0h * (1 + alph))
    return htau, hinf
end

# Choose initial conditions for the system such that the gating variables
# are at steady state for the user-specified voltage v
function initial_conditions(v, p)
    mtau, minf = m_lims(v, p)
    htau, hinf = h_lims(v, p)

    return (v, minf, hinf)
end

# Given time t and state (v, m, h, n),
# return (vdot, mdot, hdot, ndot)
function f(t, state; p=kamtParam(), stim=Stim())
    v, m, h = state

    # calculate current density due to ion channels
    ik = p.gbar*m*h*(v - p.ek)
    itot = ik

    # calculate current density due to stimulus
    if t>=stim.t0 && t<stim.t1
        itot -= stim.i_e
    end

    # calculate the voltage dependent rates for the gating variables
    mtau, minf = m_lims(v, p)
    htau, hinf = h_lims(v, p)

    return (-itot/p.c_m, (minf-m)/mtau, (hinf-h)/htau)
end

function run_kamt(t_end; v0=-65mV, stim=Stim(), param=kamtParam(), sample_dt=0.025ms)
    v_scale = 1V
    t_scale = 1s

    v0, m0, h0 = initial_conditions(v0, param)
    y0 = [v0/v_scale, m0, h0]

    samples = collect(0s: sample_dt: t_end)

    fbis(t, y, ydot) = begin
        vdot, mdot, hdot =
            f(t*t_scale, (y[1]*v_scale, y[2], y[3]), stim=stim, p=param)

        ydot[1], ydot[2], ydot[3] =
            vdot*t_scale/v_scale, mdot*t_scale, hdot*t_scale

        return Sundials.CV_SUCCESS
    end

    # Ideally would run with vector absolute tolerance to account for v_scale,
    # but this would prevent us using the nice cvode wrapper.

    res = Sundials.cvode(fbis, y0, scale.(samples, t_scale), abstol=1e-6, reltol=5e-10)

    return samples, res[:, 1]*v_scale
end

end # module kamtChannels
