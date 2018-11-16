module naxChannels

export Stim, run_nax

using Sundials
using Unitful
using Unitful.DefaultSymbols

struct naxParam
    ena       # Na channel reversal potential

    c_m       # membrane spacific capacitance
    gbar    # Na channel cross-membrane conductivity
    sh

    tha
    qa
    Ra
    Rb

    thi1
    thi2
    qd
    qg

    mmin
    hmin
    q10       # temperature dependent rate coefficient
              # (= 3^((T-T₀)/10K) with T₀ = 6.3 °C)
    Rg
    Rd

    thinf
    qinf

    celsius
    # constructor with default values, corresponding
    # to a resting potential of -65 mV and temperature 6.3 °C
    naxParam(;
        # default values from nax.mod
        ena    =  50mV,

        c_m    = 0.018F*m^-2,
        gbar   = 0.04S*cm^-2,
        sh     = 10mV,

        tha    = -30mV,
        qa     = 7.2mV,
        Ra     = 0.4ms^-1,
        Rb     = 0.124ms^-1,

        thi1   = -45mV,
        thi2   = -45mV,
        qd     = 1.5mV,
        qg     =1.5mV,

        mmin   = 0.02ms,
        hmin   = 0.5ms,
        q10    = 2,

        Rg     = 0.01ms^-1,
        Rd     = 0.03ms^-1,

        thinf  = -50mV,
        qinf   = 4mV,

        celsius = 35
    ) = new(ena, c_m, gbar, sh, tha, qa, Ra, Rb, thi1, thi2, qd, qg, mmin, hmin, q10, Rg, Rd, thinf, qinf, celsius)
end

struct Stim
    t0        # start time of stimulus
    t1        # stop time of stimulus
    i_e       # stimulus current density

    Stim() = new(0s, 0s, 0A/m^2)
    Stim(t0, t1, i_e) = new(t0, t1, i_e)
end

scale(quantity, unit) = uconvert(NoUnits, quantity/unit)

trap(v, th, a, q) = a*(v - th) / (1 - exp(-(v - th)/q))

# "m" sodium activation system
function m_lims(v, p)
    qt = p.q10 ^ ((p.celsius - 24) / 10)
    t  = p.tha + p.sh;
    a  = trap(v, t, p.Ra, p.qa)
    b  = trap(-v, -t, p.Rb, p.qa)
    mtau = max(1mV/(a+b)/qt, p.mmin)
    minf = a/(a+b)
    return mtau, minf
end

# "h" sodium inactivation system
function h_lims(v, p)
    qt = p.q10 ^ ((p.celsius - 24) / 10)
    t = p.thi1 + p.sh
    a = trap(v, t, p.Rd, p.qd)
    b = trap(-v, -t, p.Rg, p.qg)
    htau = max(1mV/(a+b)/qt, p.hmin)
    hinf = 1/(1 + exp((v - p.thinf - p.sh)/p.qinf))
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
function f(t, state; p=naxParam(), stim=Stim())
    v, m, h = state

    # calculate current density due to ion channels
    gna = p.gbar*m*m*m*h
    ina = gna*(v - p.ena)
    itot = ina

    # calculate current density due to stimulus
    if t>=stim.t0 && t<stim.t1
        itot -= stim.i_e
    end

    # calculate the voltage dependent rates for the gating variables
    mtau, minf = m_lims(v, p)
    htau, hinf = h_lims(v, p)

    return (-itot/p.c_m, (minf-m)/mtau, (hinf-h)/htau)
end

function run_nax(t_end; v0=-65mV, stim=Stim(), param=naxParam(), sample_dt=0.025ms)
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

end # module naxChannels
