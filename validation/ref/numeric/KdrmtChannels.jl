module kdrmtChannels

export Stim, run_kdrmt

using Sundials
using Unitful
using Unitful.DefaultSymbols

struct kdrmtParam
    ek        # K channel reversal potential

    c_m       # membrane spacific capacitance
    gbar      # Na channel cross-membrane conductivity

    q10       # temperature dependent rate coefficient
              # (= 3^((T-T₀)/10K) with T₀ = 6.3 °C)
    a0m
    vhalfm
    zetam
    gmm

    celsius
    # constructor with default values, corresponding
    # to a resting potential of -65 mV and temperature 35 °C
    kdrmtParam(;
        # default values from kdrmt.mod
        ek    =  -90mV,

        c_m    = 0.018F*m^-2,
        gbar   = 0.0001S*cm^-2,

        q10    = 3,

        a0m    = 0.0035ms^-1,
        vhalfm = -50mV,
        zetam  = 0.055mV^-1,
        gmm    = 0.5,

        celsius = 35
    ) = new(ek, c_m, gbar, q10, a0m, vhalfm, zetam, gmm, celsius)
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
    minf = 1/(1 + exp(-(v - 21mV)/10mV))
    t    = p.zetam * (v - p.vhalfm)
    alpm = exp(t)
    betm = exp(p.gmm * t)
    mtau = betm / (qt * p.a0m * (1 + alpm))
    return mtau, minf
end

# Choose initial conditions for the system such that the gating variables
# are at steady state for the user-specified voltage v
function initial_conditions(v, p)
    mtau, minf = m_lims(v, p)

    return (v, minf)
end

# Given time t and state (v, m, h, n),
# return (vdot, mdot, hdot, ndot)
function f(t, state; p=kdrmtParam(), stim=Stim())
    v, m = state

    # calculate current density due to ion channels
    ik = p.gbar*m*(v - p.ek)
    itot = ik

    # calculate current density due to stimulus
    if t>=stim.t0 && t<stim.t1
        itot -= stim.i_e
    end

    # calculate the voltage dependent rates for the gating variables
    mtau, minf = m_lims(v, p)

    return (-itot/p.c_m, (minf-m)/mtau)
end

function run_kdrmt(t_end; v0=-65mV, stim=Stim(), param=kdrmtParam(), sample_dt=0.025ms)
    v_scale = 1V
    t_scale = 1s

    v0, m0 = initial_conditions(v0, param)
    y0 = [v0/v_scale, m0]

    samples = collect(0s: sample_dt: t_end)

    fbis(t, y, ydot) = begin
        vdot, mdot =
            f(t*t_scale, (y[1]*v_scale, y[2]), stim=stim, p=param)

        ydot[1], ydot[2] =
            vdot*t_scale/v_scale, mdot*t_scale

        return Sundials.CV_SUCCESS
    end

    # Ideally would run with vector absolute tolerance to account for v_scale,
    # but this would prevent us using the nice cvode wrapper.

    res = Sundials.cvode(fbis, y0, scale.(samples, t_scale), abstol=1e-6, reltol=5e-10)

    return samples, res[:, 1]*v_scale
end

end # module kdrmtChannels
