module spikeChannels

export Stim, run_spike

using Sundials
using Unitful
using Unitful.DefaultSymbols

c_m    = 0.018F*m^-2

struct naxParam
    ena       # Na channel reversal potential

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
    ) = new(ena, gbar, sh, tha, qa, Ra, Rb, thi1, thi2, qd, qg, mmin, hmin, q10, Rg, Rd, thinf, qinf, celsius)
end

struct kamtParam
    ek        # K channel reversal potential

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
    ) = new(ek, gbar, q10, a0m, vhalfm, zetam, gmm, a0h, vhalfh, zetah, gmh, sha, shi, celsius)
end

struct kdrmtParam
    ek        # K channel reversal potential

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

        gbar   = 0.0001S*cm^-2,

        q10    = 3,

        a0m    = 0.0035ms^-1,
        vhalfm = -50mV,
        zetam  = 0.055mV^-1,
        gmm    = 0.5,

        celsius = 35
    ) = new(ek, gbar, q10, a0m, vhalfm, zetam, gmm, celsius)
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
function na_m_lims(v, p)
    qt = p.q10 ^ ((p.celsius - 24) / 10)
    t  = p.tha + p.sh;
    a  = trap(v, t, p.Ra, p.qa)
    b  = trap(-v, -t, p.Rb, p.qa)
    mtau = max(1mV/(a+b)/qt, p.mmin)
    minf = a/(a+b)
    return mtau, minf
end

# "h" sodium inactivation system
function na_h_lims(v, p)
    qt = p.q10 ^ ((p.celsius - 24) / 10)
    t = p.thi1 + p.sh
    a = trap(v, t, p.Rd, p.qd)
    b = trap(-v, -t, p.Rg, p.qg)
    htau = max(1mV/(a+b)/qt, p.hmin)
    hinf = 1/(1 + exp((v - p.thinf - p.sh)/p.qinf))
    return htau, hinf
end

# "m" activation system
function ka_m_lims(v, p)
    qt   = p.q10 ^ ((p.celsius - 24) / 10)
    alpm = exp(p.zetam * (v - p.vhalfm))
    betm = exp(p.zetam * p.gmm * (v - p.vhalfm))

    minf = 1/(1 + exp(-(v - p.sha - 7.6mV) / 14mV))
    mtau = betm / (qt * p.a0m * (1 + alpm))
    return mtau, minf
end

# "h" activation system
function ka_h_lims(v, p)
    qt   = p.q10 ^ ((p.celsius - 24) / 10)
    alph = exp(p.zetah * (v - p.vhalfh))
    beth = exp(p.zetah * p.gmh * (v - p.vhalfh))

    hinf = 1/(1 + exp((v - p.shi + 47.4mV) / 6mV))
    htau = beth / (qt * p.a0h * (1 + alph))
    return htau, hinf
end

# "m" activation system
function kd_m_lims(v, p)
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
function initial_conditions(v, p_na, p_ka, p_kd)
    na_mtau, na_minf = na_m_lims(v, p_na)
    na_htau, na_hinf = na_h_lims(v, p_na)

    ka_mtau, ka_minf = ka_m_lims(v, p_ka)
    ka_htau, ka_hinf = ka_h_lims(v, p_ka)

    kd_mtau, kd_minf = kd_m_lims(v, p_kd)

    return (v, na_minf, na_hinf, ka_minf, ka_hinf, kd_minf)
end

# Given time t and state (v, m, h, n),
# return (vdot, mdot, hdot, ndot)
function f(t, state; p_na=naxParam(), p_ka=kamtParam(), p_kd=kdrmtParam(), stim=Stim())
    v, na_m, na_h, ka_m, ka_h, kd_m = state

    # calculate current density due to ion channels
    gna = p_na.gbar * na_m * na_m * na_m * na_h
    ina = gna*(v - p_na.ena)

    ika = p_ka.gbar * ka_m * ka_h * (v - p_ka.ek)

    ikd = p_kd.gbar * kd_m * (v - p_kd.ek)

    itot = ina + ika + ikd

    # calculate current density due to stimulus
    if t>=stim.t0 && t<stim.t1
        itot -= stim.i_e
    end

    # calculate the voltage dependent rates for the gating variables
    na_mtau, na_minf = na_m_lims(v, p_na)
    na_htau, na_hinf = na_h_lims(v, p_na)

    ka_mtau, ka_minf = ka_m_lims(v, p_ka)
    ka_htau, ka_hinf = ka_h_lims(v, p_ka)

    kd_mtau, kd_minf = kd_m_lims(v, p_kd)
    return (-itot/c_m, (na_minf-na_m)/na_mtau, (na_hinf-na_h)/na_htau, (ka_minf-ka_m)/ka_mtau, (ka_hinf-ka_h)/ka_htau, (kd_minf-kd_m)/kd_mtau)
end

function run_spike(t_end; v0=-65mV, stim=Stim(), na_p=naxParam(), ka_p=kamtParam(), kd_p=kdrmtParam(), sample_dt=0.025ms)
    v_scale = 1V
    t_scale = 1s

    v0, na_m0, na_h0, ka_m0, ka_h0, kd_m0 = initial_conditions(v0, na_p, ka_p, kd_p)
    y0 = [v0/v_scale, na_m0, na_h0, ka_m0, ka_h0, kd_m0]

    samples = collect(0s: sample_dt: t_end)

    fbis(t, y, ydot) = begin
        vdot, na_mdot, na_hdot, ka_mdot, ka_hdot, kd_mdot =
            f(t*t_scale, (y[1]*v_scale, y[2], y[3], y[4], y[5], y[6]), p_na=na_p, p_ka=ka_p, p_kd=kd_p, stim=stim)

        ydot[1], ydot[2], ydot[3], ydot[4], ydot[5], ydot[6] =
            vdot*t_scale/v_scale, na_mdot*t_scale, na_hdot*t_scale, ka_mdot*t_scale, ka_hdot*t_scale, kd_mdot*t_scale

        return Sundials.CV_SUCCESS
    end

    # Ideally would run with vector absolute tolerance to account for v_scale,
    # but this would prevent us using the nice cvode wrapper.

    res = Sundials.cvode(fbis, y0, scale.(samples, t_scale), abstol=1e-6, reltol=5e-10)

    return samples, res[:, 1]*v_scale
end

end # module spikeChannels
