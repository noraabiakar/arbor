:    calcium accumulation into a volume of area*depth next to the
:    membrane with a decay (time constant tau) to resting level
:    given by the global calcium variable cai0_ca_ion

NEURON {
    THREADSAFE
    SUFFIX ccanl
    USEION nca READ inca WRITE ncai VALENCE 2
    USEION lca READ ilca WRITE lcai VALENCE 2
    USEION tca READ itca WRITE tcai VALENCE 2
    RANGE caiinf, catau
}

UNITS {
    (mV) = (millivolt)
    (molar) = (1/liter)
    (mM) = (milli/liter)
    (mA) = (milliamp)
    : FARADAY = 96520 (coul)
    : R = 8.3134	(joule/degC)
}

PARAMETER {
    celsius
    depth   = 200 (nm)
    catau   = 9 (ms)
    caiinf  = 50.e-6 (mM)
    cao     = 2 (mM)
}

ASSIGNED {
    cai (mM)
}

STATE {
    ncai (mM)
    lcai (mM)
    tcai (mM)
}

INITIAL {
    ncai = caiinf/3
    lcai = caiinf/3
    tcai = caiinf/3
    cai  = caiinf
}


BREAKPOINT {
    SOLVE integrate METHOD cnexp
}

DERIVATIVE integrate {
    ncai' = -(inca)/depth/96520 * (1e7) + (caiinf/3 - ncai)/catau
    lcai' = -(ilca)/depth/96520 * (1e7) + (caiinf/3 - lcai)/catau
    tcai' = -(itca)/depth/96520 * (1e7) + (caiinf/3 - tcai)/catau
    cai   = ncai + lcai + tcai
}

FUNCTION ktf(celsius) {
    ktf = 1000*8.3134*(celsius + 273.15)/(2*96520)
}
