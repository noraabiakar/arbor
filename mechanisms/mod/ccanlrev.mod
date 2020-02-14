:    reversal potential calculation for ccanl.

NEURON {
    THREADSAFE
    SUFFIX ccanlrev
    USEION nca READ ncai WRITE enca VALENCE 2
    USEION lca READ lcai WRITE elca VALENCE 2
    USEION tca READ tcai WRITE etca VALENCE 2
}

UNITS {
    (mV) = (millivolt)
    (mM) = (milli/liter)
}

PARAMETER {
    celsius
    caiinf  = 50.e-6 (mM)
    cao     = 2 (mM)
}

ASSIGNED {
    eca (mV)
}

STATE {
}

INITIAL {
    LOCAL eca
    eca = ktf(celsius)* log(cao/caiinf)
    enca = eca
    elca = eca
    etca = eca
}


BREAKPOINT {
    LOCAL eca, cai
    cai   = ncai + lcai + tcai
    eca = ktf(celsius)* log(cao/cai)
    enca  = eca
    elca  = eca
    etca  = eca
}

FUNCTION ktf(celsius) {
    ktf = 1000*8.3134*(celsius + 273.15)/(2*96520)
}
