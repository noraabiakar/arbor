TITLE T-calcium channel From Migliore CA3
: T-type calcium channel


UNITS {
    (mA) = (milliamp)
    (mV) = (millivolt)
    : KTOMV = .0853 (mV/degC)
}

PARAMETER {
    celsius = 6.3	(degC)
    gcatbar=.003 (mho/cm2)
}


NEURON {
    THREADSAFE
    SUFFIX cat
    USEION tca WRITE itca VALENCE 2
    USEION ca READ cai, cao VALENCE 2
    RANGE gcatbar
}

STATE {
    m
    h
}

ASSIGNED {
    v (mV)
    gcat (mho/cm2)
}

INITIAL {
    LOCAL a_m, b_m, a_h, b_h
    a_m = alpha_m(v)
    b_m = beta_m(v)
    a_h = alpha_h(v)
    b_h = beta_h(v)

    m = trap(a_m, b_m)
    h = trap(a_h, b_h)
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    LOCAL nu, f, ghk

    f = KTF(celsius)/2
    nu = v/f
    ghk=-f*(1. - (cai/cao)*exp(nu))*exprelr(nu)

    gcat = gcatbar*m*m*h
    itca = gcat*ghk

}

DERIVATIVE states {	: exact when v held constant
    LOCAL a_m, b_m, a_h, b_h, minf, mtau, hinf, htau

    a_m = alpha_m(v)
    b_m = beta_m(v)
    a_h = alpha_h(v)
    b_h = beta_h(v)

    minf = trap(a_m, b_m)
    mtau = trap2(a_m, b_m)
    hinf = trap(a_h, b_h)
    htau = trap2(a_h, b_h)

    m' = (minf - m)/mtau
    h' = (hinf - h)/htau
}

FUNCTION KTF(celsius) {
    KTF = ((25./293.15)*(celsius + 273.15))
}

FUNCTION alpha_m(v) {
    alpha_m = 0.2*(-1.0*v+19.26)/(exp((-1.0*v+19.26)/10.0)-1.0)
}

FUNCTION beta_m(v) {
    beta_m = 0.009*exp(-v/22.03)
}

FUNCTION alpha_h(v) {
    alpha_h = 1.e-6*exp(-v/16.26)
}

FUNCTION beta_h(v) {
    beta_h = 1/(exp((-v+29.79)/10)+1)
}

FUNCTION trap(a, b) {
    trap = a/(a+b)
}

FUNCTION trap2(a, b) {
    trap2 = 1/(a+b)
}