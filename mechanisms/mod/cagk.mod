TITLE CaGk
: Calcium activated K channel.
: Modified from Moczydlowski and Latorre (1983) J. Gen. Physiol. 82

UNITS {
    (molar) = (1/liter)
    (mV) =	(millivolt)
    (mA) =	(milliamp)
    (mM) =	(millimolar)
}


NEURON {
    THREADSAFE
    SUFFIX cagk
    USEION nca READ ncai VALENCE 2
    USEION lca READ lcai VALENCE 2
    USEION tca READ tcai VALENCE 2
    USEION k READ ek WRITE ik
    RANGE gkbar, gkca, oinf, otau, cai, v
}

UNITS {
    :FARADAY = (faraday)  (kilocoulombs) : Doesn't translate to GPU - use 96.485309 instead
    :R = 8.313424 (joule/degC)
}

PARAMETER {
    celsius		(degC)
    gkbar=.01	(mho/cm2)	: Maximum Permeability

    d1 = .84
    d2 = 1.
    k1 = .48e-3	(mM)
    k2 = .13e-6	(mM)
    abar = .28	(/ms)
    bbar = .48	(/ms)
    st=1        (1)
}

ASSIGNED {
    oinf
    otau (ms)
    gkca (mho/cm2)
    cai  (mM)
}

INITIAL {
    cai = ncai + lcai + tcai
    rate(v, cai, celsius)
    o = oinf
}

STATE {
    o	: fraction of open channels
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    gkca = gkbar*o^st
    ik = gkca*(v - ek)
}

DERIVATIVE states {	: exact when v held constant; integrates over dt step
    cai = ncai + lcai + tcai
    rate(v, cai, celsius)
    o' = (oinf - o)/otau
}

FUNCTION exp1(k, d, v, celsius) {
    exp1 = k*exp(-2*d*96.485309*v/8.313424/(273.15 + celsius))
}

PROCEDURE rate(v, c, celsius) {
    LOCAL x, y, z
    x = c*abar/(c + exp1(k1, d1, v, celsius))
    y = c/exp1(k2, d2, v, celsius)
    z = bbar/(1 + y)
    otau = 1/(x + z)
    oinf = x*otau
}

