TITLE gskch.mod  calcium-activated potassium channel (non-voltage-dependent)

: gsk granule

UNITS {
    (molar) = (1/liter)
    (mM)    = (millimolar)
    (mA)	= (milliamp)
    (mV)	= (millivolt)
}

NEURON {
    THREADSAFE
    SUFFIX gskch
    USEION sk READ esk WRITE isk VALENCE 1
    USEION nca READ ncai VALENCE 2
    USEION lca READ lcai VALENCE 2
    USEION tca READ tcai VALENCE 2
    RANGE gsk, gskbar, qinf, qtau, isk
}

PARAMETER {
    celsius=6.3 (degC)
    v=0		(mV)
    gskbar=0  (mho/cm2)
}

STATE { q }

ASSIGNED {
    gsk (mho/cm2)
    qinf
    qtau (ms)
    cai (mM)
}


BREAKPOINT {          :Computes i=g*q^2*(v-esk)
    SOLVE state METHOD cnexp
    gsk = gskbar * q*q
    isk = gsk * (v-esk)
}

INITIAL {
    cai = ncai + lcai + tcai
    rate(cai, celsius)
    q=qinf
}


DERIVATIVE state {  :Computes state variable q at current v and dt.
    cai = ncai + lcai + tcai
    rate(cai, celsius)
    q' = (qinf - q)/qtau
}

PROCEDURE rate(cai, celsius) {  :Computes rate and other constants at current v.
    LOCAL alpha, beta, q10
    q10 = 3^((celsius - 6.3)/10)
    alpha = 1.25e1 * cai * cai
    beta = 0.00025
    qtau = 1 / (alpha + beta)
    qinf = alpha * qtau
}
