TITLE ichan2.mod  
 
: konduktivitas valtozas hatasa- somaban 
 
UNITS {
    (mA) =(milliamp)
    (mV) =(millivolt)
    (uF) = (microfarad)
    (molar) = (1/liter)
    (nA) = (nanoamp)
    (mM) = (millimolar)
    (um) = (micron)
}
 
: interface 
NEURON { 
    THREADSAFE
    SUFFIX ichan2
    USEION nat READ enat WRITE inat VALENCE 1
    USEION kf READ ekf WRITE ikf  VALENCE 1
    USEION ks READ eks WRITE iks  VALENCE 1

    NONSPECIFIC_CURRENT il
    RANGE  gnat,    gkf,    gks
    RANGE  gnatbar, gkfbar, gksbar
    RANGE  gl, el
    RANGE  minf, mtau, hinf, htau, nfinf, nftau, nsinf, nstau
    GLOBAL q10
}
 
PARAMETER {
    v       (mV)
    gnatbar=0 (mho/cm2)
    gkfbar=0  (mho/cm2)
    gksbar=0  (mho/cm2)
    gl=0      (mho/cm2)
    el=0      (mV)

    celsius = 6.3 (degC)
}
 
STATE {
    m h nf ns
}
 
ASSIGNED {
    gnat (mho/cm2)
    gkf  (mho/cm2)
    gks  (mho/cm2)

    minf
    hinf
    nfinf
    nsinf

    mtau  (ms)
    htau  (ms)
    nftau (ms)
    nstau (ms)

    q10
} 

: currents
BREAKPOINT {
    SOLVE states METHOD cnexp

    gnat = gnatbar*m*m*m*h
    inat = gnat*(v - enat)

    gkf = gkfbar*nf*nf*nf*nf
    ikf = gkf*(v-ekf)

    gks = gksbar*ns*ns*ns*ns
    iks = gks*(v-eks)

    il = gl*(v-el)
}
 
INITIAL {
    trates(v, celsius)
    m  = minf
    h  = hinf
    nf = nfinf
    ns = nsinf
}

: states
DERIVATIVE states {
    trates(v, celsius)
    m'  = (minf - m)/mtau
    h'  = (hinf - h)/htau
    nf' = (nfinf - nf)/nftau
    ns' = (nsinf - ns)/nstau
}

: rates
PROCEDURE rates(v, celsius) {
    LOCAL  alpha, beta, sum

    q10   = 3^((celsius - 6.3)/10)

    alpha = -0.3 * vtrap((v+60-17),-5)
    beta  =  0.3 * vtrap((v+60-45), 5)
    sum   = alpha + beta
    mtau  = 1 / sum
    minf  = alpha / sum

    alpha = 0.23 / exp((v+60+5)/20)
    beta  = 3.33 / (1 + exp((v+60-47.5)/-10))
    sum   = alpha + beta
    htau  = 1 / sum
    hinf  = alpha / sum

    alpha = -0.028 * vtrap((v+65-35),-6)
    beta  = 0.1056 / exp((v+65-10)/40)
    sum   = alpha + beta
    nstau = 1 / sum
    nsinf = alpha / sum

    alpha = -0.07 * vtrap((v+65-47),-6)
    beta  = 0.264 / exp((v+65-22)/40)
    sum   = alpha + beta
    nftau = 1 / sum
    nfinf = alpha / sum

}
 
PROCEDURE trates(v, celsius) {
    rates(v, celsius)
}
 
FUNCTION vtrap(x,y) {  :Traps for 0 in denominator of rate eqns.
    vtrap = y*exprelr(x/y)
}
