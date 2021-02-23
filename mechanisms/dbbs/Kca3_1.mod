TITLE Calcium dependent potassium channel
: Implemented in Rubin and Cleland (2006) J Neurophysiology
: Parameters from Bhalla and Bower (1993) J Neurophysiology
: Adapted from /usr/local/neuron/demo/release/nachan.mod - squid
:   by Andrew Davison, The Babraham Institute  [Brain Res Bulletin, 2000]

:Suffix from Kca3 to Kca3_1

NEURON {
  SUFFIX Kca3_1
	USEION k READ ek WRITE ik
	USEION ca READ cai
	RANGE gkbar, ik, Yconcdep, Yvdep
	RANGE Yalpha, Ybeta, tauY, Y_inf
}

UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
	(molar) = (1/liter)
	(mM) = (millimolar)
}

CONSTANT {
	q10 = 3
}

PARAMETER {
  celsius  (degC)
	v (mV)
	dt (ms)
	gkbar= 0.120 (mho/cm2) <0,1e9>
	Ybeta = 0.05 (/ms)
}


STATE {
	Y
}

ASSIGNED {
	Yalpha   (/ms)
	Yvdep
	Yconcdep (/ms)
	tauY (ms)
	Y_inf
	qt
}

INITIAL {
	rate(v,cai)
	Y = Yalpha/(Yalpha + Ybeta)
	qt = q10^((celsius-37)/10)
}

BREAKPOINT {
	SOLVE state METHOD cnexp
	ik = gkbar*Y*(v - ek)
}

DERIVATIVE state {
	rate(v,cai)
	Y' = Yalpha*(1-Y) - Ybeta*Y
}

PROCEDURE rate(v(mV),cai(mM)) {
	vdep(v)
	concdep(cai)
	Yalpha = Yvdep*Yconcdep
	tauY = 1/(Yalpha + Ybeta)
	Y_inf = Yalpha/(Yalpha + Ybeta) /qt
}

PROCEDURE vdep(v(mV)) {
	Yvdep = exp((v+70)/27)
}

PROCEDURE concdep(cai(mM)) {
	if (cai < 0.01) {
		Yconcdep = 500*( 0.015-cai )/(exp((0.015-cai)/0.0013) - 1)
	} else {
		Yconcdep = 500*0.005/(exp(0.005/0.0013) - 1)
	}
}
