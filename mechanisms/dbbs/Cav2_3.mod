TITLE Ca R-type channel with medium threshold for activation
: used in distal dendritic regions, together with calH.mod, to help
: the generation of Ca++ spikes in these regions
: uses channel conductance (not permeability)
: written by Yiota Poirazi on 11/13/00 poirazi@LNC.usc.edu
: From car to Cav2_3

NEURON {
		SUFFIX Cav2_3
	  THREADSAFE
	  USEION ca READ eca WRITE ica
	  RANGE gcabar, m, h, g, gmax
	  RANGE inf_0, inf_1, tau_0, tau_1
}

UNITS {
	  (mA) = (milliamp)
	  (mV) = (millivolt)
}

PARAMETER {              : parameters that can be entered when function is called in cell-setup
    v             (mV)
    celsius = 34	(degC)
    gcabar = 0    (mho/cm2) : initialized conductance
}

STATE {	m h }            : unknown activation and inactivation parameters to be solved in the DEs

ASSIGNED {               : parameters needed to solve DE
    inf_0
    inf_1
	  tau_0 (ms)
	  tau_1 (ms)
    g      (mho/cm2)
    gmax   (mho/cm2)
}

BREAKPOINT {
	  SOLVE states METHOD cnexp
    g = gcabar*m*m*m*h
	  ica = g*(v - eca)
    if (g > gmax) {
        gmax = g
    }
}

INITIAL {
    mhn(v)
    m = inf_0
    h = inf_1
    g = gcabar*m*m*m*h
    ica = g*(v - eca) : initial Ca++ current value
    gmax = g
}

DERIVATIVE states {
	  mhn(v)
	  m' =  (inf_0 - m)/tau_0
	  h' =  (inf_1 - h)/tau_1
}

FUNCTION varss(v (mV), i) {
	  if (i==0) {
	      varss = 1 / (1 + exp((v+48.5)/(-3))) : Ca activation
	  }
	  else {
        varss = 1/ (1 + exp((v+53)/(1)))    : Ca inactivation
	  }
}

FUNCTION vartau(v (mV), i) (ms) {
	  if (i==0) {
        vartau = 50  : activation variable time constant
    }
	  else {
        vartau = 5   : inactivation variable time constant
    }
}

PROCEDURE mhn(v (mV)) {
	LOCAL a, b :rest = -70
	tau_0 = vartau(v, 0)
	tau_1 = vartau(v, 1)
	inf_0 = varss(v, 0)
	inf_1 = varss(v, 1)
}
