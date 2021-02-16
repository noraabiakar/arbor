TITLE Mod file for component: Component(id=cah type=ionChannelHH)

COMMENT

    This NEURON file has been generated by org.neuroml.export (see https://github.com/NeuroML/org.neuroml.export)
         org.neuroml.export  v1.7.0
         org.neuroml.model   v1.7.0
         jLEMS               v0.10.2

ENDCOMMENT

NEURON {
    SUFFIX cah
    USEION ca WRITE ica VALENCE 2 ? Assuming valence = 2 (Ca ion); TODO check this!!
    
    RANGE gion                           
    RANGE gmax                              : Will be changed when ion channel mechanism placed on cell!
    RANGE conductance                       : parameter
    
    RANGE g                                 : exposure
    
    RANGE fopen                             : exposure
    RANGE r_instances                       : parameter
    
    RANGE r_alpha                           : exposure
    
    RANGE r_beta                            : exposure
    
    RANGE r_tau                             : exposure
    
    RANGE r_inf                             : exposure
    
    RANGE r_rateScale                       : exposure
    
    RANGE r_fcond                           : exposure
    RANGE r_forwardRate_rate                : parameter
    RANGE r_forwardRate_midpoint            : parameter
    RANGE r_forwardRate_scale               : parameter
    
    RANGE r_forwardRate_r                   : exposure
    RANGE r_reverseRate_rate                : parameter
    RANGE r_reverseRate_midpoint            : parameter
    RANGE r_reverseRate_scale               : parameter
    
    RANGE r_reverseRate_r                   : exposure
    RANGE r_q10Settings_fixedQ10            : parameter
    
    RANGE r_q10Settings_q10                 : exposure
    RANGE r_reverseRate_x                   : derived variable
    RANGE conductanceScale                  : derived variable
    RANGE fopen0                            : derived variable
    
}

UNITS {
    
    (nA) = (nanoamp)
    (uA) = (microamp)
    (mA) = (milliamp)
    (A) = (amp)
    (mV) = (millivolt)
    (mS) = (millisiemens)
    (uS) = (microsiemens)
    (molar) = (1/liter)
    (kHz) = (kilohertz)
    (mM) = (millimolar)
    (um) = (micrometer)
    (umol) = (micromole)
    (S) = (siemens)
    
}

PARAMETER {
    
    gmax = 0  (S/cm2)                       : Will be changed when ion channel mechanism placed on cell!
    
    conductance = 1.0E-5 (uS)
    r_instances = 2 
    r_forwardRate_rate = 1.7 (kHz)
    r_forwardRate_midpoint = 5 (mV)
    r_forwardRate_scale = 13.9 (mV)
    r_reverseRate_rate = 0.1 (kHz)
    r_reverseRate_midpoint = -8.5 (mV)
    r_reverseRate_scale = -5 (mV)
    r_q10Settings_fixedQ10 = 0.2 
}

ASSIGNED {
    
    gion   (S/cm2)                          : Transient conductance density of the channel? Standard Assigned variables with ionChannel
    v (mV)
    celsius (degC)
    temperature (K)
    eca (mV)
    
    r_forwardRate_r (kHz)                  : derived variable
    
    r_reverseRate_x                        : derived variable
    
    r_reverseRate_r (kHz)                  : conditional derived var...
    
    r_q10Settings_q10                      : derived variable
    
    r_rateScale                            : derived variable
    
    r_alpha (kHz)                          : derived variable
    
    r_beta (kHz)                           : derived variable
    
    r_fcond                                : derived variable
    
    r_inf                                  : derived variable
    
    r_tau (ms)                             : derived variable
    
    conductanceScale                       : derived variable
    
    fopen0                                 : derived variable
    
    fopen                                  : derived variable
    
    g (uS)                                 : derived variable
    rate_r_q (/ms)
    
}

STATE {
    r_q  
    
}

INITIAL {
    eca = 120.0
    
    temperature = celsius + 273.15
    
    rates(v)
    rates(v) ? To ensure correct initialisation.
    
    r_q = r_inf
    
}

BREAKPOINT {
    
    SOLVE states METHOD cnexp
    
    ? DerivedVariable is based on path: conductanceScaling[*]/factor, on: Component(id=cah type=ionChannelHH), from conductanceScaling; null
    ? Path not present in component, using factor: 1
    
    conductanceScale = 1 
    
    ? DerivedVariable is based on path: gates[*]/fcond, on: Component(id=cah type=ionChannelHH), from gates; Component(id=r type=gateHHrates)
    ? multiply applied to all instances of fcond in: <gates> ([Component(id=r type=gateHHrates)]))
    fopen0 = r_fcond ? path based, prefix = 
    
    fopen = conductanceScale  *  fopen0 ? evaluable
    g = conductance  *  fopen ? evaluable
    gion = gmax * fopen 
    
    ica = gion * (v - eca)
    
}

DERIVATIVE states {
    rates(v)
    r_q' = rate_r_q 
    
}

PROCEDURE rates(v) {
    
    r_forwardRate_r = r_forwardRate_rate  / (1 + exp(0 - (v -  r_forwardRate_midpoint )/ r_forwardRate_scale )) ? evaluable
    r_reverseRate_x = (v -  r_reverseRate_midpoint ) /  r_reverseRate_scale ? evaluable
    if (r_reverseRate_x  != 0)  { 
        r_reverseRate_r = r_reverseRate_rate  *  r_reverseRate_x  / (1 - exp(0 -  r_reverseRate_x )) ? evaluable cdv
    } else if (r_reverseRate_x  == 0)  { 
        r_reverseRate_r = r_reverseRate_rate ? evaluable cdv
    }
    
    r_q10Settings_q10 = r_q10Settings_fixedQ10 ? evaluable
    ? DerivedVariable is based on path: q10Settings[*]/q10, on: Component(id=r type=gateHHrates), from q10Settings; Component(id=null type=q10Fixed)
    ? multiply applied to all instances of q10 in: <q10Settings> ([Component(id=null type=q10Fixed)]))
    r_rateScale = r_q10Settings_q10 ? path based, prefix = r_
    
    ? DerivedVariable is based on path: forwardRate/r, on: Component(id=r type=gateHHrates), from forwardRate; Component(id=null type=HHSigmoidRate)
    r_alpha = r_forwardRate_r ? path based, prefix = r_
    
    ? DerivedVariable is based on path: reverseRate/r, on: Component(id=r type=gateHHrates), from reverseRate; Component(id=null type=HHExpLinearRate)
    r_beta = r_reverseRate_r ? path based, prefix = r_
    
    r_fcond = r_q ^ r_instances ? evaluable
    r_inf = r_alpha /( r_alpha + r_beta ) ? evaluable
    r_tau = 1/(( r_alpha + r_beta ) *  r_rateScale ) ? evaluable
    
     
    rate_r_q = ( r_inf  -  r_q ) /  r_tau ? Note units of all quantities used here need to be consistent!
    
     
    
     
    
     
    
     
    
}

