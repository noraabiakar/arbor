TITLE Mod file for component: Component(id=BK type=ionChannelHH)

COMMENT

    This NEURON file has been generated by org.neuroml.export (see https://github.com/NeuroML/org.neuroml.export)
         org.neuroml.export  v1.7.0
         org.neuroml.model   v1.7.0
         jLEMS               v0.10.2

ENDCOMMENT

NEURON {
    SUFFIX BK
    USEION ca READ cai,cao VALENCE 2
    USEION k WRITE ik VALENCE 1 ? Assuming valence = 1; TODO check this!!
    
    RANGE gion                           
    RANGE gmax                              : Will be changed when ion channel mechanism placed on cell!
    RANGE conductance                       : parameter
    
    RANGE g                                 : exposure
    
    RANGE fopen                             : exposure
    RANGE c_instances                       : parameter
    
    RANGE c_alpha                           : exposure
    
    RANGE c_beta                            : exposure
    
    RANGE c_tau                             : exposure
    
    RANGE c_inf                             : exposure
    
    RANGE c_rateScale                       : exposure
    
    RANGE c_fcond                           : exposure
    RANGE c_forwardRate_TIME_SCALE          : parameter
    RANGE c_forwardRate_VOLT_SCALE          : parameter
    RANGE c_forwardRate_CONC_SCALE          : parameter
    
    RANGE c_forwardRate_r                   : exposure
    RANGE c_reverseRate_TIME_SCALE          : parameter
    RANGE c_reverseRate_VOLT_SCALE          : parameter
    RANGE c_reverseRate_CONC_SCALE          : parameter
    
    RANGE c_reverseRate_r                   : exposure
    RANGE c_q10Settings_q10Factor           : parameter
    RANGE c_q10Settings_experimentalTemp    : parameter
    RANGE c_q10Settings_TENDEGREES          : parameter
    
    RANGE c_q10Settings_q10                 : exposure
    RANGE c_forwardRate_V                   : derived variable
    RANGE c_forwardRate_ca_conc             : derived variable
    RANGE c_reverseRate_V                   : derived variable
    RANGE c_reverseRate_ca_conc             : derived variable
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
    c_instances = 1 
    c_forwardRate_TIME_SCALE = 1 (ms)
    c_forwardRate_VOLT_SCALE = 1 (mV)
    c_forwardRate_CONC_SCALE = 1000000 (mM)
    c_reverseRate_TIME_SCALE = 1 (ms)
    c_reverseRate_VOLT_SCALE = 1 (mV)
    c_reverseRate_CONC_SCALE = 1000000 (mM)
    c_q10Settings_q10Factor = 3 
    c_q10Settings_experimentalTemp = 303.15 (K)
    c_q10Settings_TENDEGREES = 10 (K)
}

ASSIGNED {
    
    gion   (S/cm2)                          : Transient conductance density of the channel? Standard Assigned variables with ionChannel
    v (mV)
    celsius (degC)
    temperature (K)
    ek (mV)
    
    c_forwardRate_V                        : derived variable
    
    c_forwardRate_ca_conc                  : derived variable
    
    c_forwardRate_r (kHz)                  : derived variable
    
    c_reverseRate_V                        : derived variable
    
    c_reverseRate_ca_conc                  : derived variable
    
    c_reverseRate_r (kHz)                  : derived variable
    
    c_q10Settings_q10                      : derived variable
    
    c_rateScale                            : derived variable
    
    c_alpha (kHz)                          : derived variable
    
    c_beta (kHz)                           : derived variable
    
    c_fcond                                : derived variable
    
    c_inf                                  : derived variable
    
    c_tau (ms)                             : derived variable
    
    conductanceScale                       : derived variable
    
    fopen0                                 : derived variable
    
    fopen                                  : derived variable
    
    g (uS)                                 : derived variable
    rate_c_q (/ms)
    
}

STATE {
    c_q  
    
}

INITIAL {
    ek = -88.0
    
    temperature = celsius + 273.15
    
    rates(v, cai)
    rates(v, cai) ? To ensure correct initialisation.
    
    c_q = c_inf
    
}

BREAKPOINT {
    
    SOLVE states METHOD cnexp
    
    ? DerivedVariable is based on path: conductanceScaling[*]/factor, on: Component(id=BK type=ionChannelHH), from conductanceScaling; null
    ? Path not present in component, using factor: 1
    
    conductanceScale = 1 
    
    ? DerivedVariable is based on path: gates[*]/fcond, on: Component(id=BK type=ionChannelHH), from gates; Component(id=c type=gateHHrates)
    ? multiply applied to all instances of fcond in: <gates> ([Component(id=c type=gateHHrates)]))
    fopen0 = c_fcond ? path based, prefix = 
    
    fopen = conductanceScale  *  fopen0 ? evaluable
    g = conductance  *  fopen ? evaluable
    gion = gmax * fopen 
    
    ik = gion * (v - ek)
    
}

DERIVATIVE states {
    rates(v, cai)
    c_q' = rate_c_q 
    
}

PROCEDURE rates(v, cai) {
    LOCAL caConc
    
    caConc = cai
    
    c_forwardRate_V = v /  c_forwardRate_VOLT_SCALE ? evaluable
    c_forwardRate_ca_conc = caConc /  c_forwardRate_CONC_SCALE ? evaluable
    c_forwardRate_r = (7/(1 + (0.0015 * (exp ( c_forwardRate_V /-11.765))/( c_forwardRate_ca_conc  * 1e6)))) /  c_forwardRate_TIME_SCALE ? evaluable
    c_reverseRate_V = v /  c_reverseRate_VOLT_SCALE ? evaluable
    c_reverseRate_ca_conc = caConc /  c_reverseRate_CONC_SCALE ? evaluable
    c_reverseRate_r = (1/(1 + ( c_reverseRate_ca_conc  * 1e6)/(0.00015* (exp ( c_reverseRate_V /-11.765)) ))) /  c_reverseRate_TIME_SCALE ? evaluable
    c_q10Settings_q10 = c_q10Settings_q10Factor ^((temperature -  c_q10Settings_experimentalTemp )/ c_q10Settings_TENDEGREES ) ? evaluable
    ? DerivedVariable is based on path: q10Settings[*]/q10, on: Component(id=c type=gateHHrates), from q10Settings; Component(id=null type=q10ExpTemp)
    ? multiply applied to all instances of q10 in: <q10Settings> ([Component(id=null type=q10ExpTemp)]))
    c_rateScale = c_q10Settings_q10 ? path based, prefix = c_
    
    ? DerivedVariable is based on path: forwardRate/r, on: Component(id=c type=gateHHrates), from forwardRate; Component(id=null type=Golgi_KC_c_alpha_rate)
    c_alpha = c_forwardRate_r ? path based, prefix = c_
    
    ? DerivedVariable is based on path: reverseRate/r, on: Component(id=c type=gateHHrates), from reverseRate; Component(id=null type=Golgi_KC_c_beta_rate)
    c_beta = c_reverseRate_r ? path based, prefix = c_
    
    c_fcond = c_q ^ c_instances ? evaluable
    c_inf = c_alpha /( c_alpha + c_beta ) ? evaluable
    c_tau = 1/(( c_alpha + c_beta ) *  c_rateScale ) ? evaluable
    
     
    
     
    
     
    
     
    
     
    
     
    
     
    
     
    
     
    
     
    
     
    
     
    rate_c_q = ( c_inf  -  c_q ) /  c_tau ? Note units of all quantities used here need to be consistent!
    
     
    
     
    
     
    
     
    
}

