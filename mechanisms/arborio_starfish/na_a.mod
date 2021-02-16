TITLE Mod file for component: Component(id=na_a type=ionChannelHH)

COMMENT

    This NEURON file has been generated by org.neuroml.export (see https://github.com/NeuroML/org.neuroml.export)
         org.neuroml.export  v1.7.0
         org.neuroml.model   v1.7.0
         jLEMS               v0.10.2

ENDCOMMENT

NEURON {
    SUFFIX na_a
    USEION na WRITE ina VALENCE 1 ? Assuming valence = 1; TODO check this!!
    
    RANGE gion                           
    RANGE gmax                              : Will be changed when ion channel mechanism placed on cell!
    RANGE conductance                       : parameter
    
    RANGE g                                 : exposure
    
    RANGE fopen                             : exposure
    RANGE m_instances                       : parameter
    RANGE m_SEC                             : parameter
    
    RANGE m_tau                             : exposure
    
    RANGE m_inf                             : exposure
    
    RANGE m_fcond                           : exposure
    
    RANGE m_q                               : exposure
    RANGE m_steadyState_rate                : parameter
    RANGE m_steadyState_midpoint            : parameter
    RANGE m_steadyState_scale               : parameter
    
    RANGE m_steadyState_x                   : exposure
    RANGE h_instances                       : parameter
    
    RANGE h_tau                             : exposure
    
    RANGE h_inf                             : exposure
    
    RANGE h_rateScale                       : exposure
    
    RANGE h_fcond                           : exposure
    RANGE h_timeCourse_tau                  : parameter
    RANGE h_timeCourse_midpoint             : parameter
    RANGE h_timeCourse_scale                : parameter
    
    RANGE h_timeCourse_t                    : exposure
    RANGE h_steadyState_rate                : parameter
    RANGE h_steadyState_midpoint            : parameter
    RANGE h_steadyState_scale               : parameter
    
    RANGE h_steadyState_x                   : exposure
    RANGE h_tauUnscaled                     : derived variable
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
    m_instances = 3 
    m_SEC = 1000 (ms)
    m_steadyState_rate = 1 
    m_steadyState_midpoint = -30 (mV)
    m_steadyState_scale = 5.5 (mV)
    h_instances = 1 
    h_timeCourse_tau = 1.5 (ms)
    h_timeCourse_midpoint = -40 (mV)
    h_timeCourse_scale = -33 (mV)
    h_steadyState_rate = 1 
    h_steadyState_midpoint = -60 (mV)
    h_steadyState_scale = -5.8 (mV)
}

ASSIGNED {
    
    gion   (S/cm2)                          : Transient conductance density of the channel? Standard Assigned variables with ionChannel
    v (mV)
    celsius (degC)
    temperature (K)
    ena (mV)
    
    
    m_steadyState_x                        : derived variable
    
    m_inf                                  : derived variable
    
    m_tau (ms)                             : derived variable
    
    m_q                                    : derived variable
    
    m_fcond                                : derived variable
    
    h_timeCourse_t (ms)                    : derived variable
    
    h_steadyState_x                        : derived variable
    
    h_rateScale                            : derived variable
    
    h_fcond                                : derived variable
    
    h_inf                                  : derived variable
    
    h_tauUnscaled (ms)                     : derived variable
    
    h_tau (ms)                             : derived variable
    
    conductanceScale                       : derived variable
    
    fopen0                                 : derived variable
    
    fopen                                  : derived variable
    
    g (uS)                                 : derived variable
    rate_h_q (/ms)
    
}

STATE {
    h_q  
    
}

INITIAL {
    ena = 55.0
    
    temperature = celsius + 273.15
    
    rates(v)
    rates(v) ? To ensure correct initialisation.
    
    h_q = h_inf
    
}

BREAKPOINT {
    
    SOLVE states METHOD cnexp
    
    ? DerivedVariable is based on path: conductanceScaling[*]/factor, on: Component(id=na_a type=ionChannelHH), from conductanceScaling; null
    ? Path not present in component, using factor: 1
    
    conductanceScale = 1 
    
    ? DerivedVariable is based on path: gates[*]/fcond, on: Component(id=na_a type=ionChannelHH), from gates; Component(id=m type=gateHHInstantaneous)
    ? multiply applied to all instances of fcond in: <gates> ([Component(id=m type=gateHHInstantaneous), Component(id=h type=gateHHtauInf)]))
    fopen0 = m_fcond * h_fcond ? path based, prefix = 
    
    fopen = conductanceScale  *  fopen0 ? evaluable
    g = conductance  *  fopen ? evaluable
    gion = gmax * fopen 
    
    ina = gion * (v - ena)
    
}

DERIVATIVE states {
    rates(v)
    h_q' = rate_h_q 
    
}

PROCEDURE rates(v) {
    
    m_steadyState_x = m_steadyState_rate  / (1 + exp(0 - (v -  m_steadyState_midpoint )/ m_steadyState_scale )) ? evaluable
    ? DerivedVariable is based on path: steadyState/x, on: Component(id=m type=gateHHInstantaneous), from steadyState; Component(id=null type=HHSigmoidVariable)
    m_inf = m_steadyState_x ? path based, prefix = m_
    
    m_tau = 0 *  m_SEC ? evaluable
    m_q = m_inf ? evaluable
    m_fcond = m_q ^ m_instances ? evaluable
    h_timeCourse_t = h_timeCourse_tau  * exp((v -  h_timeCourse_midpoint )/ h_timeCourse_scale ) ? evaluable
    h_steadyState_x = h_steadyState_rate  / (1 + exp(0 - (v -  h_steadyState_midpoint )/ h_steadyState_scale )) ? evaluable
    ? DerivedVariable is based on path: q10Settings[*]/q10, on: Component(id=h type=gateHHtauInf), from q10Settings; null
    ? Path not present in component, using factor: 1
    
    h_rateScale = 1 
    
    h_fcond = h_q ^ h_instances ? evaluable
    ? DerivedVariable is based on path: steadyState/x, on: Component(id=h type=gateHHtauInf), from steadyState; Component(id=null type=HHSigmoidVariable)
    h_inf = h_steadyState_x ? path based, prefix = h_
    
    ? DerivedVariable is based on path: timeCourse/t, on: Component(id=h type=gateHHtauInf), from timeCourse; Component(id=null type=ExpTime)
    h_tauUnscaled = h_timeCourse_t ? path based, prefix = h_
    
    h_tau = h_tauUnscaled  /  h_rateScale ? evaluable
    
     
    
     
    
     
    rate_h_q = ( h_inf  -  h_q ) /  h_tau ? Note units of all quantities used here need to be consistent!
    
     
    
     
    
     
    
}

