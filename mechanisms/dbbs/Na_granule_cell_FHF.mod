TITLE Cerebellum Granule Cell Model

COMMENT
basato sul modello di Raman a 13 stati. genera corrente di sodio transiente, persistente e risorgente
with Long-Term Inactivation States L3,L4,L5,L6.

#modified by Stefano Masoli PhD to be the channel it was supposed to be from the beginning.
ENDCOMMENT

NEURON {
	SUFFIX Na_granule_cell_FHF
	USEION na READ ena WRITE ina
	RANGE gnabar, ina, g
	RANGE gamma, delta, epsilon, Con, Coff, Oon, Ooff, Lon, Loff
	RANGE Aalfa, Valfa, Abeta, Vbeta, Ateta, Vteta, Agamma, Adelta, Aepsilon, ACon, ACoff, AOon, AOoff
	RANGE n1,n2,n3,n4
}

UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
}

PARAMETER {
	v (mV)
	celsius = 32  	(degC)
	gnabar = 0.013	(mho/cm2)

	Aalfa = 353.91 ( /ms)
	Valfa = 13.99 ( /mV)
	Abeta = 1.272  ( /ms)
	Vbeta = 13.99 ( /mV)
	Agamma = 150 ( /ms)
	Adelta = 40  ( /ms)
	Aepsilon = 1.75 ( /ms)
	Ateta = 0.0201 ( /ms)
	Vteta = 25

	ACon = 0.025    ( /ms)
	ACoff = 0.5     ( /ms)
	AOon = 0.75     ( /ms)
	AOoff = 0.002   ( /ms)

	n1 = 5.422
	n2 = 3.279
	n3 = 1.83
	n4 = 0.738

	:L3 to L6
	ALon = 0.001   ( /ms)
	ALoff = 0.5    ( /ms) :0.15

	c = 20
	d = 0.075
}

ASSIGNED {
	g (mho/cm2)
	gamma
	delta
	epsilon
	Con
	Coff
	Oon
	Ooff
	Lon
	Loff
	a
	b
	Q10
}

STATE {
	C1
	C2
	C3
	C4
	C5
	O
	OB
	I1
	I2
	I3
	I4
	I5
	I6
	L3
	L4
	L5
	L6
}


INITIAL {
	C1=1
	C2=0
	C3=0
	C4=0
	C5=0
	O=0
	OB=0
	I1=0
	I2=0
	I3=0
	I4=0
	I5=0
	I6=0
	L3=0
	L4=0
	L5=0
	L6=0
	Q10 =3^((celsius-20)/10 )
	gamma = Q10 * Agamma
	delta = Q10 * Adelta
	epsilon = Q10 * Aepsilon
	Con = Q10 * ACon
	Coff = Q10 * ACoff
	Oon = Q10 * AOon
	Ooff = Q10 * AOoff

	Lon = Q10 * ALon
	Loff = Q10 * ALoff
	a = (Oon/Con)^0.25
	b = (Ooff/Coff)^0.25

}

BREAKPOINT {
	SOLVE kstates METHOD sparse
	g = gnabar * O	      	: (mho/cm2)
	ina = g * (v - ena)  	: (mA/cm2)
}


FUNCTION alfa(v(mV))(/ms){
	alfa = Q10*Aalfa*exp(v/Valfa)
}

FUNCTION beta(v(mV))(/ms){
	beta = Q10*Abeta*exp(-v/Vbeta)
}

FUNCTION teta(v(mV))(/ms){
	teta = Q10*Ateta*exp(-v/Vteta)
}


KINETIC kstates {
	LOCAL a0, b0, t0
	a0 = alfa(v)
	b0 = beta(v)
	t0 = teta(v)
	: 1 riga
	~ C1 <-> C2 (n1*a0,n4*b0)
	~ C2 <-> C3 (n2*a0,n3*b0)
	~ C3 <-> C4 (n3*a0,n2*b0)
	~ C4 <-> C5 (n4*a0,n1*b0)
	~ C5 <-> O  (gamma,delta)
	~  O <-> OB (epsilon,t0)

	: 2 riga
	~ I1 <-> I2	(n1*a0*a,n4*b0*b)
	~ I2 <-> I3	(n2*a0*a,n3*b0*b)
	~ I3 <-> I4	(n3*a0*a,n2*b0*b)
	~ I4 <-> I5 (n4*a0*a,n1*b0*b)
	~ I5 <-> I6 (gamma,delta)

	: 3 riga
	~ L3 <-> L4 (n3*a0*c,n2*a0*d)
	~ L4 <-> L5 (n4*a0*c,n1*a0*d)
	~ L5 <-> L6 (gamma,delta)

	: connette 1 riga con 2 riga
	~ C1 <-> I1 (Con,Coff)
	~ C2 <-> I2 (Con*a,Coff*b)
	~ C3 <-> I3 (Con*a^2,Coff*b^2)
	~ C4 <-> I4 (Con*a^3,Coff*b^3)
	~ C5 <-> I5 (Con*a^4,Coff*b^4)
	~  O <-> I6 (Oon,Ooff)

	: connette 1 riga con 3 riga
	~ C3 <-> L3 (Lon,Loff)
	~ C4 <-> L4 (Lon*c,Loff*d)
	~ C5 <-> L5 (Lon*c^2,Loff*d^2)
	~  O <-> L6 (Lon*c^2,Loff*d^2)

	CONSERVE C1+C2+C3+C4+C5+O+OB+I1+I2+I3+I4+I5+I6+L3+L4+L5+L6=1
}
