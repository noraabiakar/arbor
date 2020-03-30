NEURON {
    SUFFIX test2_kin_diff
}

STATE {
    a b c
}

BREAKPOINT {
    SOLVE state METHOD sparse
}

KINETIC state {
    LOCAL f, r
    f = 2.0
    r = 1.0

    ~ 2a  + b <-> c (f, r)
}

INITIAL {
    a = 0.2
    b = 0.3
    c = 0.5
}
