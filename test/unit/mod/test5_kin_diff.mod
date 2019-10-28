NEURON {
    SUFFIX test5_kin_diff
}

STATE {
    a
}

BREAKPOINT {
    SOLVE state METHOD sparse
}

KINETIC state {
    LOCAL r0
    r0 = 2

    ~ a <-> (0, r0)
}

INITIAL {
    a = 0.2
}
