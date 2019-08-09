NEURON {
    SUFFIX test_linear
}

STATE {
    s d h
}

PARAMETER {
    a4 = 7.3
}

ASSIGNED {
    a0 : = 2.5
    a1 : = 0.5
    a2 : = 3
    a3 : = 2.3
}

BREAKPOINT {
    SOLVE state
}

LINEAR state {
    ~                (a4 - a3)*d -    a2*h = 0
    ~ (a0 + a1)*s - (-a1 + a0)*d           = 0
    ~           s +            d +       h = 1
}
