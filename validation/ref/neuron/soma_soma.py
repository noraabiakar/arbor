#!/usr/bin/env python
#coding: utf-8

import json
import math
import sys
import os
import re
import numpy as np
import neuron
import nrn_validation as V
from neuron import h
from builtins import range

default_seg_parameters = {
    'Ra':  100,    # Intracellular resistivity in Ω·cm
    'cm':  1.0,    # Membrane areal capacitance in µF/cm^2
}

default_pas_parameters = {
    'g':         0.001,  # pas conductance in S/cm^2
    'e':         -65,    # mV
}

default_nax_parameters = {
    'gbar':      0.01,   # sodium conductance in S/cm^2
    'sh':        5,      # mV
    'tha':      -30,     # mV
    'qa':        7.2,    # mV
    'Ra':        0.4,    # /ms
    'Rb':        0.124,  # /ms
    'thi1':      -45,    # mV
    'thi2':      -45,    # mV
    'qd':        1.5,    # mV
    'qg':        1.5,    # mV
    'mmin':      0.02,   # ms
    'hmin':      0.5,    # ms
    'q10':       2,      # unitless
    'Rg':        0.01,   # /ms
    'Rd':        0.03,   # /ms
    'thinf':     -50,    # mV
    'qinf':      4,      # mV
    'celsius':   35,     # C
    'ena':       50      # mV
}

default_kamt_parameters = {
    'gbar':      0.002,  # potasium conductance in S/cm^2
    'q10':       3,      # unitless
    'a0m':       0.04,   # /ms
    'vhalfm':    -45,    # mV
    'zetam':     0.1,    # /mV
    'gmm':       0.75,   # unitless
    'a0h':       0.018,  # /ms
    'vhalfh':    -70,    # mV
    'zetah':     0.2,    # /mV
    'gmh':       0.99,   # unitless
    'sha':       9.9,    # mV
    'shi':       5.7,    # mV
    'celsius':   35,     # C
    'ek':        -90     # mV
}

default_kdrmt_parameters = {
    'gbar':      0.002,  # potasium conductance in S/cm^2
    'q10':       3,      # unitless
    'a0m':       0.0035, # /ms
    'vhalfm':    -50,    # mV
    'zetam':     0.055,  # /mV
    'gmm':       0.5,    # unitless
    'celsius':   35,     # C
    'ek':        -90     # mV
}

class cell:
    def __init__(self):
        self.soma = None
        self.sections = {}
        self.stims = []
        self.netcons = []


    def add_iclamp(self, t0, dt, i, to=None, pos=1):
        # If no section specified, attach to middle of soma
        if to is None:
            sec = self.soma
            pos = 0.5
        else:
            sec = self.sections[to]

        stim = h.IClamp(sec(pos))
        stim.delay = t0
        stim.dur = dt
        stim.amp = i
        self.stims.append(stim)

    def add_soma(self, length, diam):
        p_seg  = default_seg_parameters
        p_pas  = default_pas_parameters
        p_nax  = default_nax_parameters
        p_kamt  = default_kamt_parameters
        p_kdrmt  = default_kdrmt_parameters

        soma = h.Section(name='soma')
        h.celsius = 35
        soma.diam = diam
        soma.L = length

        soma.Ra = 100
        soma.cm = 1.8

        # Insert active nax channels in the soma.
        soma.insert('nax')
        soma.gbar_nax = 0.04
        soma.sh_nax = 10

        soma.insert('kamt')
        soma.gbar_kamt = 0.004

        soma.insert('kdrmt')
        soma.gbar_kdrmt = 0.0001

        soma.ena = 50
        soma.ek = -90

        self.soma = soma


soma_cell0 = cell()
soma_cell1 = cell()

soma_cell0.add_soma(25, 20)
soma_cell1.add_soma(25, 20)

cmat = h.Matrix(2,2,2)
gmat = h.Matrix(2,2,2)

y = h.Vector(2)
b = h.Vector(2)
xvec = h.Vector(2)

xvec.x[0] = 0.95
xvec.x[1] = 0.95

sl = h.SectionList()
sl.append(sec = soma_cell0.soma)
sl.append(sec = soma_cell1.soma)

area_src = soma_cell0.soma.diam*math.pi * soma_cell0.soma.L
area_dest = soma_cell1.soma.diam*math.pi * soma_cell1.soma.L

gmat.x[0][0] = 0.760265*0.1/area_src
gmat.x[0][1] = -0.760265*0.1/area_src
gmat.x[1][1] = 0.760265*0.1/area_dest
gmat.x[1][0] = -0.760265*0.1/area_dest

soma_cell1.soma.gbar_nax = 0.015

gj = h.LinearMechanism(cmat, gmat, y, b, sl, xvec)

soma_cell0.add_iclamp(0, 100, 0.1)
soma_cell1.add_iclamp(10, 100, 0.1)


data = V.run_nrn_sim(100, report_dt=None, model='soma')
print(json.dumps(data))
V.nrn_stop()

