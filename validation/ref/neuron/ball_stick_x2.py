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

    def add_dendrite(self, name, geom, to=None):
        p_pas  = default_pas_parameters

        dend = h.Section(name=name)
        dend.push()
        for x, d in geom:
            h.pt3dadd(x, 0, 0, d)
        h.pop_section()

        dend.Ra = 100
        dend.cm = 1.8

        # Add passive membrane properties to dendrite.
        dend.insert('pas')
        dend.g_pas = p_pas['g']
        dend.e_pas = p_pas['e']

        dend.nseg = 5

        if to is None:
            if self.soma is not None:
                dend.connect(self.soma(1))
        else:
            dend.connect(self.sections[to](1))

        self.sections[name] = dend

    def add_gap_junction(self, other, cm, gm, y, b, sl, xvec, ggap, src=None, dest=None):
        xvec.x[0] = 0.95
        xvec.x[1] = 0.95

        if src is None:
            sl.append(sec = self.soma)
            area_src = self.soma.diam*math.pi * self.soma.L
        else:
            sl.append(sec = self.sections[src])
            area_src = self.sections[src].diam*math.pi * self.sections[src].L

        if dest is None:
            sl.append(sec = other.soma)
            area_dest = other.soma.diam*math.pi * other.soma.L
        else:
            sl.append(sec = other.sections[dest])
            area_dest = other.sections[dest].diam*math.pi * other.sections[dest].L

        gmat.x[0][0] =  ggap*0.1/area_src
        gmat.x[0][1] = -ggap*0.1/area_src
        gmat.x[1][1] =  ggap*0.1/area_dest
        gmat.x[1][0] = -ggap*0.1/area_dest

        gj = h.LinearMechanism(cm, gm, y, b, sl, xvec)
        return gj

# Linear mechanism state, needs to be part of the "state"
cmat = h.Matrix(2,2,2)
gmat = h.Matrix(2,2,2)
y = h.Vector(2)
b = h.Vector(2)
xvec = h.Vector(2)
sl = h.SectionList()

# Create 2 soma cells
cell0 = cell()
cell1 = cell()

# Add somas
cell0.add_soma(25, 20)
cell1.add_soma(25, 20)

# Add dendrite
# Dendrite geometry: all 100 µm long, 1 µm diameter.
geom = [(0,1), (100, 1)]
cell0.add_dendrite('stick0', geom)
cell1.add_dendrite('stick1', geom)

# Add gap junction
# gj = cell0.add_gap_junction(cell1, cmat, gmat, y, b, sl, xvec, 5, 'stick0', 'stick1')

# Optionally modify some parameters
# cell1.soma.gbar_nax = 0.015

# Add current stim
cell0.add_iclamp(0, 100, 0.1)
cell1.add_iclamp(10, 100, 0.1)

# Run simulation
data = V.run_nrn_sim(100, report_dt=None, model='soma')

print(json.dumps(data))
V.nrn_stop()

