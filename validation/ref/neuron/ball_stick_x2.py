#!/usr/bin/env python
#coding: utf-8

import json
import math
import sys
import os
import re
import numpy as np
import neuron
from neuron import h
from builtins import range

# Run 'current' model, return list of traces.
# Samples at cable mid- and end-points taken every `sample_dt`;
# Voltage on all compartments per section reported every `report_dt`.
def hoc_execute_quiet(arg):
    with open(os.devnull, 'wb') as null:
        fd = sys.stdout.fileno()
        keep = os.dup(fd)
        sys.stdout.flush()
        os.dup2(null.fileno(), fd)
        h(arg)
        sys.stdout.flush()
        os.dup2(keep, fd)

def hoc_setup():
    hoc_execute_quiet('load_file("stdrun.hoc")')

def hoc_quit():
    hoc_execute_quiet('quit()')

def combine(*dicts, **kw):
    r = {}
    for d in dicts:
        r.update(d)
    r.update(kw)
    return r

def run_nrn_sim(tend, sample_dt=0.025, report_t=None, report_dt=None, dt=None, **meta):
    if dt is None:
        dt = 0.0

    # Instrument mid-point and ends of each section for traces.
    vtraces = []
    vtrace_t_hoc = h.Vector()

    ncomps = set([s.nseg for s in h.allsec() if s.name()!='soma'])
    if len(ncomps)==1:
        common_ncomp = { 'ncomp': ncomps.pop() }
    else:
        common_ncomp = {}

    i = 0
    for s in h.allsec():
        vend = h.Vector()
        vend.record(s(0.5)._ref_v, sample_dt)
        vtraces.append((s.name()+str(i)+".mid", vend))
        i = i + 1
        if s.nseg!=1 or s.name()!='soma':
            vmid = h.Vector()
            vmid.record(s(1.0)._ref_v, sample_dt)
            vtraces.append((s.name()+".end", vmid))

    vtrace_t_hoc.record(h._ref_t, sample_dt)

    # Instrument every segment for section voltage reports.
    if report_t is None:
        if report_dt is not None:
            report_t = [report_dt*(1+i) for i in range(int(tend/report_dt))]
        else:
            report_t = []
    elif not isinstance(report_t, list):
        report_t = [report_t]

    vreports = []
    vreport_t_hoc = h.Vector(report_t)

    if report_t:
        for s in h.allsec():
            nseg = s.nseg;
            ps = [0] + [(i+0.5)/nseg for i in range(nseg)] + [1]
            vs = [h.Vector() for p in ps]
            for p, v in zip(ps, vs):
                v.record(s(p)._ref_v, vreport_t_hoc)
            vreports.append((s.name(), s.L, s.nseg, ps, vs))

    # Run sim
    if dt==0:
        # Use CVODE instead
        h.cvode.active(1)
        abstol = 1e-6
        h.cvode.atol(abstol)
        common_meta = { 'dt': 0, 'cvode': True, 'abstol': abstol }
    else:
        h.dt = dt
        h.steps_per_ms = 1/dt # or else NEURON might noisily fudge dt
        common_meta = { 'dt': dt, 'cvode': False }

    h.secondorder = 0
    h.tstop = tend
    h.run()

    # convert results to traces with metadata
    traces = []

    vtrace_t = list(vtrace_t_hoc)
    traces.append(combine(common_meta, meta, common_ncomp, {
        'name':  'membrane voltage',
        'sim':   'neuron',
        'units': 'mV',
        'data':  combine({n: list(v) for n, v in vtraces}, time=vtrace_t)
    }))

    # and section reports too
    vreport_t = list(vreport_t_hoc)
    for name, length, nseg, ps, vs in vreports:
        obs = np.column_stack([np.array(v) for v in vs])
        xs = [length*p for p in ps]
        for i, t in enumerate(report_t):
            if i>=obs.shape[0]:
                break

            traces.append(combine(common_meta, meta, {
                'name': 'membrane voltage',
                'sim':  'neuron',
                'units': {'x': 'µm', name: 'mV'},
                'ncomp': nseg,
                'time': t,
                'data': {
                    'x': xs,
                    name: list(obs[i,:])
                }
            }))

    return traces

def nrn_assert_no_sections():
    for s in h.allsec():
        assert False, 'a section exists'

def nrn_stop():
    hoc_quit()


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
        dend.g_pas = 0.001
        dend.e_pas = -65

        dend.nseg = 200
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
            area_src = h.area(0.95, sec = self.soma)
        else:
            sl.append(sec = self.sections[src])
            area_src = h.area(0.95, sec = self.sections[src])

        if dest is None:
            sl.append(sec = other.soma)
            area_dest = h.area(0.95, sec = other.soma)
        else:
            sl.append(sec = other.sections[dest])
            area_dest = h.area(0.95, sec = other.sections[dest])

        gmat.x[0][0] =  ggap*0.1/area_src
        gmat.x[0][1] = -ggap*0.1/area_src
        gmat.x[1][1] =  ggap*0.1/area_dest
        gmat.x[1][0] = -ggap*0.1/area_dest

        gj = h.LinearMechanism(cm, gm, y, b, sl, xvec)
        return gj

hoc_setup()

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
gj = cell0.add_gap_junction(cell1, cmat, gmat, y, b, sl, xvec, 0.760265, 'stick0', 'stick1')

# Optionally modify some parameters
cell1.soma.gbar_nax = 0.015

# Add current stim
cell0.add_iclamp(0, 100, 0.1)
cell1.add_iclamp(10, 100, 0.1)

# Run simulation
data = run_nrn_sim(100, report_dt=None, model='soma')

print(json.dumps(data))
nrn_stop()

