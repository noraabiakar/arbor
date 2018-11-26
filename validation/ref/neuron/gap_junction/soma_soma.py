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

class cell:
    def __init__(self, gid, params=None):
        self.soma = None
        self.gid = gid
        self.sections = {}
        self.stims = []
        self.netcons = []

        soma = h.Section(name='soma', cell=self)
        h.celsius = 35
        soma.diam = 20
        soma.L = 25

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

        self.ncomp=1
        self.nseg=1

        self.sections = []
        self.sections.append([soma])

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

    def add_gap_junction(self, other, cm, gm, y, b, sl, xvec, ggap):
        xvec.x[0] = 0.95
        xvec.x[1] = 0.95
        sl.append(sec = self.soma)
        sl.append(sec = other.soma)
        area_src = self.soma.diam*math.pi * self.soma.L
        area_dest = other.soma.diam*math.pi * other.soma.L

        gmat.x[0][0] =  ggap*0.1/area_src
        gmat.x[0][1] = -ggap*0.1/area_src
        gmat.x[1][1] =  ggap*0.1/area_dest
        gmat.x[1][0] = -ggap*0.1/area_dest

        gj = h.LinearMechanism(cm, gm, y, b, sl, xvec)
        return gj

    def add_point_gap(self, other, ggap):
        other.mk_halfgap(self.gid, ggap)
        self.mk_halfgap(other.gid, ggap)

    def mk_halfgap(self, gid_src, ggap):
        pc.source_var(self.soma(.5)._ref_v, self.gid, sec=self.soma)
        hg = h.HalfGap(self.soma(.5))
        pc.target_var(hg, hg._ref_vgap, gid_src)
        hg.g = ggap

    def set_recorder(self):
        """Set soma, dendrite, and time recording vectors on the cell.

        :param cell: Cell to record from.
        :return: the soma, dendrite, and time vectors as a tuple.
        """
        soma_v = h.Vector()   # Membrane potential vector at soma
        t = h.Vector()        # Time stamp vector
        soma_v.record(self.soma(0.5)._ref_v)
        t.record(h._ref_t)
        return soma_v, t

# pc.set_gid2node(soma_cell0.gid, rank)
# nc0 = h.NetCon(soma_cell0.soma(.5)._ref_v, None, sec=soma_cell0.soma)
# pc.cell(soma_cell0.gid, nc0)
#
# pc.set_gid2node(soma_cell1.gid, rank)
# nc1 = h.NetCon(soma_cell1.soma(.5)._ref_v, None, sec=soma_cell1.soma)
# pc.cell(soma_cell1.gid, nc1)

# Add gap junction
# gj = soma_cell0.add_gap_junction(soma_cell1, cmat, gmat, y, b, sl, xvec, 0.760265)
# soma_cell0.add_point_gap(soma_cell1, 0.760265)

# Optionally modify some parameters
# soma_cell1.soma.gbar_nax = 0.015

# Add current stim
# soma_cell0.add_iclamp(0, 100, 0.1)
# soma_cell1.add_iclamp(10, 100, 0.1)

# Run simulation
# data = V.run_nrn_sim(100, report_dt=None, dt=0.025, model='soma')

# print(json.dumps(data))
# V.nrn_stop()

