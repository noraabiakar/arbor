#!/usr/bin/env python
#coding: utf-8

import math
from neuron import h
from builtins import range

# cell is composed of a single soma with nax, kamt and kdrmt mechanisms
class cell:
    def __init__(self, gid, params=None):
        self.pc = h.ParallelContext()
        self.soma = None
        self.gid = gid
        self.sections = {}
        self.stims = []
        self.halfgap_list = []

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

    # Add a gap_junction between self and other
    # the voltages of 'self' and 'other' need to be visible to the other cell's half gap junction
    # 'source_var' assigns a voltage variable to a unique id
    # 'target_var' attaches a voltage variable (identified using its unique id) to another voltage variable
    # to expose the voltage of 'self' to the half gap_junction at 'other':
    # 1. assign the voltage of the soma on 'self' to a unique id (cell gid) using 'source_var'
    # 2. attach the voltage of the half gap_junction at 'other' to the voltage of the soma of 'self'
    #    using 'target_var' and the unique id (gid) of the soma of 'self'
    def add_point_gap(self, other, ggap):  #ggap in nS
        if self.pc.gid_exists(self.gid):
            self.mk_halfgap(other, ggap)

        if self.pc.gid_exists(other.gid):
            other.mk_halfgap(self, ggap)

    # assign the voltage at the soma to the gid of the cell
    # create half gap_junction at the soma, and assign its variables: vgap and g
    # vgap gets the voltage assigned to the gid of the 'other' cell
    # g gets ggap
    def mk_halfgap(self, other, ggap):
        # soma seg
        soma_seg = self.pc.gid2cell(self.gid).soma(.5)

        # assign the voltage at the soma to the gid of the cell
        self.pc.source_var(soma_seg._ref_v, self.gid, sec=soma_seg.sec)

        # create half gap_junction on the soma
        hg = h.HalfGap(soma_seg)

        # attach vgap to the voltage assigned for the 'other' cell's gid
        self.pc.target_var(hg, hg._ref_vgap, other.gid)

        # set the conductance of the half gap_junction
        # must match the second half of the gap_junction
        hg.g = ggap

        # save the state
        self.halfgap_list.append(hg)

    def set_recorder(self):
        # set soma and time recording vectors on the cell.
        # return: the soma and time vectors as a tuple.
        soma_v = h.Vector()   # Membrane potential vector at soma
        t = h.Vector()        # Time stamp vector
        soma_v.record(self.soma(0.5)._ref_v)
        t.record(h._ref_t)
        return soma_v, t


