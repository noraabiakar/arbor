#!/usr/bin/env python
#coding: utf-8

import math
from neuron import h
from builtins import range

class cell:
    def __init__(self, gid, params=None):
        self.pc = h.ParallelContext()
        self.soma = None
        self.gid = gid
        self.sections = {}
        self.stims = []
        self.netcons = []
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

    def add_gap_junction(self, other, cm, gm, y, b, sl, xvec, ggap):
        xvec.x[0] = 0.95
        xvec.x[1] = 0.95
        sl.append(sec = self.soma)
        sl.append(sec = other.soma)
        area_src = self.soma.diam*math.pi * self.soma.L
        area_dest = other.soma.diam*math.pi * other.soma.L

        gm.x[0][0] =  ggap*0.1/area_src
        gm.x[0][1] = -ggap*0.1/area_src
        gm.x[1][1] =  ggap*0.1/area_dest
        gm.x[1][0] = -ggap*0.1/area_dest

        gj = h.LinearMechanism(cm, gm, y, b, sl, xvec)
        return gj

    def add_point_gap(self, other, ggap):
        if self.pc.gid_exists(self.gid):
            print(self.gid)
            seg0 = self.pc.gid2cell(self.gid).soma(.5)
            self.pc.source_var(seg0._ref_v, self.gid, sec=seg0.sec)
            hg0 = h.HalfGap(seg0)
            self.pc.target_var(hg0, hg0._ref_vgap, other.gid)
            hg0.g = ggap
            self.halfgap_list.append(hg0)

        if self.pc.gid_exists(other.gid):
            seg1 = self.pc.gid2cell(other.gid).soma(.5)
            self.pc.source_var(seg1._ref_v, other.gid, sec=seg1.sec)
            hg1 = h.HalfGap(seg1)
            self.pc.target_var(hg1, hg1._ref_vgap, self.gid)
            hg1.g = ggap
            self.halfgap_list.append(hg1)

    def mk_halfgap(self, gid_src, ggap, seg):
        self.pc.source_var(seg._ref_v, self.gid, sec=seg.sec)
        hg = h.HalfGap(seg)
        self.pc.target_var(hg, hg._ref_vgap, gid_src)
        hg.g = ggap
        self.halfgap_list.append(hg)

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


