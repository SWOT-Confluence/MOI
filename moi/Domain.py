#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 22:11:07 2021

@author: mtd
"""


class Domain:
    def __init__(self,RiverData):
        self.nR=RiverData["nR"] #number of reaches
        self.xkm=RiverData["xkm"] #reach midpoint distance downstream [m]
        self.L=RiverData["L"]  #reach lengths, [m]
        self.nt=RiverData["nt"] #number of overpasses
        self.t=RiverData["t"] #time, [days]
        self.dt=RiverData["dt"] #time delta between successive overpasses, [seconds]
        
        
