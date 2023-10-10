"""
Symoblic Interval Class
Sung Woo Choi, 07/20/2023

"""

# !/usr/bin/python3
import copy
import torch
import glpk
import gurobipy as gp
from gurobipy import GRB


class SymbolicInterval(object):

    def __init__(self, lb, ub, al, au, prev_layer=None):
        self.lb = lb
        self.ub = ub
        self.al = al
        self.au = au
        self.dim = lb.shape[0]
        self.prev_layer = prev_layer

    def affineMap(self, W, b):
        lb, ub = self.backSub(al=si_min(W), au=si_max(W), )


    def backSub(self, al, au):
        lb = self.lb_backSub(al=al, au=au)
        ub = self.ub_backSub(al=al, au=au)
        return lb, ub

    def getRanges(self):
        lb = self.lb_backSub()
        ub = self.ub_backSub()
        return lb, ub
    
    def lb_backSub(self, al=None, au=None):
        pass

    def ub_backSub(self, al=None, au=None):
        pass


def si_max(a):
    a[a < 0] = 0
    return a

def si_min(a):
    a[a > 0] = 0
    return a

    