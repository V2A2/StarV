#########################################################################
##   This file is part of the StarV verifier                           ##
##                                                                     ##
##   Copyright (c) 2025 The StarV Team                                 ##
##   License: BSD-3-Clause                                             ##
##                                                                     ##
##   Primary contacts: Hoang Dung Tran <dungtran@ufl.edu> (UF)         ##
##                     Sung Woo Choi <sungwoo.choi@ufl.edu> (UF)       ##
##                     Yuntao Li <yli17@ufl.edu> (UF)                  ##
##                     Qing Liu <qliu1@ufl.edu> (UF)                   ##
##                                                                     ##
##   See CONTRIBUTORS for full author contacts and affiliations.       ##
##   This program is licensed under the BSD 3‑Clause License; see the  ##
##   LICENSE file in the root directory.                               ##
#########################################################################
"""
Concatenate Layer Class
Sung Woo Choi, 06/07/2023
"""

import copy
import numpy as np

class ConcatenateLayer(object):
    """ ConcatenateLayer class
        Author: Sung Woo Choi
        Date: 06/07/2023
    """

    @staticmethod
    def evaluate(x, axis=None):
        """ concatenates a list of numpy arrays, x """
        return np.concatenate(x, axis=axis)
    
    @staticmethod
    def reach(in_sets, method=None, lp_solver='gurobi', pool=None, RF=0.0, DR=0, show=False):
        """ main concatenate layer method 
            Args:
                @in_sets: a list of input sets (SparseStar)

            Return:
                @S: concatenated reachable set
        """

        S = copy.deepcopy(in_sets[0])
        for i in range(1, len(in_sets)):
            S = S.concatenate(in_sets[i])
        return [S]
