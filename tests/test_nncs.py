"""
Test for nncs module
Dung Tran
8/14/2023
"""

from StarV.net.network import rand_ffnn
from StarV.nncs.nncs import NNCS, ReachPRM_NNCS, getCurrentSpecConstraints
from StarV.plant.dlode import DLODE
from StarV.set.probstar import ProbStar
from StarV.spec.dProbStarTL import _ALWAYS_, _EVENTUALLY_, AtomicPredicate, Formula, _LeftBracket_, _RightBracket_, _AND_
from StarV.spec.dProbStarTL import DynamicFormula, Predicate
import numpy as np

class Test(object):
    """Testing nncs module"""

    def __init__(self):
        self.n_fails = 0
        self.n_tests = 0

    def test_constructor(self):

        self.n_tests = self.n_tests + 1

        try:
            controller = rand_ffnn([1, 2, 1], ['relu', 'relu'])
            plant = DLODE.rand(2,1,1)
            sys = NNCS(controller, plant, type='linear-NNCS')
            sys.info()

        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successful!')

    def test_reachDLNNCS(self):
        'test reachability analysis for discrete linear NNCS'

        self.n_tests = self.n_tests + 1

        controller = rand_ffnn([1, 2, 1], ['relu', 'relu'])    # network controller
        plant = DLODE.rand(2,1,1)                              # plant dynamics
        sys = NNCS(controller, plant, type='DLNNCS')           # NNCS object
        sys.info()

        X0 = ProbStar.rand(2,2,np.array([-1.0, -1.0]), np.array([1., 1.]))  # initial set
        reachPRM = ReachPRM_NNCS()
        reachPRM.initSet = X0
        reachPRM.lpSolver = 'gurobi'
        reachPRM.numSteps = 2
        reachPRM.numCores = 1
        reachPRM.show = True
        reachPRM.method = 'exact-probstar'

        sys.reach(reachPRM)
        
        print('RX = {}'.format(sys.RX))
        print('RY = {}'.format(sys.RY))
        print('RU = {}'.format(sys.RU))
        

    def test_stepSim_DLNNCS(self):
        'test step simulation of discrete linear NNCS'

        self.n_tests = self.n_tests + 1

        controller = rand_ffnn([1, 2, 1], ['relu', 'relu'])    # network controller
        plant = DLODE.rand(2,1,1)                              # plant dynamics
        sys = NNCS(controller, plant, type='DLNNCS')           # NNCS object
        sys.info()
        X0 = ProbStar.rand(2,2,np.array([-1.0, -1.0]), np.array([1., 1.]))  # initial set
    
        
    def test_getCurrentSpecConstraints(self):
        'test getCurrentSpecContraints method'
       
        self.n_tests = self.n_tests + 1
        A = np.array([3., -1.])
        b = np.array([1.])

        P1 = AtomicPredicate(A, b)
        op1 = _EVENTUALLY_(0, 2)
        lb1 = _LeftBracket_()
        rb1 = _RightBracket_()

        f = [op1, lb1, P1, rb1]
        spec = Formula(f)
        spec.print()

        abs_spec = spec.getDynamicFormula()
        abs_spec.print()

        P = getCurrentSpecConstraints(abs_spec.F[0], 0)
        P.print_info()

if __name__ == "__main__":
    test = Test()
    print('\n=======================\
    ================================\
    ================================\
    ===============================\n')
    #test.test_constructor()
    #test.test_reachDLNNCS()
    test.test_getCurrentSpecConstraints()
    print('\n========================\
    =================================\
    =================================\
    =================================\n')
    print('Testing NNCS Class: fails: {}, successfull: {}, \
    total tests: {}'.format(test.n_fails,
                            test.n_tests - test.n_fails,
                            test.n_tests))
            
