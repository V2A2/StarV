"""
Test for nncs module
Dung Tran
8/14/2023
"""

from StarV.net.network import rand_ffnn
from StarV.nncs.nncs import (
    NNCS,
    AEBS_NNCS,
    DynNN_NNCS,
    ReachPRM_NNCS,
    ReachPRM_DYNNN_NNCS,
    VerifyPRM_NNCS,
    getCurrentSpecConstraints,
    reachBFS_DLNNCS,
    reachBFS_AEBS,
    reachBFS_DynNN_NNCS,
    reachDFS_DLNNCS,
    simulate_DLNNCS,
    verifyBFS_DLNNCS,
    verify_temporal_specs_DLNNCS,
)
from StarV.plant.dlode import DLODE
from StarV.set.probstar import ProbStar
from StarV.spec.dProbStarTL import _ALWAYS_, _EVENTUALLY_, AtomicPredicate, Formula, _LeftBracket_, _RightBracket_, _AND_
from StarV.spec.dProbStarTL import DynamicFormula, Predicate
from StarV.util.load import load_acc_model, load_acc_trapezius_model, load_AEBS_model
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

    def test_approx_probstar_dlnncs_and_refinement(self):
        'test approximate ProbStar reachability, safety verification, and adaptive initial-space refinement'
        self.n_tests = self.n_tests + 1
        np.random.seed(42)

        controller = rand_ffnn([2, 6, 1], ['relu', 'purelin'])
        A = np.array([[1.0, 0.2], [0.0, 0.95]])
        B = np.array([[0.02], [0.2]])
        C = np.eye(2)
        D = np.zeros((2, 1))
        plant = DLODE(A, B, C, D)
        sys = NNCS(controller, plant, type='DLNNCS')

        lb0 = np.array([-0.5, -0.5])
        ub0 = np.array([0.5, 0.5])
        mu0 = np.zeros(2)
        Sig0 = np.diag([0.08, 0.08])
        V0 = np.hstack([np.zeros((2, 1)), np.eye(2)])
        X0 = ProbStar(V0, np.array([]), np.array([]), mu0, Sig0, lb0, ub0)

        unsafe_mat = np.array([[-1.0, 0.0]])  # x_0 >= 0.15 <=> -x_0 <= -0.15
        unsafe_vec = np.array([-0.15])

        # 1. Exact verification
        vprm_exact = VerifyPRM_NNCS()
        vprm_exact.initSet = X0.clone()
        vprm_exact.numSteps = 3
        vprm_exact.method = 'exact-probstar'
        vprm_exact.lpSolver = 'gurobi'
        vprm_exact.show = False
        vprm_exact.unsafeSpec = [unsafe_mat, unsafe_vec]
        res_exact = verifyBFS_DLNNCS(sys, vprm_exact)

        # 2. Single-partition Approximate ProbStar verification
        vprm_approx = VerifyPRM_NNCS()
        vprm_approx.initSet = X0.clone()
        vprm_approx.numSteps = 3
        vprm_approx.method = 'approx-probstar'
        vprm_approx.lpSolver = 'gurobi'
        vprm_approx.show = False
        vprm_approx.unsafeSpec = [unsafe_mat, unsafe_vec]
        res_approx = verifyBFS_DLNNCS(sys, vprm_approx)

        for k in range(len(res_exact.Qt)):
            assert len(res_approx.RX[k]) == 1, f'Expected 1 ProbStar at step {k}, got {len(res_approx.RX[k])}'
            p_ex = res_exact.Qt[k]
            p_lb = res_approx.Qt_lb[k]
            p_hat = res_approx.Qt[k]
            p_ub = res_approx.Qt_ub[k]
            assert p_lb <= p_ex + 1e-4, f'Step {k}: p_lb ({p_lb}) > p_exact ({p_ex})'
            assert p_ex <= p_ub + 1e-4, f'Step {k}: p_exact ({p_ex}) > p_ub ({p_ub})'
            assert p_lb - 1e-7 <= p_hat <= p_ub + 1e-7

        # 3. Adaptive Initial-Space Refinement
        vprm_ref = VerifyPRM_NNCS()
        vprm_ref.initSet = X0.clone()
        vprm_ref.numSteps = 3
        vprm_ref.method = 'approx-probstar'
        vprm_ref.lpSolver = 'gurobi'
        vprm_ref.show = False
        vprm_ref.unsafeSpec = [unsafe_mat, unsafe_vec]
        vprm_ref.refine_max_splits = 4
        vprm_ref.refine_tol = 1e-4
        res_ref = verifyBFS_DLNNCS(sys, vprm_ref)

        for k in range(len(res_exact.Qt)):
            p_ex = res_exact.Qt[k]
            gap_single = res_approx.Qt_ub[k] - res_approx.Qt_lb[k]
            gap_ref = res_ref.Qt_ub[k] - res_ref.Qt_lb[k]
            assert gap_ref <= gap_single + 1e-6
            assert res_ref.Qt_lb[k] <= p_ex + 1e-4
            assert p_ex <= res_ref.Qt_ub[k] + 1e-4

        # 4. Pointwise simulation soundness check
        samples0 = X0.sampling_alpha0(50, seed=7)
        for s in range(samples0.shape[1]):
            alpha_s = samples0[:, s]
            sim_trace = simulate_DLNNCS(sys, alpha_s, [], 3)
            for k, x_s in enumerate(sim_trace):
                rk = res_approx.RX[k][0]
                d_lb0, W_lb0, d_ub0, W_ub0 = rk.getSoundInputBounds()
                # Note getSoundInputBounds returns (W_lb, d_lb, W_ub, d_ub)
                W_lb, d_lb, W_ub, d_ub = rk.getSoundInputBounds()
                x_lo = d_lb + W_lb @ alpha_s
                x_hi = d_ub + W_ub @ alpha_s
                assert np.all(x_s >= x_lo - 1e-6), f'Lower envelope violated at step {k}: {x_s} < {x_lo}'
                assert np.all(x_s <= x_hi + 1e-6), f'Upper envelope violated at step {k}: {x_s} > {x_hi}'

    def test_smooth_nonlinear_and_temporal_nncs(self):
        'test Tanh/Sigmoid NNCS controller and dProbStarTL temporal verification with approx-probstar'
        self.n_tests = self.n_tests + 1
        np.random.seed(11)

        # Smooth Tanh controller (cannot be verified by exact-probstar!)
        tanh_ctrl = rand_ffnn([2, 5, 1], ['tansig', 'purelin'])
        A = np.array([[1.0, 0.1], [0.0, 0.9]])
        B = np.array([[0.01], [0.1]])
        plant = DLODE(A, B, np.eye(2), np.zeros((2, 1)))
        sys_tanh = NNCS(tanh_ctrl, plant, type='DLNNCS')

        lb0 = np.array([-0.4, -0.4])
        ub0 = np.array([0.4, 0.4])
        X0 = ProbStar(
            np.hstack([np.zeros((2, 1)), np.eye(2)]),
            np.array([]), np.array([]),
            np.zeros(2), np.diag([0.05, 0.05]), lb0, ub0
        )

        rprm = ReachPRM_NNCS()
        rprm.initSet = X0.clone()
        rprm.numSteps = 3
        rprm.method = 'approx-probstar'
        rprm.show = False
        RX_tanh, _ = reachBFS_DLNNCS(sys_tanh, rprm)
        assert len(RX_tanh) == 5
        assert all(len(rk) == 1 for rk in RX_tanh)

        # Temporal logic verification on ReLU DLNNCS: exact vs approx-probstar
        relu_ctrl = rand_ffnn([2, 4, 1], ['relu', 'purelin'])
        sys_relu = NNCS(relu_ctrl, plant, type='DLNNCS')
        P_unsafe = AtomicPredicate(np.array([-1.0, 0.0]), np.array([-0.1]))
        spec = Formula([_EVENTUALLY_(0, 2), _LeftBracket_(), P_unsafe, _RightBracket_()])

        vprm_ex = VerifyPRM_NNCS()
        vprm_ex.initSet = X0.clone()
        vprm_ex.numSteps = 2
        vprm_ex.method = 'exact-probstar'
        vprm_ex.show = False
        vprm_ex.temporalSpecs = [spec]
        _, p_max_ex, p_min_ex, _, _, _ = verify_temporal_specs_DLNNCS(sys_relu, vprm_ex)

        vprm_ap = VerifyPRM_NNCS()
        vprm_ap.initSet = X0.clone()
        vprm_ap.numSteps = 2
        vprm_ap.method = 'approx-probstar'
        vprm_ap.show = False
        vprm_ap.temporalSpecs = [spec]
        _, p_max_ap, p_min_ap, _, _, _ = verify_temporal_specs_DLNNCS(sys_relu, vprm_ap)

        assert p_min_ap[0] <= p_min_ex[0] + 1e-4
        assert p_max_ex[0] <= p_max_ap[0] + 1e-4

    def test_acc_aebs_dynnn_approx_probstar(self):
        'test ACC, AEBS, and DynNN_NNCS with approx-probstar'
        self.n_tests = self.n_tests + 1
        np.random.seed(19)

        # 1. ACC Trapezius + DLODE plant
        acc_sys, X0_acc, ref_acc, unsafe_mat, unsafe_vec = load_acc_trapezius_model()
        vprm_acc = VerifyPRM_NNCS()
        vprm_acc.initSet = X0_acc
        vprm_acc.numSteps = 2
        vprm_acc.refInputs = ref_acc
        vprm_acc.method = 'approx-probstar'
        vprm_acc.show = False
        vprm_acc.unsafeSpec = [unsafe_mat, unsafe_vec]
        res_acc = verifyBFS_DLNNCS(acc_sys, vprm_acc)
        assert len(res_acc.Qt) == 4
        assert all(len(rk) == 1 for rk in res_acc.RX)

        # 2. AEBS_NNCS
        controller, transformer, norm_mat, scale_mat, plant, initSets = load_AEBS_model()
        aebs_sys = AEBS_NNCS(controller, transformer, norm_mat, scale_mat, plant)
        rprm_aebs = ReachPRM_NNCS()
        rprm_aebs.initSet = initSets[0]
        rprm_aebs.numSteps = 2
        rprm_aebs.method = 'approx-probstar'
        rprm_aebs.show = False
        RX_aebs, _ = reachBFS_AEBS(aebs_sys, rprm_aebs)
        assert len(RX_aebs) == 4
        assert all(len(rk) == 1 for rk in RX_aebs)

        # 3. DynNN_NNCS
        ctrl_dyn = rand_ffnn([2, 4, 1], ['relu', 'purelin'])
        plant_dyn = rand_ffnn([3, 5, 2], ['relu', 'purelin'])
        dyn_sys = DynNN_NNCS(ctrl_dyn, plant_dyn)
        X0_dyn = ProbStar(
            np.hstack([np.zeros((2, 1)), np.eye(2)]),
            np.array([]), np.array([]),
            np.zeros(2), np.diag([0.04, 0.04]),
            np.array([-0.3, -0.3]), np.array([0.3, 0.3])
        )
        rprm_dyn = ReachPRM_DYNNN_NNCS()
        rprm_dyn.initSet = X0_dyn
        rprm_dyn.numSteps = 2
        rprm_dyn.method = 'approx-probstar'
        rprm_dyn.show = False
        RX_dyn, _ = reachBFS_DynNN_NNCS(dyn_sys, rprm_dyn)
        assert len(RX_dyn) == 3
        assert all(len(rk) == 1 for rk in RX_dyn)

if __name__ == "__main__":
    test = Test()
    print('\n=======================\
    ================================\
    ================================\
    ===============================\n')
    test.test_constructor()
    test.test_reachDLNNCS()
    test.test_getCurrentSpecConstraints()
    test.test_approx_probstar_dlnncs_and_refinement()
    test.test_smooth_nonlinear_and_temporal_nncs()
    test.test_acc_aebs_dynnn_approx_probstar()
    print('\n========================\
    =================================\
    =================================\
    =================================\n')
    print('Testing NNCS Class: fails: {}, successfull: {}, \
    total tests: {}'.format(test.n_fails,
                            test.n_tests - test.n_fails,
                            test.n_tests))
            
