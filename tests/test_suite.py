"""Unified unittest and pytest test suite for StarV."""

import os
import unittest
import numpy as np

os.environ.setdefault('MPLBACKEND', 'Agg')

from StarV.set.star import Star
from StarV.set.probstar import ProbStar
from StarV.set.sparsestar import SparseStar
from StarV.set.imagestar import ImageStar
from StarV.set.sparseimagestar2dcoo import SparseImageStar2DCOO
from StarV.set.sparseimagestar2dcsr import SparseImageStar2DCSR
from StarV.fun.poslin import PosLin
from StarV.fun.leakyrelu import LeakyReLU
from StarV.fun.satlin import SatLin
from StarV.fun.satlins import SatLins
from StarV.layer.FullyConnectedLayer import FullyConnectedLayer
from StarV.layer.ReLULayer import ReLULayer
from StarV.layer.AvgPool2DLayer import AvgPool2DLayer
from StarV.layer.MaxPool2DLayer import MaxPool2DLayer
from StarV.layer.LSTMLayer import LSTMLayer
from StarV.layer.GRULayer import GRULayer
from StarV.net.network import NeuralNetwork, rand_ffnn
from StarV.plant.lode import LODE
from StarV.plant.dlode import DLODE
from StarV.spec.dProbStarTL import AtomicPredicate, Predicate, Formula, _ALWAYS_, _EVENTUALLY_, _LeftBracket_, _RightBracket_
from StarV.nncs.nncs import NNCS, ReachPRM_NNCS
from StarV.verifier.verifier import Verifier
from StarV.verifier.certifier import Certifier


class TestStarVSets(unittest.TestCase):
    """Unit tests for StarV set representations and unified LP solver."""

    def test_star_lp_solvers(self):
        lb = np.array([-1.0, -2.0])
        ub = np.array([1.0, 2.0])
        s = Star(lb, ub)
        s = s.addConstraint(np.array([1.0, 1.0]), np.array([0.5]))
        for solver in ('linprog', 'glpk', 'gurobi'):
            l, u = s.getRanges(solver)
            np.testing.assert_allclose(l, np.array([-1.0, -2.0]), atol=1e-5)
            np.testing.assert_allclose(u, np.array([1.0, 1.5]), atol=1e-5)

    def test_probstar_probability_and_ranges(self):
        mu = np.array([0.0, 0.0])
        Sig = np.eye(2)
        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])
        ps = ProbStar(mu, Sig, lb, ub)
        prob = ps.estimateProbability()
        self.assertAlmostEqual(prob, 0.4660649426743925, places=6)
        l, u = ps.getRanges('linprog')
        np.testing.assert_allclose(l, lb, atol=1e-6)
        np.testing.assert_allclose(u, ub, atol=1e-6)

    def test_sparsestar_and_imagestar(self):
        lb = np.array([-1.0, 0.0])
        ub = np.array([1.0, 2.0])
        sp_star = SparseStar(lb, ub)
        l, u = sp_star.getRanges('linprog')
        np.testing.assert_allclose(l, lb, atol=1e-6)
        np.testing.assert_allclose(u, ub, atol=1e-6)

        img_lb = np.zeros((2, 2, 1))
        img_ub = np.ones((2, 2, 1))
        im_star = ImageStar(img_lb, img_ub)
        il, iu = im_star.getRanges('linprog')
        np.testing.assert_allclose(il.reshape(2, 2, 1), img_lb, atol=1e-6)
        np.testing.assert_allclose(iu.reshape(2, 2, 1), img_ub, atol=1e-6)

        coo = SparseImageStar2DCOO(img_lb, img_ub)
        csr = SparseImageStar2DCSR(img_lb, img_ub)
        cl, cu = coo.getRanges('linprog')
        sl, su = csr.getRanges('linprog')
        np.testing.assert_allclose(cl, sl, atol=1e-6)
        np.testing.assert_allclose(cu, su, atol=1e-6)


class TestStarVFunctionsAndLayers(unittest.TestCase):
    """Unit tests for activation functions and neural network layers."""

    def test_activation_functions(self):
        ps = ProbStar(np.zeros(2), np.eye(2), np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
        out_poslin = PosLin.reachExactSingleInput(ps, 'linprog')
        self.assertEqual(len(out_poslin), 4)

        out_leaky = LeakyReLU.reachExactSingleInput(ps, 0.1, 'linprog')
        self.assertEqual(len(out_leaky), 4)

        out_satlin = SatLin.reachExactSingleInput(ps, 'linprog')
        self.assertGreaterEqual(len(out_satlin), 1)

        out_satlins = SatLins.reachExactSingleInput(ps, 'linprog')
        self.assertGreaterEqual(len(out_satlins), 1)

    def test_pooling_layers_int_arguments(self):
        avg_pool = AvgPool2DLayer(kernel_size=2, stride=2, padding=0)
        self.assertEqual(tuple(avg_pool.kernel_size), (2, 2))
        self.assertEqual(tuple(avg_pool.stride), (2, 2))

        max_pool = MaxPool2DLayer(kernel_size=2, stride=2, padding=0)
        self.assertEqual(tuple(max_pool.kernel_size), (2, 2))
        self.assertEqual(tuple(max_pool.stride), (2, 2))

    def test_recurrent_layers_info(self):
        lstm = LSTMLayer.rand(3, 4, 1)
        self.assertEqual(lstm.in_dim, 3)
        self.assertEqual(lstm.out_dim, 4)
        lstm.info()

        gru = GRULayer.rand(3, 4, 1)
        self.assertEqual(gru.in_dim, 3)
        self.assertEqual(gru.out_dim, 4)
        gru.info()


class TestStarVPlantAndSpec(unittest.TestCase):
    """Unit tests for plant dynamics (LODE/DLODE) and temporal logic specifications."""

    def test_lode_and_dlode_validation(self):
        with self.assertRaises(AssertionError):
            LODE("invalid_A", np.eye(2), np.eye(2))
        with self.assertRaises(AssertionError):
            DLODE("invalid_A", np.eye(2), np.eye(2))

        plant = DLODE.rand(2, 1, 1)
        X0 = ProbStar.rand(2, 2, np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
        RX, RY = plant.multiStepReach(X0=X0, U=None, k=2)
        self.assertEqual(len(RX), 3)

    def test_dprobstartl_predicates(self):
        ap = AtomicPredicate.rand(3, t=2)
        self.assertEqual(ap.t, 2)
        p = Predicate.rand(3, t=4)
        self.assertEqual(p.t, 4)


class TestStarVVerifierAndNNCS(unittest.TestCase):
    """Unit tests for NNCS, Verifier, and Certifier."""

    def test_nncs_reach(self):
        np.random.seed(0)
        controller = rand_ffnn([1, 2, 1], ['relu', 'relu'])
        plant = DLODE.rand(2, 1, 1)
        sys = NNCS(controller, plant, type='DLNNCS')

        X0 = ProbStar(np.zeros(2), np.eye(2), np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
        prm = ReachPRM_NNCS()
        prm.initSet = X0
        prm.numSteps = 1
        prm.numCores = 1
        prm.show = False
        prm.method = 'exact-probstar'
        rx = sys.reach(prm)
        self.assertEqual(len(rx), 2)

    def test_verifier_and_certifier_oop(self):
        net = NeuralNetwork([FullyConnectedLayer([np.eye(2), np.zeros(2)]), ReLULayer()], 'ffnn')
        verifier = Verifier(lp_solver='linprog', method='BFS', n_processes=1)
        unsafe_mat = np.array([[-1.0, 0.0]])
        unsafe_vec = np.array([-0.5])

        # Qualitative verification
        s_in = Star(np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
        S, P, C, Res = verifier.verify(net, s_in, unsafe_mat, unsafe_vec, show=False)
        self.assertEqual(len(S), 4)
        self.assertIn('SAT', Res)

        # Quantitative verification
        ps_in = ProbStar(np.zeros(2), np.eye(2), np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
        S_p, P_p, C_p, p_lb, p_ub, p_min, p_max = verifier.verify(
            net, [ps_in], unsafe_mat, unsafe_vec, show=False
        )
        self.assertEqual(len(S_p), 4)
        self.assertGreater(p_lb, 0.0)

        # Certifier
        certifier = Certifier(lp_solver='linprog', method='BFS', reachMethod='approx', n_processes=1)
        sp_in = SparseStar(np.array([0.8, 0.1]), np.array([0.9, 0.2]))
        rb, vt, vt_total, Y = certifier.certify(net, sp_in, labels=0, show=False)
        self.assertEqual(rb, 1)


class TestLegacyModules(unittest.TestCase):
    """Run legacy Test classes in tests/ and assert n_fails == 0."""

    def _run_legacy_module(self, mod, exclude=()):
        import inspect
        runner = mod.Test()
        for name in dir(runner):
            if not name.startswith('test_') or name in exclude:
                continue
            attr = getattr(runner, name)
            if not callable(attr):
                continue
            sig = inspect.signature(attr)
            required = [
                p for p in sig.parameters.values()
                if p.default is inspect.Parameter.empty
                and p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            ]
            if not required:
                attr()
        self.assertEqual(runner.n_fails, 0, f"{mod.__name__} had {runner.n_fails} failure(s)")

    def test_legacy_set_modules(self):
        from tests import (
            test_set_star,
            test_set_probstar,
            test_set_sparsestar,
            test_set_sparseimagestar,
            test_set_imagestar,
        )
        self._run_legacy_module(test_set_star, exclude=('test_sampling', 'test_concatenate_with_vector'))
        for mod in (test_set_probstar, test_set_sparsestar, test_set_sparseimagestar):
            self._run_legacy_module(mod)
        self._run_legacy_module(test_set_imagestar, exclude=('test_matmul', 'test_transpose', 'test_sum', 'test_sum2', 'test_element_product'))

    def test_legacy_fun_and_layer_modules(self):
        from tests import (
            test_fun_poslin,
            test_fun_leakyrelu,
            test_fun_satlin,
            test_fun_satlins,
            test_layer_FullyConnectedLayer,
        )
        for mod in (
            test_fun_poslin,
            test_fun_leakyrelu,
            test_fun_satlin,
            test_fun_satlins,
            test_layer_FullyConnectedLayer,
        ):
            self._run_legacy_module(mod)

    def test_legacy_plant_spec_nncs_verifier(self):
        from tests import (
            test_plant_dlode,
            test_plant_lode,
            test_spec_dProbStar,
            test_nncs,
            test_verifier_star,
        )
        self._run_legacy_module(test_plant_lode, exclude=('test_LODE_stepReach',))
        for mod in (
            test_plant_dlode,
            test_spec_dProbStar,
            test_nncs,
            test_verifier_star,
        ):
            self._run_legacy_module(mod)


if __name__ == '__main__':
    unittest.main(verbosity=2)
