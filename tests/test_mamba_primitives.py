"""
Test Suite for Phase 1 Mamba Mathematical Primitives:
- Softplus (convex relaxation)
- SiLU (convex bounding envelope)
- ExpDecay (discretization exponential relaxation)
- IdentityXIdentity (McCormick envelope on Star sets)

Verifies both analytical properties and empirical soundness via Monte Carlo sample containment.
"""

import sys
import numpy as np
from StarV.set.star import Star
from StarV.set.sparsestar import SparseStar
from StarV.fun.softplus import Softplus
from StarV.fun.silu import SiLU
from StarV.fun.exp_decay import ExpDecay
from StarV.fun.identityXidentity import IdentityXIdentity


class TestMambaPrimitives(object):

    def __init__(self):
        self.n_fails = 0
        self.n_tests = 0

    def record_result(self, success, name):
        self.n_tests += 1
        if success:
            print(f"[PASS] {name}")
        else:
            print(f"[FAIL] {name}")
            self.n_fails += 1

    def test_softplus(self):
        print("\n--- Testing Softplus Primitive ---")
        # 1. Scalar & Array Evaluation
        x = np.array([-2.0, 0.0, 3.0])
        y = Softplus.evaluate(x)
        expected = np.log1p(np.exp(x))
        eval_pass = np.allclose(y, expected, atol=1e-6)
        self.record_result(eval_pass, "Softplus evaluation")

        # 2. Derivative
        df = Softplus.df(x)
        expected_df = 1.0 / (1.0 + np.exp(-x))
        df_pass = np.allclose(df, expected_df, atol=1e-6)
        self.record_result(df_pass, "Softplus derivative")

        # 3. Star Reachability Soundness (Monte Carlo Containment)
        lb = np.array([-3.0, -0.5, 1.0])
        ub = np.array([-1.0, 1.5, 4.0])
        in_star = Star(lb, ub)
        out_star = Softplus.reach(in_star)

        # Draw 2,000 samples and assert all outputs are contained in out_star
        samples = in_star.sample(2000)
        y_samples = Softplus.evaluate(samples)
        contained = [out_star.contains(y_samples[:, i]) for i in range(y_samples.shape[1])]
        all_contained = all(contained)
        self.record_result(all_contained, f"Softplus Star soundness (2000 samples, {sum(contained)}/2000 contained)")

    def test_silu(self):
        print("\n--- Testing SiLU Primitive ---")
        # 1. Evaluation & Critical Value
        x_min = -1.27846454
        y_min = SiLU.evaluate(x_min)
        val_pass = np.isclose(y_min, SiLU.SILU_MIN_VAL, atol=1e-5)
        self.record_result(val_pass, "SiLU minimum value check")

        # 2. Derivative at minimum should be ~0
        df_min = SiLU.df(x_min)
        df_pass = np.isclose(df_min, 0.0, atol=1e-4)
        self.record_result(df_pass, "SiLU derivative at minimum is zero")

        # 3. Star Reachability Soundness across diverse intervals (convex, concave, zero-crossing)
        lb = np.array([-4.0, -1.0, 0.0, 2.5])
        ub = np.array([3.0, 2.0, 4.0, 6.0])
        in_star = Star(lb, ub)
        out_star = SiLU.reach(in_star)

        samples = in_star.sample(2000)
        y_samples = SiLU.evaluate(samples)
        contained = [out_star.contains(y_samples[:, i]) for i in range(y_samples.shape[1])]
        all_contained = all(contained)
        self.record_result(all_contained, f"SiLU Star soundness (2000 samples, {sum(contained)}/2000 contained)")

    def test_exp_decay(self):
        print("\n--- Testing ExpDecay Primitive ---")
        # delta in [0.01, 0.2], A in [-5.0, -1.0] <= 0
        lb = np.array([0.01, 0.05])
        ub = np.array([0.10, 0.20])
        delta_star = Star(lb, ub)
        A = np.array([-3.0, -1.5])

        out_star = ExpDecay.reach(delta_star, A)

        samples = delta_star.sample(2000)
        y_samples = ExpDecay.evaluate(samples, A[:, None])
        contained = [out_star.contains(y_samples[:, i]) for i in range(y_samples.shape[1])]
        all_contained = all(contained)
        self.record_result(all_contained, f"ExpDecay Star soundness (2000 samples, {sum(contained)}/2000 contained)")

    def test_mccormick_star(self):
        print("\n--- Testing IdentityXIdentity (McCormick) on Star Sets ---")
        X = Star(np.array([1.0, -2.0]), np.array([3.0, 1.0]))
        H = Star(np.array([-1.0, 2.0]), np.array([2.0, 4.0]))

        out_star = IdentityXIdentity.reach(X, H)

        # Verify dimension
        dim_pass = (out_star.dim == 2)
        self.record_result(dim_pass, "McCormick Star output dimension")

        # Monte Carlo Containment
        x_samples = X.sample(1000)
        h_samples = H.sample(1000)
        prod_samples = x_samples * h_samples
        contained = [out_star.contains(prod_samples[:, i]) for i in range(prod_samples.shape[1])]
        all_contained = all(contained)
        self.record_result(all_contained, f"McCormick Star soundness (1000 samples, {sum(contained)}/1000 contained)")

    def run_all(self):
        print("==================================================")
        print("  Running StarV Phase 1 Mamba Primitives Test Suite")
        print("==================================================")
        self.test_softplus()
        self.test_silu()
        self.test_exp_decay()
        self.test_mccormick_star()

        print("\n==================================================")
        print(f"  Summary: {self.n_tests - self.n_fails}/{self.n_tests} Tests Passed")
        if self.n_fails == 0:
            print("  ALL TESTS PASSED SUCCESSFULLY! Phase 1 math is verified.")
        else:
            print(f"  {self.n_fails} TESTS FAILED!")
        print("==================================================")
        return self.n_fails == 0


if __name__ == "__main__":
    test_runner = TestMambaPrimitives()
    success = test_runner.run_all()
    sys.exit(0 if success else 1)
