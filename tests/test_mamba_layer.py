"""
Test Suite for Phase 2 Mamba Layer Integration:
- Numerical equivalence against PyTorch reference implementation (MicroMamba)
- Star reachability soundness under zero initial state (Monte Carlo containment)
- Star reachability soundness under non-zero initial state
- Joint trajectory reachability across time steps via PredicatePool stacking
- Multi-step sequence reachability (L=3)

Verifies full Mamba (S6) block reachability in StarV.
"""

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from StarV.set.star import Star
from StarV.layer.MambaLayer import MambaLayer


class MicroMamba(nn.Module):
    """Reference PyTorch implementation of the canonical Mamba (S6) block."""

    def __init__(self, d_model=2, d_inner=2, d_state=2, d_conv=2, dt_rank=2):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_inner
        self.d_state = d_state
        self.d_conv = d_conv
        self.dt_rank = dt_rank

        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            kernel_size=d_conv,
            groups=d_inner,
            bias=True
        )
        self.x_proj = nn.Linear(d_inner, dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
        self.A_log = nn.Parameter(torch.randn(d_inner, d_state))
        self.D = nn.Parameter(torch.randn(d_inner))
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, u, h0=None):
        """
        u: (L, d_model)
        h0: optional (d_inner, d_state)
        """
        L = u.shape[0]
        xz = self.in_proj(u)
        x = xz[:, :self.d_inner]
        z = xz[:, self.d_inner:]

        x_trans = x.transpose(0, 1).unsqueeze(0)
        x_padded = F.pad(x_trans, (self.d_conv - 1, 0))
        x_conv = self.conv1d(x_padded).squeeze(0).transpose(0, 1)
        s = F.silu(x_conv)

        A = -torch.exp(self.A_log)
        if h0 is None:
            h = torch.zeros(self.d_inner, self.d_state, dtype=u.dtype, device=u.device)
        else:
            h = h0.clone()

        outs, hs = [], []
        for t in range(L):
            st = s[t]
            x_proj_out = self.x_proj(st)
            dt = x_proj_out[:self.dt_rank]
            Bt = x_proj_out[self.dt_rank : self.dt_rank + self.d_state]
            Ct = x_proj_out[self.dt_rank + self.d_state :]

            delta = F.softplus(self.dt_proj(dt))
            Abar = torch.exp(delta.unsqueeze(-1) * A)
            dB = (delta * st).unsqueeze(-1) * Bt.unsqueeze(0)
            h = Abar * h + dB

            y = torch.matmul(h, Ct) + self.D * st
            g = y * F.silu(z[t])
            out = self.out_proj(g)
            outs.append(out)
            hs.append(h.view(-1))

        return torch.stack(outs, dim=0), torch.stack(hs, dim=0)


class TestMambaLayer(object):

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

    def test_numerical_equivalence_pytorch(self):
        print("\n--- Testing Numerical Equivalence (PyTorch vs MambaLayer) ---")
        torch.manual_seed(42)
        model = MicroMamba(d_model=2, d_inner=2, d_state=2, d_conv=2, dt_rank=2).double()
        mamba_layer = MambaLayer.from_torch(model)

        u = torch.randn(3, 2, dtype=torch.float64)
        torch_out, torch_h = model(u)
        np_out, np_h = mamba_layer.evaluate(u.numpy(), return_hidden=True)

        diff_out = np.max(np.abs(torch_out.detach().numpy() - np_out))
        diff_h = np.max(np.abs(torch_h.detach().numpy() - np_h))

        pass_out = diff_out < 1e-10
        pass_h = diff_h < 1e-10
        self.record_result(pass_out, f"Output numerical equivalence (max diff = {diff_out:.2e})")
        self.record_result(pass_h, f"Hidden state numerical equivalence (max diff = {diff_h:.2e})")

    def test_reach_soundness_zero_initial_state(self):
        print("\n--- Testing Reachability Soundness (H0 = None) ---")
        torch.manual_seed(42)
        model = MicroMamba(d_model=2, d_inner=2, d_state=2, d_conv=2, dt_rank=2).double()
        mamba_layer = MambaLayer.from_torch(model, lp_solver='gurobi')

        u0 = np.array([0.1, -0.2])
        u1 = np.array([-0.1, 0.3])
        eps = 0.05
        star0 = Star(u0 - eps, u0 + eps)
        star1 = Star(u1 - eps, u1 + eps)
        input_stars = [star0, star1]

        Y, H, pool = mamba_layer.reach(input_stars, lp_solver='gurobi')

        N_SAMPLES = 500
        samples0 = star0.sample(N_SAMPLES)
        samples1 = star1.sample(N_SAMPLES)

        y0_cnt, y1_cnt = 0, 0
        h0_cnt, h1_cnt = 0, 0

        for i in range(N_SAMPLES):
            u_traj = np.array([samples0[:, i], samples1[:, i]])
            outs, hs = mamba_layer.evaluate(u_traj, return_hidden=True)
            if Y[0].contains(outs[0]):
                y0_cnt += 1
            if Y[1].contains(outs[1]):
                y1_cnt += 1
            if H[0].contains(hs[0]):
                h0_cnt += 1
            if H[1].contains(hs[1]):
                h1_cnt += 1

        self.record_result(y0_cnt == N_SAMPLES, f"Output Y[0] containment ({y0_cnt}/{N_SAMPLES})")
        self.record_result(y1_cnt == N_SAMPLES, f"Output Y[1] containment ({y1_cnt}/{N_SAMPLES})")
        self.record_result(h0_cnt == N_SAMPLES, f"Hidden state H[0] containment ({h0_cnt}/{N_SAMPLES})")
        self.record_result(h1_cnt == N_SAMPLES, f"Hidden state H[1] containment ({h1_cnt}/{N_SAMPLES})")

    def test_reach_soundness_nonzero_initial_state(self):
        print("\n--- Testing Reachability Soundness (Non-zero H0) ---")
        torch.manual_seed(42)
        model = MicroMamba(d_model=2, d_inner=2, d_state=2, d_conv=2, dt_rank=2).double()
        mamba_layer = MambaLayer.from_torch(model, lp_solver='gurobi')

        u0 = np.array([0.1, -0.2])
        u1 = np.array([-0.1, 0.3])
        eps = 0.05
        star0 = Star(u0 - eps, u0 + eps)
        star1 = Star(u1 - eps, u1 + eps)
        input_stars = [star0, star1]

        h0_nom = np.array([0.05, -0.05, 0.02, -0.02])
        h0_star = Star(h0_nom - 0.01, h0_nom + 0.01)

        Y, H, pool = mamba_layer.reach(input_stars, H0=h0_star, lp_solver='gurobi')

        N_SAMPLES = 500
        samples0 = star0.sample(N_SAMPLES)
        samples1 = star1.sample(N_SAMPLES)
        samples_h0 = h0_star.sample(N_SAMPLES)

        y0_cnt, y1_cnt = 0, 0
        di, N = mamba_layer.d_inner, mamba_layer.d_state

        for i in range(N_SAMPLES):
            u_traj = torch.tensor(np.array([samples0[:, i], samples1[:, i]]), dtype=torch.float64)
            h0_init = torch.tensor(samples_h0[:, i].reshape(di, N), dtype=torch.float64)
            outs_t, hs_t = model(u_traj, h0=h0_init)

            out_np = outs_t.detach().numpy()
            if Y[0].contains(out_np[0]):
                y0_cnt += 1
            if Y[1].contains(out_np[1]):
                y1_cnt += 1

        self.record_result(y0_cnt == N_SAMPLES, f"Non-zero H0: Y[0] containment ({y0_cnt}/{N_SAMPLES})")
        self.record_result(y1_cnt == N_SAMPLES, f"Non-zero H0: Y[1] containment ({y1_cnt}/{N_SAMPLES})")

    def test_joint_trajectory_stacking(self):
        print("\n--- Testing Joint Trajectory Stacking (PredicatePool.stack) ---")
        torch.manual_seed(42)
        model = MicroMamba(d_model=2, d_inner=2, d_state=2, d_conv=2, dt_rank=2).double()
        mamba_layer = MambaLayer.from_torch(model, lp_solver='gurobi')

        u0 = np.array([0.1, -0.2])
        u1 = np.array([-0.1, 0.3])
        eps = 0.04
        star0 = Star(u0 - eps, u0 + eps)
        star1 = Star(u1 - eps, u1 + eps)

        Y, H, pool = mamba_layer.reach([star0, star1], lp_solver='gurobi')
        stacked_Y = pool.stack(Y)

        N_SAMPLES = 300
        samples0 = star0.sample(N_SAMPLES)
        samples1 = star1.sample(N_SAMPLES)
        stacked_cnt = 0

        for i in range(N_SAMPLES):
            u_traj = np.array([samples0[:, i], samples1[:, i]])
            outs = mamba_layer.evaluate(u_traj)
            joint_out = np.concatenate(outs)
            if stacked_Y.contains(joint_out):
                stacked_cnt += 1

        self.record_result(stacked_cnt == N_SAMPLES,
                           f"Joint trajectory Star containment ({stacked_cnt}/{N_SAMPLES})")

    def test_sequence_length_3(self):
        print("\n--- Testing Sequence Length L=3 Reachability ---")
        torch.manual_seed(42)
        model = MicroMamba(d_model=2, d_inner=2, d_state=2, d_conv=2, dt_rank=2).double()
        mamba_layer = MambaLayer.from_torch(model, lp_solver='gurobi')

        inputs = [
            Star(np.array([0.1, -0.1]) - 0.03, np.array([0.1, -0.1]) + 0.03),
            Star(np.array([0.0, 0.2]) - 0.03, np.array([0.0, 0.2]) + 0.03),
            Star(np.array([-0.1, 0.1]) - 0.03, np.array([-0.1, 0.1]) + 0.03)
        ]

        Y, H, pool = mamba_layer.reach(inputs, lp_solver='gurobi')

        N_SAMPLES = 200
        s0 = inputs[0].sample(N_SAMPLES)
        s1 = inputs[1].sample(N_SAMPLES)
        s2 = inputs[2].sample(N_SAMPLES)

        all_ok = True
        for i in range(N_SAMPLES):
            u_traj = np.array([s0[:, i], s1[:, i], s2[:, i]])
            outs = mamba_layer.evaluate(u_traj)
            for t in range(3):
                if not Y[t].contains(outs[t]):
                    all_ok = False
                    break
            if not all_ok:
                break

        self.record_result(all_ok, f"L=3 all-steps trajectory containment (checked {N_SAMPLES} samples)")

    def run_all(self):
        print("==================================================")
        print("  Running StarV Phase 2 Mamba Layer Test Suite")
        print("==================================================")
        self.test_numerical_equivalence_pytorch()
        self.test_reach_soundness_zero_initial_state()
        self.test_reach_soundness_nonzero_initial_state()
        self.test_joint_trajectory_stacking()
        self.test_sequence_length_3()

        print("\n==================================================")
        print(f"  Summary: {self.n_tests - self.n_fails}/{self.n_tests} Tests Passed")
        if self.n_fails == 0:
            print("  ALL TESTS PASSED SUCCESSFULLY! Phase 2 is verified.")
        else:
            print(f"  {self.n_fails} TESTS FAILED!")
        print("==================================================")
        return self.n_fails == 0


if __name__ == "__main__":
    test_runner = TestMambaLayer()
    success = test_runner.run_all()
    sys.exit(0 if success else 1)
