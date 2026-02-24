"""
Execution engine for lifting expression graphs.
Author: Zhuoyang Zhou
Date: 02/18/2026

"""

import copy
import numpy as np

from StarV.set.star import Star
from StarV.operators.Sine import SinLayer
from StarV.operators.Cosine import CosLayer
from StarV.operators.PowerEven import PowerEvenOperater
from StarV.operators.PowerOdd import PowerOddOperater
from StarV.operators.Multiply import MultiplyOperater
from StarV.lifting.exprgraph import ExpressionGraph


class ExpressionExecutor:
    """
    Execute an ExpressionGraph on top of a Star set.

    Notes for Step 1:
    - `sin`/`cos` now use shared-predicate high-dimensional reach with `idx`.
    - `pow_even`/`pow_odd` now use shared-predicate high-dimensional reach with `idx`.
    """

    def __init__(self, operator_lib=None, lp_solver='gurobi', RF=0.0):
        self.lp_solver = lp_solver
        self.RF = RF

        default_lib = {
            'sin': self._run_sin,
            'cos': self._run_cos,
            'pow_even': self._run_pow_even,
            'pow_odd': self._run_pow_odd,
            'mul': self._run_mul,
        }

        self.operator_lib = default_lib if operator_lib is None else operator_lib

    def run(self, I: Star, graph: ExpressionGraph, var_dims: dict[str, int]) -> tuple[Star | list[Star], dict]:
        if not isinstance(I, Star):
            raise TypeError('I must be a Star')
        if not isinstance(graph, ExpressionGraph):
            raise TypeError('graph must be an ExpressionGraph')

        graph.validate()
        order = graph.topological_sort()

        stars: list[Star] = [copy.deepcopy(I)]
        node_dim_map: dict[int | str, int] = {}

        for nid in order:
            node = graph.nodes[nid]
            op = node.op

             # All active branches must share the same state dimension.
            in_dims = {S.dim for S in stars}
            if len(in_dims) != 1:
                raise ValueError(
                    f"Inconsistent branch dimensions before node '{nid}': {sorted(in_dims)}"
                )
            in_dim = next(iter(in_dims))

            if op == 'var':
                var_name = node.params['name']
                if var_name not in var_dims:
                    raise KeyError(f"Missing var mapping for '{var_name}'")
                dim = int(var_dims[var_name])
                if dim < 0 or dim >= in_dim:
                    raise ValueError(
                        f"var_dims['{var_name}']={dim} out of range [0, {in_dim - 1}]"
                    )
                node_dim_map[nid] = dim
                node.out_dim = dim
                continue

            new_stars: list[Star] = []

            for S in stars:
                if op == 'const':
                    value = float(node.params['value'])
                    R = self._append_linear_dim(S, coeff=np.zeros(S.dim), bias=value)
                    new_stars.extend(self._as_star_list(R))
                    continue

                if op == 'neg':
                    self._require_num_inputs(node, 1)
                    x_dim = node_dim_map[node.inputs[0]]
                    coeff = np.zeros(S.dim)
                    coeff[x_dim] = -1.0
                    R = self._append_linear_dim(S, coeff=coeff, bias=0.0)
                    new_stars.extend(self._as_star_list(R))
                    continue

                if op == 'add':
                    self._require_num_inputs(node, 2)
                    x_dim = node_dim_map[node.inputs[0]]
                    y_dim = node_dim_map[node.inputs[1]]
                    coeff = np.zeros(S.dim)
                    coeff[x_dim] = 1.0
                    coeff[y_dim] = 1.0
                    R = self._append_linear_dim(S, coeff=coeff, bias=0.0)
                    new_stars.extend(self._as_star_list(R))
                    continue

                if op == 'sub':
                    self._require_num_inputs(node, 2)
                    x_dim = node_dim_map[node.inputs[0]]
                    y_dim = node_dim_map[node.inputs[1]]
                    coeff = np.zeros(S.dim)
                    coeff[x_dim] = 1.0
                    coeff[y_dim] = -1.0
                    R = self._append_linear_dim(S, coeff=coeff, bias=0.0)
                    new_stars.extend(self._as_star_list(R))
                    continue

                if op in ('sin', 'cos', 'pow_even', 'pow_odd'):
                    self._require_num_inputs(node, 1)
                    src_dim = node_dim_map[node.inputs[0]]
                    R = self.operator_lib[op](S, src_dim, node.params)
                    new_stars.extend(self._as_star_list(R))
                    continue

                if op == 'mul':
                    self._require_num_inputs(node, 2)
                    x_dim = node_dim_map[node.inputs[0]]
                    y_dim = node_dim_map[node.inputs[1]]
                    R = self.operator_lib['mul'](S, x_dim, y_dim, node.params)
                    new_stars.extend(self._as_star_list(R))
                    continue

                raise NotImplementedError(f"Op '{op}' is not implemented in ExpressionExecutor")

            if len(new_stars) == 0:
                raise RuntimeError(f"Node '{nid}' with op '{op}' produced no output Stars")

            out_dims = {S.dim for S in new_stars}
            if len(out_dims) != 1:
                raise ValueError(
                    f"Inconsistent branch dimensions after node '{nid}' ({op}): {sorted(out_dims)}"
                )
            out_dim = next(iter(out_dims))
            if out_dim != in_dim + 1:
                raise ValueError(
                    f"Node '{nid}' ({op}) expected dim {in_dim + 1} after append but got {out_dim}"
                )

            node_dim_map[nid] = out_dim - 1
            node.out_dim = out_dim - 1
            stars = new_stars

        meta = {
            'node_to_dim': node_dim_map,
            'output_node_ids': list(graph.output_node_ids),
            'output_dims': [node_dim_map[nid] for nid in graph.output_node_ids],
        }
        if len(stars) == 1:
            return stars[0], meta
        return stars, meta

    @staticmethod
    def _require_num_inputs(node, n_expected: int) -> None:
        if len(node.inputs) != n_expected:
            raise ValueError(
                f"Node '{node.node_id}' with op '{node.op}' expects {n_expected} inputs, "
                f"but got {len(node.inputs)}"
            )

    @staticmethod
    def _append_linear_dim(S: Star, coeff: np.ndarray, bias: float = 0.0) -> Star:
        if coeff.shape != (S.dim,):
            raise ValueError(f'coeff shape must be ({S.dim},), got {coeff.shape}')

        A = np.zeros((S.dim + 1, S.dim))
        A[: S.dim, :] = np.eye(S.dim)
        A[S.dim, :] = coeff

        b = np.zeros(S.dim + 1)
        b[S.dim] = bias
        return S.affineMap(A=A, b=b)

    @staticmethod
    def _append_independent_interval_dim(S: Star, lb: float, ub: float) -> Star:
        if ub < lb:
            raise ValueError(f'Invalid interval: lb={lb}, ub={ub}')

        if abs(ub - lb) <= 1e-12:
            return ExpressionExecutor._append_linear_dim(S, coeff=np.zeros(S.dim), bias=float(lb))

        center = 0.5 * (lb + ub)
        radius = 0.5 * (ub - lb)

        new_V = np.zeros((S.dim + 1, S.nVars + 2))
        new_V[: S.dim, : S.nVars + 1] = S.V
        new_V[S.dim, 0] = center
        new_V[S.dim, S.nVars + 1] = radius

        if len(S.C) > 0:
            new_C = np.hstack([S.C, np.zeros((S.C.shape[0], 1))])
            new_d = S.d.copy()
        else:
            new_C = np.empty((0, S.nVars + 1))
            new_d = np.empty((0,))

        new_pred_lb = np.hstack([S.pred_lb, -1.0])
        new_pred_ub = np.hstack([S.pred_ub, 1.0])
        return Star(new_V, new_C, new_d, new_pred_lb, new_pred_ub)

    @staticmethod
    def _project_dim_to_1d(S: Star, src_dim: int) -> Star:
        if src_dim < 0 or src_dim >= S.dim:
            raise ValueError(f'src_dim={src_dim} out of range [0, {S.dim - 1}]')
        A = np.zeros((1, S.dim))
        A[0, src_dim] = 1.0
        return S.affineMap(A=A, b=np.zeros(1))

    @staticmethod
    def _ensure_single_star(R, op_name: str) -> Star:
        if isinstance(R, list):
            if len(R) == 0:
                raise RuntimeError(f"Operator '{op_name}' returned an empty list")
            if len(R) > 1:
                raise NotImplementedError(
                    f"Operator '{op_name}' returned a union of {len(R)} Stars. "
                    'Step 1 executor currently expects a single Star result.'
                )
            return R[0]
        return R

    @staticmethod
    def _as_star_list(R) -> list[Star]:
        if isinstance(R, list):
            for i, S in enumerate(R):
                if not isinstance(S, Star):
                    raise TypeError(f'List output contains non-Star element at index {i}')
            return R
        if not isinstance(R, Star):
            raise TypeError('Operator output must be a Star or list[Star]')
        return [R]

    def _run_sin(self, S: Star, src_dim: int, params: dict) -> Star:
        split = bool(params.get('split', False))
        max_splits = int(params.get('max_splits', 0))

        R = SinLayer().reach(
            S,
            idx=src_dim,
            method='approx',
            lp_solver=self.lp_solver,
            RF=self.RF,
            split=split,
            max_splits=max_splits,
        )
        return R

    def _run_cos(self, S: Star, src_dim: int, params: dict) -> Star:
        split = bool(params.get('split', False))
        max_splits = int(params.get('max_splits', 0))

        R = CosLayer().reach(
            S,
            idx=src_dim,
            method='approx',
            lp_solver=self.lp_solver,
            RF=self.RF,
            split=split,
            max_splits=max_splits,
        )
        return R

    def _run_pow_even(self, S: Star, src_dim: int, params: dict) -> Star | list[Star]:
        n = int(params.get('n', 2))
        if n <= 0 or n % 2 != 0:
            raise ValueError(f"pow_even requires positive even n, got {n}")

        split = bool(params.get('split', False))
        max_splits = int(params.get('max_splits', 0))

        R = PowerEvenOperater().reach(
            S,
            n=n,
            idx=src_dim,
            method='approx',
            lp_solver=self.lp_solver,
            RF=self.RF,
            split=split,
            max_splits=max_splits,
        )
        return R

    def _run_pow_odd(self, S: Star, src_dim: int, params: dict) -> Star | list[Star]:
        n = int(params.get('n', 1))
        if n <= 0 or n % 2 != 1:
            raise ValueError(f"pow_odd requires positive odd n, got {n}")

        split = bool(params.get('split', False))
        max_splits = int(params.get('max_splits', 0))

        R = PowerOddOperater().reach(
            S,
            n=n,
            idx=src_dim,
            method='approx',
            lp_solver=self.lp_solver,
            RF=self.RF,
            split=split,
            max_splits=max_splits,
        )
        return R

    def _run_mul(self, S: Star, x_dim: int, y_dim: int, params: dict) -> Star | list[Star]:
        split = bool(params.get('split', False))
        max_splits = int(params.get('max_splits', 0))

        R = MultiplyOperater.reach(
            S,
            idx_x=x_dim,
            idx_y=y_dim,
            lp_solver=self.lp_solver,
            RF=self.RF,
            split=split,
            max_splits=max_splits,
        )
        return R


if __name__ == '__main__':
    from StarV.lifting.exprgraph import ExpressionGraph

    lb = np.array([-1.0, -0.5])
    ub = np.array([1.0, 0.5])
    I0 = Star(lb, ub)

    g = ExpressionGraph()
    g.add_var('x', 'x')
    g.add_var('theta', 'theta')
    g.add_node('x2', op='pow_even', inputs=['x'], params={'n': 2})
    g.add_node('s', op='sin', inputs=['theta'])
    g.add_node('z', op='mul', inputs=['x2', 's'])
    g.set_outputs(['z'])

    executor = ExpressionExecutor(lp_solver='gurobi', RF=0.0)
    R, meta = executor.run(I0, g, var_dims={'x': 0, 'theta': 1})

    if isinstance(R, list):
        print('Num output Stars:', len(R))
        print('Output Star dims:', [Si.dim for Si in R])
    else:
        print('Num output Stars:', 1)
        print('Output Star dim:', R.dim)
    print('Node -> dim mapping:', meta['node_to_dim'])
