"""
RecurrentLayer layer class
Bryan Duong, 6/27/2024
"""

import numpy as np
import multiprocessing
from StarV.set.probstar import ProbStar
from StarV.layer.ReLULayer import ReLULayer
from StarV.layer.LeakyReLULayer import LeakyReLULayer
from StarV.layer.SatLinLayer import SatLinLayer
from StarV.layer.SatLinsLayer import SatLinsLayer
from StarV.layer.fullyConnectedLayer import fullyConnectedLayer


class RecurrentLayer(object):

    def __init__(
        self,
        Whh: np.ndarray,
        bh: np.ndarray,
        Whx: np.ndarray,
        Woh: np.ndarray,
        bo: np.ndarray,
    ) -> None:
        """
        Initialize the RecurrentLayer object.

        Args:
            Whh: Weight matrix for hidden state to hidden state connections.
            bh: Bias vector for hidden state.
            Whx: Weight matrix for input to hidden state connections.
            Woh: Weight matrix for hidden state to output connections.
            bo: Bias vector for output.

        Returns:
            None
        """

        assert isinstance(Whh, np.ndarray), "Weight matrix should be a numpy array"
        assert isinstance(bh, np.ndarray), "Bias vector should be a numpy array"
        assert isinstance(Whx, np.ndarray), "Weight matrix should be a numpy array"
        assert isinstance(Woh, np.ndarray), "Weight matrix should be a numpy array"
        assert isinstance(bo, np.ndarray), "Bias vector should be a numpy array"

        self.Whh = Whh
        self.bh = bh
        self.Whx = Whx
        self.Woh = Woh
        self.bo = bo

    def evaluate(self, x: ProbStar, step: int) -> list:
        """
        Evaluates the RecurrentLayer model for a given input and number of steps.

        Args:
            x (ProbStar): The input Probstar.
            step (int): The number of steps to evaluate.

        Returns:
            list: The list of outputs for each step.
        """
        if not isinstance(x, ProbStar):
            raise Exception(
                "error: input is not a ProbStar set, type of input = {}".format(type(x))
            )
        if not isinstance(step, int):
            raise Exception(
                "error: step is not an integer, type of step = {}".format(type(step))
            )
        output = []
        for i in step:
            """
            h = Whh @ h + Whx @ x + bh
            x = Woh @ h + bo
            """
            h = h.affineMap(self.Whh, self.bh) + x.affineMap(self.Whx)
            o = h.affineMap(self.Woh, self.bo)
            output.append(ReLULayer.evaluate(o))
        return output

    @staticmethod
    def rand(in_dim: int, out_dim: int) -> "RecurrentLayer":
        """
        Generate a random RecurrentLayer model.

        Args:
            in_dim (int): The input dimension.
            out_dim (int): The output dimension.

        Returns:
            RecurrentLayer: The random RecurrentLayer model.
        """
        Whh = np.random.rand(out_dim, out_dim)
        bh = np.random.rand(out_dim)
        Whx = np.random.rand(out_dim, in_dim)
        Woh = np.random.rand(out_dim, out_dim)
        bo = np.random.rand(out_dim)
        return RecurrentLayer(Whh, bh, Whx, Woh, bo)

    def exactReach(self, input_set: list, func="ReLU", lp_solver="gurobi") -> list:
        """
        Reachability analysis of RecurrentLayer model using exact method

        Args:
            input_set (list): The list of input sets.
            lp_solver (str): The linear programming solver to use.

        Returns:
            list: The list of reachable set.
        """
        hidden_set = []
        output_set = []

        for i, input in enumerate(input_set):
            if i == 0:
                hidden = input.affineMap(self.Whx, self.bh)
                hidden_set_current = ReLULayer.reach(
                    [hidden], method="exact", lp_solver=lp_solver
                )
            else:
                hidden_set_current = []
                for hidden_input in hidden_set_previous:
                    hj_affn = hidden_input.affineMap(self.Whh)
                    xi_affn = input.affineMap(self.Whx, self.bh)
                    hj_xi = hj_affn.minKowskiSum(xi_affn)

                    if func == "ReLU":
                        hj_reach = ReLULayer.reach(
                            [hj_xi], method="exact", lp_solver=lp_solver
                        )
                    elif func == "LeakyReLU":
                        hj_reach = LeakyReLULayer.reach(
                            [hj_xi], method="exact", lp_solver=lp_solver
                        )
                    elif func == "SatLin":
                        hj_reach = SatLinLayer.reach(
                            [hj_xi], method="exact", lp_solver=lp_solver
                        )
                    elif func == "Satlins":
                        hj_reach = SatLinsLayer.reach(
                            [hj_xi], method="exact", lp_solver=lp_solver
                        )
                    elif func == "fullyConnected":
                        hj_reach = fullyConnectedLayer.reach(
                            [hj_xi], method="exact", lp_solver=lp_solver
                        )

                    hidden_set_current.extend(hj_reach)

            hidden_set.append(hidden_set_current)
            hidden_set_previous = hidden_set_current

        output_set = [[] for _ in range(len(hidden_set))]
        for i, layer in enumerate(hidden_set):
            for hidden in layer:
                o = hidden.affineMap(self.Woh, self.bo)
                output_set[i].extend(
                    ReLULayer.reach([o], method="exact", lp_solver=lp_solver)
                )

        return output_set

    def reach(
        self,
        input_set: list,
        method="exact",
        func="ReLU",
        lp_solver="gurobi",
        pool=None,
        RF=0.0,
    ) -> list:
        """
        Reachability analysis of RecurrentLayer

        Args:
            input_set (list): The list of input sets.
            method (str): The reachability analysis method to use. Default is "exact".
            lp_solver (str): The linear programming solver to use. Default is "gurobi".
            pool (multiprocessing.Pool): The multiprocessing pool to use. Default is None.
            RF (float): The relaxation factor. Default is 0.0.

        Returns:
            list: The list of reachable sets.
        """
        if method == "exact":
            output_set = self.exactReach(input_set, func, lp_solver)
        elif method == "relax":
            output_set = self.reachRelax(input_set, func, lp_solver, RF)
        else:
            raise ValueError("Invalid reachability analysis method: {}".format(method))
        return output_set

    def reachRelax(self, input_set: list, lp_solver="gurobi", RF=0.2) -> list:
        pass
