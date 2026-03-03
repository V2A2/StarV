"""
Lift executor: execute ExprNodes in topo order and build lifted Star.

Author: Zhuoyang Zhou
Date: 02/18/2026  Updated: 03/01/2026
"""

import numpy as np

from StarV.set.star import Star
from StarV.dynamic.sine import Sine
from StarV.dynamic.cosine import Cosine
from StarV.dynamic.powereven import PowerEven
from StarV.dynamic.powerodd import PowerOdd
from StarV.dynamic.multiply import Multiply


class LiftExecutor:

    def __init__(self, lpSolver="gurobi", RF=0.0, verbose=False):
        self.lpSolver = lpSolver
        self.RF = RF
        self.verbose = verbose

    # ============================================================
    # Main API
    # ============================================================

    def run(self, initialStar, orderedNodes, varIndexMap):

        if not isinstance(initialStar, Star):
            raise ValueError("initialStar must be a Star")

        stars = [initialStar]   # allow union
        idToIndex = {}

        for node in orderedNodes:

            op = node.op

            if self.verbose:
                print("Executing:", node.id, "| op =", op)

            # ----------------------------------------------------
            # var
            # ----------------------------------------------------
            if op == "var":
                idx = self.resolveVar(node, varIndexMap)
                node.outIndex = idx
                idToIndex[node.id] = idx
                continue

            # ----------------------------------------------------
            # const
            # ----------------------------------------------------
            if op == "const":
                value = float(node.params["value"])
                newStars = []
                for S in stars:
                    newStars.append(self.appendConst(S, value))
                stars = newStars
                node.outIndex = stars[0].dim - 1
                idToIndex[node.id] = node.outIndex
                continue

            # resolve input indices
            inputIdx = [idToIndex[i] for i in node.inputs]

            # ----------------------------------------------------
            # affine ops
            # ----------------------------------------------------
            if op == "neg":
                stars = [self.appendNeg(S, inputIdx[0]) for S in stars]
                node.outIndex = stars[0].dim - 1
                idToIndex[node.id] = node.outIndex
                continue

            if op == "add":
                stars = [self.appendAdd(S, inputIdx[0], inputIdx[1]) for S in stars]
                node.outIndex = stars[0].dim - 1
                idToIndex[node.id] = node.outIndex
                continue

            if op == "sub":
                stars = [self.appendSub(S, inputIdx[0], inputIdx[1]) for S in stars]
                node.outIndex = stars[0].dim - 1
                idToIndex[node.id] = node.outIndex
                continue

            # ----------------------------------------------------
            # nonlinear unary
            # ----------------------------------------------------
            if op in ("sin", "cos", "powEven", "powOdd"):

                newStars = []
                for S in stars:
                    newStars.extend(
                        self.appendNonlinearUnary(S, op, inputIdx[0], node.params)
                    )

                stars = newStars
                node.outIndex = stars[0].dim - 1
                idToIndex[node.id] = node.outIndex
                continue

            # ----------------------------------------------------
            # multiply
            # ----------------------------------------------------
            if op == "mul":

                newStars = []
                for S in stars:
                    R = Multiply.reachApprox_star(
                        S,
                        idx_x=inputIdx[0],
                        idx_y=inputIdx[1],
                        lp_solver=self.lpSolver,
                        RF=self.RF,
                        split=False,
                    )
                    newStars.append(R)

                stars = newStars
                node.outIndex = stars[0].dim - 1
                idToIndex[node.id] = node.outIndex
                continue

            raise ValueError("Unsupported op: {}".format(op))

        if len(stars) == 1:
            return stars[0], idToIndex
        return stars, idToIndex

    # ============================================================
    # Basic helpers
    # ============================================================

    def resolveVar(self, node, varIndexMap):
        name = node.params.get("name", node.id)
        if name not in varIndexMap:
            raise ValueError("Variable '{}' not found in varIndexMap".format(name))
        return varIndexMap[name]

    def appendConst(self, S, value):

        V = S.V
        newV = np.zeros((S.dim + 1, V.shape[1]))
        newV[:-1, :] = V
        newV[-1, 0] = value

        return Star(newV, S.C, S.d, S.pred_lb, S.pred_ub)

    def appendNeg(self, S, idx):

        V = S.V
        newV = np.zeros((S.dim + 1, V.shape[1]))
        newV[:-1, :] = V
        newV[-1, :] = -V[idx, :]

        return Star(newV, S.C, S.d, S.pred_lb, S.pred_ub)

    def appendAdd(self, S, idxA, idxB):

        V = S.V
        newV = np.zeros((S.dim + 1, V.shape[1]))
        newV[:-1, :] = V
        newV[-1, :] = V[idxA, :] + V[idxB, :]

        return Star(newV, S.C, S.d, S.pred_lb, S.pred_ub)

    def appendSub(self, S, idxA, idxB):

        V = S.V
        newV = np.zeros((S.dim + 1, V.shape[1]))
        newV[:-1, :] = V
        newV[-1, :] = V[idxA, :] - V[idxB, :]

        return Star(newV, S.C, S.d, S.pred_lb, S.pred_ub)

    # ============================================================
    # Nonlinear unary handling
    # ============================================================

    def appendNonlinearUnary(self, S, op, idx, params):

        # extract 1D projection (reuse predicates)
        V1 = S.V[idx:idx + 1, :]
        S1 = Star(V1, S.C, S.d, S.pred_lb, S.pred_ub)

        if op == "sin":
            R = Sine.reachApprox_star(S1, lp_solver=self.lpSolver, RF=self.RF)
        elif op == "cos":
            R = Cosine.reachApprox_star(S1, lp_solver=self.lpSolver, RF=self.RF)
        elif op == "powEven":
            R = PowerEven.reachApprox_star(
                S1, n=params["n"], lp_solver=self.lpSolver, RF=self.RF
            )
        elif op == "powOdd":
            R = PowerOdd.reachApprox_star(
                S1, n=params["n"], lp_solver=self.lpSolver, RF=self.RF
            )
        else:
            raise ValueError("Unsupported unary op")

        if not isinstance(R, list):
            R = [R]

        result = []
        for R1 in R:
            result.append(self.mergeResult(S, R1))

        return result

    def mergeResult(self, S, R1):

        oldDim = S.dim
        oldVars = S.nVars
        newVars = R1.nVars

        newV = np.zeros((oldDim + 1, newVars + 1))
        newV[:-1, :oldVars + 1] = S.V
        newV[-1, :] = R1.V[0, :]

        return Star(newV, R1.C, R1.d, R1.pred_lb, R1.pred_ub)