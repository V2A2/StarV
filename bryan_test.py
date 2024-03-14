"""
Original Star set class from StarV
"""

import numpy as np
from typing import Tuple
from StarV.set.star import Star
from bryan_star import Star as nStar


def test_star_methods(S: Star) -> None:
    """
    Test various methods of the original Star class
    """

    print("---------test_star_methods()---------")

    print("_str__()")
    print(S.__str__())

    print("getMax()")
    print(S.getMax(0))

    print("getMin()")
    print(S.getMin(0))

    print("getRanges()")
    print(S.getRanges())

    print("getMinimizedConstraints()")
    print(S.getMinimizedConstraints())

    print("estimateRange()")
    print(S.estimateRange(0))

    print("estimateRanges()")
    print(S.estimateRanges())

    print("affineMap()")
    print(S.affineMap(np.array([[1, 0], [0, 1]]), np.array([0, 0])))

    print("minKowskiSum()")
    Y = Star.rand(2)
    print(S.minKowskiSum(Y))

    print("isEmptySet()")
    print(S.isEmptySet())

    print("addConstraint()")
    print(S.addConstraint(np.array([1, 0]), np.array([0])))

    print("addMultipleConstraints()")
    print(S.addMultipleConstraints(np.array([[1, 0], [0, 1]]), np.array([0, 0])))

    print("resetRow()")
    print(S.resetRow(0))

    print("rand()")
    print(S.rand(3))


def construct_star_sets() -> Tuple[Star, nStar]:
    """
    Construct Star sets
    """
    lb = np.array([0, 0])
    ub = np.array([1, 1])
    S1 = Star(lb, ub)
    S2 = nStar(lb, ub)

    # V = np.array([[0, 1, 0], [0, 0, 1]])
    # C = np.array([[1, -1], [2, 1]])
    # d = np.array([3, 0])
    # pred_lb = np.array([-5, -2])
    # pred_ub = np.array([1, 9])
    # S1 = Star(V, C, d, pred_lb, pred_ub)
    # S2 = nStar(V, C, d, pred_lb, pred_ub)

    print("---------construct()---------")
    return S1, S2


def test_affine_map(S1: Star, S2: nStar) -> None:
    """
    Test affineMap() method
    """
    W = np.array([[1, 1], [2, 3]])
    b = np.array([0, 0])
    T1 = S1.affineMap(W, b)
    T2 = S2.affineMap(W, b)
    print("---------AffineMap()---------")
    print(T1)
    print(T2)


def test_get_min(S1: Star, S2: nStar) -> None:
    """
    Test getMin() method
    """
    min1 = S1.getMin(1, lp_solver="linprog")
    min2 = S2.getMin(1, lp_solver="linprog")
    print("-------------------")
    print("original Starset {} \nmin = {}\n".format(S1, min1))
    print("new Starset {} \nmin = {}\n".format(S2, min2))


def main():
    """
    Construct two sets of stars from two Star classes,
    and then passes those sets to the testing functions for comparision.
    """
    S1, S2 = construct_star_sets()
    test_star_methods(S1)
    test_affine_map(S1, S2)


if __name__ == "__main__":
    main()
