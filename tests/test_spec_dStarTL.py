"""
Test dStarTL module: ExpandedFormula, GetRobustnessInterval,
GetSatisfactionFraction and GetSatisfactionFraction_for_RNN
Author: Qing Liu
Date: 10/29/2026
"""

import numpy as np
from StarV.set.star import Star
from StarV.spec.dProbStarTL import (
    _ALWAYS_, _EVENTUALLY_, _UNTIL_, AtomicPredicate, Formula,
    _LeftBracket_, _RightBracket_, _AND_, _OR_,
)
from StarV.spec.dStarTL import (
    ExpandedFormula, GetRobustnessInterval, GetSatisfactionFraction,
    GetSatisfactionFraction_for_RNN,
)


# Operators and predicates from the examples in StarV/spec/dStarTL.py
EVOT = _EVENTUALLY_(0, 1)
EV12 = _EVENTUALLY_(1, 2)
AWOT = _ALWAYS_(0, 1)
AW03 = _ALWAYS_(0, 3)
lb = _LeftBracket_()
rb = _RightBracket_()
AND = _AND_()
OR = _OR_()
UNTIL = _UNTIL_(2, 3)
P1 = AtomicPredicate(np.array([1.0, 0.0]), np.array([0.05]))
P2 = AtomicPredicate(np.array([0.0, 1.0]), np.array([0.02]))
P3 = AtomicPredicate(np.array([-1.0, 0.0]), np.array([0.01]))


def reachable_sets():
    """Example 1: reachable set sequence; every R[t] uses alpha in [-1, 1]^2."""
    X0 = Star(np.array([0.0, 0.0]), np.array([0.4, 0.2]))
    X1 = Star(np.array([0.5, -0.1]), np.array([1.0, 0.1]))
    X2 = Star(np.array([1.0, 0.0]), np.array([1.6, 0.3]))
    X3 = Star(np.array([1.5, -0.2]), np.array([1.8, 0.2]))
    return [X0, X1, X2, X3]


def example_specs():
    """Specifications of Examples 2 to 6."""
    P2_safe = AtomicPredicate(np.array([1.0, 0.0]), np.array([2.0]))
    P3_x = AtomicPredicate(np.array([1.0, 0.0]), np.array([-0.5]))
    P3_y = AtomicPredicate(np.array([0.0, 1.0]), np.array([1.0]))
    P_x = AtomicPredicate(np.array([1.0, 0.0]), np.array([0.75]))
    P_y = AtomicPredicate(np.array([0.0, 1.0]), np.array([0.05]))
    P6_x1 = AtomicPredicate(np.array([1.0, 0.0]), np.array([0.10]))
    return {
        # ALWAYS_[0,3] (x <= 2.0)
        2: Formula([AW03, lb, P2_safe, rb]),
        # EVENTUALLY_[0,1] (x <= -0.5 AND ALWAYS_[0,1] (y <= 1.0))
        3: Formula([EVOT, lb, P3_x, AND, lb, AWOT, P3_y, rb, rb]),
        # EVENTUALLY_[1,2] (x <= 0.75)
        4: Formula([EV12, lb, P_x, rb]),
        # EVENTUALLY_[0,1] (y <= 0.05 AND EVENTUALLY_[1,2] (x <= 0.75))
        5: Formula([EVOT, lb, P_y, AND, lb, EV12, lb, P_x, rb, rb, rb]),
        # EVENTUALLY_[0,1] ((y <= 0.05 AND EVENTUALLY_[1,2] (x <= 0.75)) OR x <= 0.10)
        6: Formula([EVOT, lb, lb, P_y, AND, lb, EV12, lb, P_x, rb, rb, rb, OR, P6_x1, rb]),
    }


def times(expr):
    """Return the times of every atomic predicate in an expanded tree."""
    if expr[0] == 'AP':
        return [expr[1].t]
    return [t for sub_expr in expr[1] for t in times(sub_expr)]


class Test(object):
    """
    Testing dStarTL classes and methods
    """

    def __init__(self):

        self.n_fails = 0
        self.n_tests = 0

    def run(self, name, check):
        self.n_tests = self.n_tests + 1
        print('\nTest {}\n'.format(name))
        try:
            check()
        except Exception as error:
            print('Test Fail! {}: {}'.format(type(error).__name__, error))
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')

    def test_expanded_formula(self):

        def check():
            # EVENTUALLY_[0,1] (P1 OR ALWAYS_[0,1] P2)
            #   = (P1[0] OR (P2[0] AND P2[1])) OR (P1[1] OR (P2[1] AND P2[2]))
            spec1 = Formula([EVOT, lb, P1, OR, lb, AWOT, P2, rb, rb])
            spec1.print()
            Ex_1 = ExpandedFormula(spec1)
            Ex_1.print()
            assert Ex_1.outer_operator == 'OR'
            assert Ex_1.expr[0] == 'OR' and times(Ex_1.expr) == [0, 0, 1, 1, 1, 2]

            # EVENTUALLY_[0,1] (P1 AND (P2 UNTIL_[2,3] P3)), UNTIL is an OR over
            # the witness times 2 and 3: (P2[0] AND P2[1] AND P3[2]) OR (... AND P3[3])
            spec2 = Formula([EVOT, lb, P1, AND, lb, P2, UNTIL, P3, rb, rb])
            spec2.print()
            Ex_2 = ExpandedFormula(spec2)
            Ex_2.print()
            first = Ex_2.expr[1][0]
            assert first[0] == 'AND' and first[1][0][1].t == 0
            until = first[1][1]
            assert until[0] == 'OR' and times(until[1][0]) == [0, 1, 2]
            assert times(until[1][1]) == [0, 1, 2, 3] and max(times(Ex_2.expr)) == 4

            # Example 2: ALWAYS_[0,3] -> AND over t = 0, 1, 2, 3
            Ex_P2 = ExpandedFormula(example_specs()[2])
            assert Ex_P2.outer_operator == 'AND'
            assert Ex_P2.expr[0] == 'AND' and times(Ex_P2.expr) == [0, 1, 2, 3]

            # Example 5 with T=4: x[t=3] is kept, nothing beyond the horizon
            Ex_P5 = ExpandedFormula(example_specs()[5], T=4)
            Ex_P5.print()
            assert sorted(set(times(Ex_P5.expr))) == [0, 1, 2, 3]
            # With T=2, EVENTUALLY_[1,2] keeps only t=1
            assert times(ExpandedFormula(example_specs()[4], T=2).expr) == [1]

        self.run('ExpandedFormula', check)

    def test_robustness_interval_formula(self):

        def check():
            R = reachable_sets()
            specs = example_specs()
            RI = GetRobustnessInterval(R, lp_solver='linprog')

            # Example 2: ALWAYS_[0,3] (x <= 2.0) is satisfied, [0.2, 0.5]
            rho_lb, rho_ub = RI.getRobustnessInterval(ExpandedFormula(specs[2]))
            print('Always: ==> rho_lb = {}, rho_ub = {}'.format(rho_lb, rho_ub))
            assert np.isclose(rho_lb, 0.2) and np.isclose(rho_ub, 0.5)

            # Example 3: nested formula is violated, [-0.9, -0.5]
            rho_lb, rho_ub = RI.getRobustnessInterval(ExpandedFormula(specs[3]))
            print('Nested: ==> rho_lb = {}, rho_ub = {}'.format(rho_lb, rho_ub))
            assert np.isclose(rho_lb, -0.9) and np.isclose(rho_ub, -0.5)

            # Examples 4 and 5: mixed cases, the interval contains 0
            rho_lb, rho_ub = RI.getRobustnessInterval(ExpandedFormula(specs[4], T=len(R)))
            assert np.isclose(rho_lb, -0.25) and np.isclose(rho_ub, 0.25)
            rho_lb, rho_ub = RI.getRobustnessInterval(ExpandedFormula(specs[5], T=len(R)))
            assert np.isclose(rho_lb, -0.25) and np.isclose(rho_ub, 0.05)

        self.run('GetRobustnessInterval', check)

    def test_satisfaction_fraction_exact_and_sampling(self):

        def check():
            R = reachable_sets()
            specs = example_specs()
            # Example 2 and 3: the robustness interval decides, no volume needed
            for example, fraction in ((2, 1.0), (3, 0.0)):
                result = GetSatisfactionFraction(
                    R, ExpandedFormula(specs[example], T=len(R))).getSatisfactionFraction()
                assert result['method'] == 'robustness' and result['satisfying_fraction'] == fraction

            # Examples 4, 5 and 6: mixed cases, exact volume fraction
            for example, fraction in ((4, 0.5), (5, 0.125), (6, 0.3125)):
                Ex_F = ExpandedFormula(specs[example], T=len(R))
                result = GetSatisfactionFraction(
                    R, Ex_F, method='exact', lp_solver='linprog').getSatisfactionFraction()
                print('Example {} exact satisfaction fraction: {}'.format(example, result))
                assert result['method'] == 'exact'
                assert np.isclose(result['satisfying_fraction'], fraction)

            # Example 6: sampling should be close to the exact fraction 0.3125
            sampling = GetSatisfactionFraction(
                R, ExpandedFormula(specs[6], T=len(R)), method='sampling',
                num_samples=20000, seed=1, lp_solver='linprog').getSatisfactionFraction()
            print('Example 6 sampling satisfaction fraction: {}'.format(sampling))
            assert sampling['method'] == 'sampling'
            assert sampling['n_satisfied'] + sampling['n_violating'] == 20000
            assert abs(sampling['satisfying_fraction'] - 0.3125) < 0.02

        self.run('GetSatisfactionFraction exact and sampling', check)

    def test_exact_robustness_milp(self):

        def check():
            R = reachable_sets()
            specs = example_specs()

            # Example 6: the MILP interval lies inside the interval-arithmetic one
            Ex_P6 = ExpandedFormula(specs[6], T=len(R))
            rho_lb, rho_ub = GetRobustnessInterval(R).getRobustnessInterval(Ex_P6)
            SAT_6 = GetSatisfactionFraction(R, Ex_P6, compute_exact_robustness=True)
            exact_lb, exact_ub = SAT_6.computeExactRobustnessInterval()
            print('Example 6: interval [{}, {}], MILP [{}, {}]'.format(rho_lb, rho_ub, exact_lb, exact_ub))
            assert rho_lb - 1e-9 <= exact_lb <= exact_ub <= rho_ub + 1e-9

            # (x[1] <= 0.75) OR (x[1] >= 0.75): the interval [-0.25, 0.25] is mixed,
            # but max(0.75 - x, x - 0.75) is never negative, so the MILP proves it.
            P_le = AtomicPredicate(np.array([1.0, 0.0]), np.array([0.75]), t=1)
            P_ge = AtomicPredicate(np.array([-1.0, 0.0]), np.array([-0.75]), t=1)
            Ex_F = ExpandedFormula(Formula([P_le, OR, P_ge]), T=len(R))
            assert np.allclose(GetRobustnessInterval(R).getRobustnessInterval(Ex_F), (-0.25, 0.25))
            SAT = GetSatisfactionFraction(R, Ex_F, compute_exact_robustness=True)
            exact_lb, exact_ub = SAT.computeExactRobustnessInterval()
            assert np.isclose(exact_lb, 0.0, atol=1e-7) and np.isclose(exact_ub, 0.25)
            result = SAT.getSatisfactionFraction()
            assert result['method'] == 'exact_robustness' and result['satisfying_fraction'] == 1.0

        self.run('Refined robustness interval by MILP', check)

    def test_chebyshev_region_balls(self):

        def check():
            # Example 4: x[1] <= 0.75 holds on alpha1 <= 0 and fails on alpha1 > 0.
            # The largest ball in each half of [-1,1]^2 has radius 0.5.
            R = reachable_sets()
            Ex_P4 = ExpandedFormula(example_specs()[4], T=len(R))
            SAT = GetSatisfactionFraction(R, Ex_P4, compute_region_balls=True)
            balls = SAT.computeChebyshevRegionBounds()
            print('Example 4 region balls: {}'.format(balls))
            for prefix in ('sat', 'viol'):
                assert balls[prefix + '_ball_found']
                assert np.isclose(balls[prefix + '_ball_radius'], 0.5)
                # ball area / box area = pi * 0.5^2 / 4
                assert np.isclose(balls[prefix + '_ball_fraction_lb'], np.pi * 0.25 / 4.0)
            assert SAT.shouldComputeRegionBalls(dict(rho_lb=-0.25, rho_ub=0.25, satisfying_fraction=0.0))
            assert not SAT.shouldComputeRegionBalls(dict(rho_lb=-0.25, rho_ub=0.25, satisfying_fraction=0.5))

        self.run('Chebyshev region balls', check)

    def rnn_trace(self):
        # R0: x = alpha1 (1 predicate variable); R1: x = alpha1 + alpha2 (2 variables)
        R0 = Star(np.array([[0.0, 1.0]]), np.array([]), np.array([]), -np.ones(1), np.ones(1))
        R1 = Star(np.array([[0.0, 1.0, 1.0]]), np.array([]), np.array([]), -np.ones(2), np.ones(2))
        return [R0, R1]

    def test_rnn_shared_predicate_trace(self):

        def check():
            R = self.rnn_trace()
            P_x = AtomicPredicate(np.array([1.0]), np.array([0.0]))
            SAT = GetSatisfactionFraction_for_RNN(R, ExpandedFormula(Formula([EVOT, lb, P_x, rb]), T=2))
            assert SAT.getBaseStar() is R[1]
            shared = SAT.constructSharedPredicateTrace()
            assert [S.nVars for S in shared] == [2, 2]
            assert np.allclose(shared[0].V, [[0.0, 1.0, 0.0]])
            try:
                SAT.padStarToSharedPredicateSpace(R[1], R[0], 1)
            except RuntimeError:
                pass
            else:
                raise AssertionError('padding to fewer variables was accepted')

        self.run('GetSatisfactionFraction_for_RNN shared predicate trace', check)

    def test_rnn_satisfaction_fraction(self):

        def check():
            # EVENTUALLY_[0,1] (x <= 0): alpha1 <= 0 OR alpha1 + alpha2 <= 0
            #   area = 2 + 2 - 1.5 = 2.5 of 4 -> 0.625
            R = self.rnn_trace()
            P_x = AtomicPredicate(np.array([1.0]), np.array([0.0]))
            Ex_F = ExpandedFormula(Formula([EVOT, lb, P_x, rb]), T=2)
            exact = GetSatisfactionFraction_for_RNN(R, Ex_F, method='exact').getRNNSatisfactionFraction()
            print('RNN exact: {}'.format(exact))
            assert np.isclose(exact['rho_lb'], -1.0) and np.isclose(exact['rho_ub'], 2.0)
            assert np.isclose(exact['satisfying_fraction'], 0.625)
            sampling = GetSatisfactionFraction_for_RNN(
                R, Ex_F, method='sampling', num_samples=20000, seed=1,
                compute_region_balls=True).getRNNSatisfactionFraction()
            assert abs(sampling['satisfying_fraction'] - 0.625) < 0.02
            assert sampling['sat_ball_found'] and sampling['viol_ball_found']
            # satisfied and violated traces stop at the robustness interval
            P_true = AtomicPredicate(np.array([1.0]), np.array([5.0]))
            P_false = AtomicPredicate(np.array([1.0]), np.array([-5.0]))
            sat = GetSatisfactionFraction_for_RNN(R, ExpandedFormula(Formula([AWOT, lb, P_true, rb]), T=2))
            viol = GetSatisfactionFraction_for_RNN(R, ExpandedFormula(Formula([AWOT, lb, P_false, rb]), T=2))
            assert sat.getRNNSatisfactionFraction()['satisfying_fraction'] == 1.0
            assert viol.getRNNSatisfactionFraction()['satisfying_fraction'] == 0.0

        self.run('GetSatisfactionFraction_for_RNN satisfaction fraction', check)


if __name__ == "__main__":

    test_dStarTL = Test()
    print('\n=======================\
    ================================\
    ================================\
    ===============================\n')
    test_dStarTL.test_expanded_formula()
    test_dStarTL.test_robustness_interval_formula()
    test_dStarTL.test_satisfaction_fraction_exact_and_sampling()
    test_dStarTL.test_exact_robustness_milp()
    test_dStarTL.test_chebyshev_region_balls()
    test_dStarTL.test_rnn_shared_predicate_trace()
    test_dStarTL.test_rnn_satisfaction_fraction()
    print('\n========================\
    =================================\
    =================================\
    =================================\n')
    print('Testing dStarTL: fails: {}, successfull: {}, \
    total tests: {}'.format(test_dStarTL.n_fails,
                            test_dStarTL.n_tests - test_dStarTL.n_fails,
                            test_dStarTL.n_tests))
