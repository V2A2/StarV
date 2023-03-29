import numpy as np

class Polyhedron(object):
    """
    Polyhedron in the form {x in R^n | A x <= b, C x = d}.
    Author: Michael Ivashchenko
    """

    def __init__(self, A, b, C=None, d=None):
        """
        Instantiates the polyhedron.

        Arguments
        ----------
        A : numpy.ndarray
            Left-hand side of the inequalities.
        b : numpy.ndarray
            Right-hand side of the inequalities.
        C : numpy.ndarray
            Left-hand side of the equalities.
        d : numpy.ndarray
            Right-hand side of the equalities.
        """

        # check and store inequalities
        if len(b.shape) > 1:
            raise ValueError('b must be a one dimensional array.')
        self._same_number_rows(A, b)
        self.A = A
        self.b = b

        # check and store equalities
        if (C is None) != (d is None):
            raise ValueError('missing C or d.')
        if C is None:
            self.C = np.zeros((0, A.shape[1]))
            self.d = np.zeros(0)
        else:
            if len(d.shape) > 1:
                raise ValueError('b must be a one dimensional array.')
            self._same_number_rows(C, d)
            self.C = C
            self.d = d

        # initializes the attributes to None
        self._empty = None
        self._bounded = None
        self._radius = None
        self._center = None
        self._vertices = None

    def add_inequality(self, A, b, indices=None):
        """
        Adds the inequality A x[indices] <= b to the existing polyhedron.

        Arguments
        ----------
        A : numpy.ndarray
            Left-hand side of the inequalities.
        b : numpy.ndarray
            Right-hand side of the inequalities.
        indices : list of int
            Set of indices of elements of x to which the inequality applies.
        """

        # check inequalities
        self._same_number_rows(A, b)
    
        # reset attributes to None
        self._delete_attributes()

        # add inequalities
        S = self._selection_matrix(indices)
        self.A = np.vstack((self.A, A.dot(S)))
        self.b = np.concatenate((self.b, b))

    def add_lower_bound(self, x_min, indices=None):
        """
        Adds the inequality x[indices] >= x_min to the existing polyhedron.
        If indices is None, the inequality is applied to all the elements of x.

        Arguments
        ----------
        x_min : numpy.ndarray
            Lower bound on the elements of x.
        indices : list of int
            Set of indices of elements of x to which the inequality applies.
        """

        # if x_min is a float make it an array
        if isinstance(x_min, float):
            x_min = np.array([x_min])

        # add the constraint - S x <= - x_min, with S selection matrix
        S = self._selection_matrix(indices)
        self.add_inequality(-S, -x_min)

    def add_upper_bound(self, x_max, indices=None):
        """
        Adds the inequality x[indices] <= x_max to the existing polyhedron.
        If indices is None, the inequality is applied to all the elements of x.

        Arguments
        ----------
        x_max : numpy.ndarray
            Upper bound on the elements of x.
        indices : list of int
            Set of indices of elements of x to which the inequality applies.
        """

        # if x_max is a float make it a 2d array
        if isinstance(x_max, float):
            x_max = np.array([x_max])

        # add the constraint S x <= x_max, with S selection matrix
        S = self._selection_matrix(indices)
        self.add_inequality(S, x_max)

    def add_bounds(self, x_min, x_max, indices=None):
        """
        Adds the inequalities x_min <= x[indices] <= x_max to the existing polyhedron.
        If indices is None, the inequality is applied to all the elements of x.

        Arguments
        ----------
        x_min : numpy.ndarray
            Lower bound on the elements of x.
        x_max : numpy.ndarray
            Upper bound on the elements of x.
        indices : list of int
            Set of indices of elements of x to which the inequality applies.
        """

        self.add_lower_bound(x_min, indices)
        self.add_upper_bound(x_max, indices)

    def _delete_attributes(self):
        """
        Resets al the attibutes of the class to None.
        """

        # reser the attributes to None
        self._empty = None
        self._bounded = None
        self._radius = None
        self._center = None
        self._vertices = None

    def _selection_matrix(self, indices=None):
        """
        Returns a selection matrix S such that S x = x[indices].

        Arguments
        ----------
        indices : list of int
            Set of indices of elements of x that have to be selected by S.

        Returns
        ----------
        S : numpy.ndarray
            Selection matrix.
        """

        # if indices is None select all the rows
        n = self.A.shape[1]
        if indices is None:
            indices = range(n)

        # delete from the identity matrix all the rows that are not in indices
        complement = [i for i in range(n) if i not in indices]
        S = np.delete(np.eye(n), complement, 0)

        return S


    @staticmethod
    def from_bounds(x_min, x_max, indices=None, n=None):
        """
        Instantiates a Polyhedron in the form {x | x_min <= x[indices] <= x_max}.
        If indices is None, the inequality is applied to all the elements of x.
        If indices is not None, n must be provided to determine the length of x.

        Arguments
        ----------
        x_min : numpy.ndarray
            Lower bound on the elements of x.
        x_max : numpy.ndarray
            Upper bound on the elements of x.
        indices : list of int
            Set of indices of elements of x to which the inequality applies.
        n : int
            Dimension of the vector x in R^n.
        """

        # check if n is provided
        if indices is not None and n is None:
            raise ValueError("specify the length of x to instantiate the polyhedron.")

        # check size of the bounds
        if x_min.size != x_max.size:
            raise ValueError("bounds must have the same size.")

        # construct the polyhedron
        if n is None:
            n = x_min.size
        A = np.zeros((0, n))
        b = np.zeros(0)
        p = Polyhedron(A, b)
        p.add_bounds(x_min, x_max, indices)

        return p

    @staticmethod
    def _same_number_rows(E, f):
        """
        Checks that E and f have the same number of rows.

        Arguments
        ----------
        E : numpy.ndarray
            Left-hand side of the (in)equalities.
        f : numpy.ndarray
            Right-hand side of the (in)equalities.
        """

        # check nomber of rows
        if E.shape[0] != f.size:
            raise ValueError("incoherent size of the inputs.")

    def contains(self, x, tol=1.e-7):
        """
        Determines if the given point belongs to the polytope.

        Arguments
        ----------
        x : numpy.ndarray
            Point whose belonging to the polyhedron must be verified.
        tol : float
            Maximum distance of a point from the polyhedron to be considered an internal point.

        Returns
        ----------
        contains_x : bool
            True if the point x is inside the polyhedron, False otherwise.
        """

        # check inequalities
        in_ineq = np.max(self.A.dot(x) - self.b) <= tol

        # check equalities
        in_eq = True
        if self.C.shape[0] > 0:
            in_eq = np.abs(np.max(self.C.dot(x) - self.d)) <= tol
        contains_x = in_ineq and in_eq

        return contains_x