from StarV.set.star import Star, pc

import numpy as np

from StarV.set.polyhedron import Polyhedron

class QuantizedStar:
    """
        Quntized Star class
        author: Mykhailo Ivashchenko
        date: 02/20/2023
        
        ==========================================================================
        Representation of a QuntizedStar:
        
        ==========================================================================
    """
    
    
    def __init__(self, *args):
        self.acceptable_set_input = ["Star", "ImageStar", "ProbStar"]
        
        if len(args) == 1:
            I = args[0]
            
            assert(I.__class__.__name__ in self.acceptable_set_input), \
            'error: %s' % "Object of type {0} is not supported. \
                           Use one of these objects for initialization: {2}". \
                           format(I.__class__.__name__, str(self.acceptable_set_input))
                           
            self.V = I.V
            self.lb = I.pred_lb
            self.ub = I.pred_ub
            self.nVars = I.nVars
            self.dim = I.dim
        else:
            raise NotImplementedError('The given input is not supported by the constructor')
        
    def contains(self, vertex):
        """
        """

        assert vertex.shape[0] == 1, 'error: Dimension mismatch'
        assert vertex.shape[1] != 1, 'error: Invalid Star point'
    
    def generate_vertices(self):
        import math
        from sklearn.utils.extmath import cartesian

        """
            Generates a set of vertices for the current QuantizedStar
            using the initialized Quantizer

            return -> np.array([*]) - a set of vertices
        """

        vertices = []
    
        for i in range(self.nVars):
            vertices.append(np.array(range(math.ceil(self.lb[i]), math.floor(self.ub[i]) + 1, 1)))
            
        vertices_cartesian = cartesian(vertices)

        p = Polyhedron.from_bounds(self.lb, self.ub)

        result = np.array([])

        i = 0
        for vertex in vertices_cartesian:
            if not p.contains(vertex):
                vertices_cartesian = np.delete(vertices_cartesian, i)
                i -= 1

            i += 1

        return vertices_cartesian

    # ======================================================================================
    def _validate_set_args(self, set):
        assert isinstance(set.V, np.ndarray), 'error: \
            basis matrix should be a numpy array'
        assert isinstance(set.pred_lb, np.ndarray), 'error: \
            lower bound vector should be a 1D numpy array'
        assert isinstance(set.pred_ub, np.ndarray), 'error: \
            upper bound vector should be a 1D numpy array'
            