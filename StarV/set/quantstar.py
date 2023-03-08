from StarV.set.star import Star

import numpy as np

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
        
        
    # ======================================================================================
    def _validate_set_args(self, set):
        assert isinstance(set.V, np.ndarray), 'error: \
            basis matrix should be a numpy array'
        assert isinstance(set.pred_lb, np.ndarray), 'error: \
            lower bound vector should be a 1D numpy array'
        assert isinstance(set.pred_ub, np.ndarray), 'error: \
            upper bound vector should be a 1D numpy array'
            