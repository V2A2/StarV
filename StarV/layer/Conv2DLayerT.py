"""
Convolutional 2D Layer Class
Sung Woo Choi, 08/11/2023
"""

import torch
import numpy as np

class Conv2DLayer(object):
    """ Conv2DLayer Class
    
        properties:

        methods:


    """

    def __init__(self,
                 layer,
                 stride = (1, 1),
                 padding = (0, 0, 0, 0), # size of padding [t b l r] for nonnegative integers
                 dilation = (1, 1),
                 module = 'default',
                 ):
        
        if module == 'default':
            kernel_weight, kernel_bias = layer

        elif module == 'pytorch':
            kernel_weight = layer.weight
            kernel_bias = layer.bias
            stride = layer.stride
            padding = layer.padding
            dilation = layer.dilation
        
        else:
            raise Exception('error: unsupported neural network module')

        assert isinstance(kernel_weight, torch.Tensor), \
        'error: kernel weight should be a numpy array'
        assert isinstance(kernel_bias, torch.Tensor), \
        'error: kernel bias should be a numpy array'

        assert isinstance(stride, tuple) or isinstance(stride, list), \
        'error: stride should be a tuple or list'
        assert isinstance(padding, tuple) or isinstance(padding, list), \
        'error: padding should be a tuple or list'
        assert isinstance(dilation, tuple) or isinstance(dilation, list), \
        'error: dilation should be a tuple or list'
        
        #and stride[0] == 1, \
        assert len(stride) == 2, 'error: invalid stride'

        #and (padding[0] == 1 or padding[1] == 4), \
        assert len(padding) == 2, 'error: invalid padding'

        # and dilation[0] == 1, \
        assert len(dilation) == 2, 'error: invalid dilation'

        self.layer = layer

        weight_shape = kernel_weight.shape
        bias_shape = kernel_bias.shape

        len_ = len(weight_shape)
        if len_ == 2:
            self.num_filters = 1
            self.num_channels = 1
            self.kernel_size = weight_shape

        elif len_ == 3:
            self.num_filters = 1
            self.num_channels = weight_shape[2]
            self.kernel_size = weight_shape[:2]

        elif len_ == 4:
            self.num_filters = weight_shape[3]
            self.num_channels = weight_shape[2]
            self.kernel_size = weight_shape[:2]

        self.weights = kernel_weight
        self.bias = kernel_bias
        
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def info(self):
        print('Convolutional 2D Layer')
        print('stride: {}'.format(self.stride))
        print('padding: {}'.format(self.padding))
        print('dilation: {}'.format(self.dilation))
        print('kernel weights: {}'.format(self.weights.shape))
        print('bias weights: {}'.format(self.bias.shape))
        return '\r\n'

if __name__ == "__main__":
    layer = torch.nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
    C2D = Conv2DLayer(layer=layer, module='pytorch')
    print(C2D.info())