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
            kernel_weight = layer.weight.detach().numpy()
            kernel_bias = layer.bias.detach().numpy()
            stride = layer.stride
            padding = layer.padding
            dilation = layer.dilation
        
        else:
            raise Exception('error: unsupported neural network module')

        assert isinstance(kernel_weight, np.ndarray), \
        'error: kernel weight should be a numpy array'
        assert isinstance(kernel_bias, np.ndarray), \
        'error: kernel bias should be a numpy array'

        assert isinstance(stride, tuple) or isinstance(stride, list), \
        'error: stride should be a tuple or list'
        assert isinstance(padding, tuple) or isinstance(padding, list), \
        'error: padding should be a tuple or list'
        assert isinstance(dilation, tuple) or isinstance(dilation, list), \
        'error: dilation should be a tuple or list'
        
        # #and stride[0] == 1, \
        # assert len(stride) == 2, 'error: invalid stride'

        # #and (padding[0] == 1 or padding[1] == 4), \
        # assert len(padding) == 2, 'error: invalid padding'

        # # and dilation[0] == 1, \
        # assert len(dilation) == 2, 'error: invalid dilation'

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


    def conv2d_naive(self, x):
        im_c, im_h, im_w = x.shape
        ker_c, ker_h, ker_w = self.num_channels,
        


def conv_2d(image):
    """
    
    x = image
    """





if __name__ == "__main__":
    in_batch = 20
    in_height = 50
    in_width = 100
    in_channel = 16
    out_channel = 33
    kernel_size = (3, 5)
    layer = torch.nn.Conv2d(in_channel, out_channel, kernel_size, stride=(2, 1), padding=(4, 2), dilation=(3, 1))
    input = torch.randn(in_batch, in_channel, in_height, in_width)
    output = layer(input)

    # print('output: {}'.format(output.shape)) # [20, 33, 26, 100] -> [in_batch, out_channel, ?,in_width]
    # print('weight: {}'.format(layer.weight.shape)) # [33, 16, 3, 5] -> [out_channel, in_channel, kerne_size]
    # print('bias: {}'.format(layer.bias.shape)) # [33] -> out_channel

    height = 5
    width = 5

    kernel_size = (2, 2)
    bias_size = 1
    
    image = np.random.rand(height, width)
    kernel = np.random.rand(kernel_size[0], kernel_size[1])
    bias = np.random.rand(bias_size)

    out = conv_2d(kernel, bias, image)
    print(out.shape)

    print('image: ', image)
    print('kernel: ', kernel)
    print('bias: ', bias)
    print('out: ', out)

    # torch_conv = torch.nn.Conv2d(1, 1, 2)
    # torch_conv_op = torch_conv(torch.from_numpy(image))
    # print(torch_conv_op)