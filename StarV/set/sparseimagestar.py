"""
Sparse Image Star Class
Sung Woo Choi, 11/28/2023

"""

import copy
import torch
import numpy as np
import scipy.sparse as sp
import gurobipy as gp
from gurobipy import GRB
from scipy.optimize import linprog
from scipy.linalg import block_diag
import glpk
import polytope as pc

GUROBI_OPT_TOL = 1e-6

# import numba_scipy
# from numba import jit, njit

# class SparseImage3D(object):
#     def __init__(self, IM, channel):

#         assert isinstance(IM, sp.coo_matrix), \
#         'error: image should scipy sparse coo matrix'
#         assert isinstance(channel, int) or isinstance(channel, np.integer), \
#         'error: channel corresponding sparse image should be an integer'
#         assert channel >= 0, \
#         'error: channel of an image should be an non-negative interger'

#         self.image = IM
#         self.channel = channel
#         self.num_height = IM.shape[0]
#         self.num_width = IM.shape[1]

#     def size(self):
#         return np.array([self.num_height, self.num_width])

#     def __str__(self, todense=False):
#         print('number of height: {}'.format(self.num_height))
#         print('number of width: {}'.format(self.num_width))
#         print('channel: {}'.format(self.channel))
#         if todense:
#             print('image: \n{}'.format(self.image.todense()))
#         else:
#             print('image: \n{}'.format(self.image))
#         return ''

#     def __repr__(self):
#         print('channel: {}'.format(self.channel))
#         print('image: {}'.format(self.image.shape))
#         return ''


# class SparseImage4D(object):
#     def __init__(self, num_height, num_width, num_channel, num_pred):
        
#         assert isinstance(num_height, int) or isinstance(num_height, np.integer), \
#         'error: number of height of corresponding sparse image should be an integer'
#         assert isinstance(num_width, int) or isinstance(num_width, np.integer), \
#         'error: number of width of corresponding sparse image should be an integer'
#         assert isinstance(num_channel, int) or isinstance(num_channel, np.integer), \
#         'error: number of channels of corresponding sparse image should be an integer'
#         assert isinstance(num_pred, int) or isinstance(num_pred, np.integer), \
#         'error: predicate number of corresponding sparse image should be an integer'

#         self.channel = [None for _ in range(num_channel)]
#         self.num_pred = num_pred
#         self.num_channel = num_channel
#         self.num_height = num_height
#         self.num_width = num_width


#     def append(self, image3d,  num_pred):

#         assert isinstance(image3d, SparseImage3D), \
#         'error: input 3d image should be SparseImage3D class'
#         assert self.num_pred == num_pred, \
#         'error: inconsistent predicate number between SparseImage4D and input image'
#         assert image3d.channel >= 0 and image3d.channel < self.num_channel, \
#         'error: given channel is out of range'
#         assert image3d.num_height == self.num_height, \
#         'error: inconsistent number of height between SparseImage4D and input image'
#         assert image3d.num_width == self.num_width, \
#         'error: inconsistent number of width between SparseImage4D and input image'

#         self.channel[image3d.channel] = image3d

#     def size(self):
#         return np.array([self.num_height, self.num_width, self.num_channel])
       
#     def __str__(self):
#         print('predicate number: ', self.num_pred)
#         print('number of channels: ', self.num_channel)
#         print('sparse 3d images:')
#         for i in range(self.num_channel):
#             print('channel: {}'.format(i))
#             if self.channel[i] is not None:
#                 print('image: {}'.format(self.channel[i].image.shape))
#                 print('image:\n{}'.format(self.channel[i].image))
#             else:
#                 print('image: {}'.format(None))
#         return ''

#     def __repr__(self):
#         print('predicate number: {}'.format(self.num_pred))
#         print('number of channels: {}'.format(self.num_channel))
#         print('sparse 3d images:')
#         for i in range(self.num_channel):
#             print('channel: {}'.format(i))
#             if self.channel[i] is not None:
#                 print('image: {}'.format(self.channel[i].image.shape))
#             else:
#                 print('image: {}'.format(None))
#         return ''
    

# class SparseImage(object):
#     def __init__(self, images):
        
#         if isinstance(images, list):
#             assert isinstance(images[0], SparseImage4D), \
#             'error: images should SparseImage4D class'

#             if len(images) > 0:        
#                 self.images = images
#                 self.num_images = len(images)
#                 self.num_channel = images[0].num_channel
#                 self.num_height = images[0].num_height
#                 self.num_width = images[0].num_width

#             else:
#                 self.images = []
#                 self.num_images = 0
#                 self.num_channel = 0
#                 self.num_height = 0
#                 self.num_width = 0

#         elif isinstance(images, None):
#             self.images = []
#             self.num_images = 0
#             self.num_channel = 0
#             self.num_height = 0
#             self.num_width = 0

#         else:
#             assert isinstance(images, SparseImage4D), \
#             'error: an image should SparseImage4D class'

#             self.images = [images]
#             self.num_images = 1
#             self.num_channel = images[0].num_channel
#             self.num_height = images[0].num_height
#             self.num_width = images[0].num_width
    
#     def append(self, image):
#         assert isinstance(image, SparseImage4D), \
#         'error: input image should be a SparseImage4D class'
#         self.images.append(image)
#         self.num_images += 1

#     def extend(self, images):
#         assert isinstance(images, list) and isinstance(images[0], SparseImage4D), \
#         'error: images should a list containing SparseImage4D class images'
#         self.images.extend(images)
#         self.num_images += len(images)

#     def to_dense(self):
#         dense = np.empty([self.num_height, self.num_width, self.num_channel, self.num_images])
#         for n_ in range(self.num_images):
#             for c_ in range(self.num_channel):
#                 dense[:, :, c_, n_] = self.images[n_].channel[c_].image.toarray()
#         return dense

#     def size(self):
#         return np.array([self.num_height, self.num_width, self.num_channel, self.num_images])
    
#     def __str__(self):
#         print('number of images: ', self.num_images)
#         print('number of channels: ', self.num_channel)
#         for i in range(self.num_images):
#             print(self.images[i])
#         return ''
    
#     @staticmethod
#     def from_numpy(input):

#         in_dim = input.ndim

#         assert isinstance(input, np.ndarray), \
#         'error: input should be numpy ndarray'
#         assert in_dim >= 2 and in_dim <= 4, \
#         'error: input should be 2D, 3D, or 4D numpy ndarray'

#         input = copy.deepcopy(input)

#         if in_dim == 2:
#             input = input[:, :, None, None]
#         elif in_dim == 3:
#             input = input[:, :, :, None]

#         h, w, c, n = input.shape

#         sp_images = []
#         for n_ in range(n):
#             im4d = SparseImage4D(h, w, c, n_)

#             for c_ in range(c):
#                 im2d = sp.coo_matrix(input[:, :, c_, n_])
#                 im3d = SparseImage3D(im2d, c_)
#                 im4d.append(im3d, n_)

#             sp_images.append(im4d)

#         return SparseImage(sp_images)


class SparseImage3D(object):
    def __init__(self, num_height, num_width, num_channel, pred):
        
        assert isinstance(num_height, int) or isinstance(num_height, np.integer), \
        'error: number of height of corresponding sparse image should be an integer'
        assert isinstance(num_width, int) or isinstance(num_width, np.integer), \
        'error: number of width of corresponding sparse image should be an integer'
        assert isinstance(num_channel, int) or isinstance(num_channel, np.integer), \
        'error: number of channels of corresponding sparse image should be an integer'
        assert isinstance(pred, int) or isinstance(pred, np.integer), \
        'error: predicate number of corresponding sparse image should be an integer'

        self.channel = [None for _ in range(num_channel)]
        self.pred = pred
        self.num_channel = num_channel
        self.num_height = num_height
        self.num_width = num_width


    def append(self, image2d, ch, pred):

        assert isinstance(image2d, sp.coo_array), \
        'error: input image2d should scipy sparse coo array'
        assert image2d.shape[0] == self.num_height, \
        'error: inconsistent number of height between SparseImage4D and input image2d'
        assert image2d.shape[1] == self.num_width, \
        'error: inconsistent number of width between SparseImage4D and input image2d'
        assert ch >= 0 and ch < self.num_channel, \
        'error: given channel is out of range for SparseImage4D'
        assert self.pred == pred, \
        'error: inconsistent predicate number between SparseImage4D and input image'
        self.channel[ch] = image2d

    def size(self):
        return np.array([self.num_height, self.num_width, self.num_channel])

    def __str__(self, to_dense=False):
        print('predicate number: ', self.pred)
        print('number of channels: ', self.num_channel)
        print('sparse 3d images:')
        for i in range(self.num_channel):
            print('channel: {}'.format(i))
            if self.channel[i] is not None:
                if to_dense is False:
                    print('image shape: {}'.format(self.channel[i].shape))
                    print('image:\n{}'.format(self.channel[i]))
                else:
                    print('image:\n{}'.format(self.channel[i].todense()))
            else:
                print('image: {}'.format(None))
        return ''

    def __repr__(self):
        print('predicate number: {}'.format(self.pred))
        print('number of channels: {}'.format(self.num_channel))
        print('sparse 3d images:')
        for i in range(self.num_channel):
            print('channel: {}'.format(i))
            if self.channel[i] is not None:
                print('image: {}'.format(self.channel[i].shape))
            else:
                print('image: {}'.format(None))
        print('')
        return ''

class SparseImage(object):
    def __init__(self, images, num_pred):
        
        if isinstance(images, list):
            assert isinstance(images[0], SparseImage3D), \
            'error: images should SparseImage4D class'

            if len(images) > 0:        
                self.images = images
                self.num_pred = num_pred
                self.num_images = len(images)
                self.num_channel = images[0].num_channel
                self.num_height = images[0].num_height
                self.num_width = images[0].num_width
                # self.dtype = images[0].dtype

            else:
                self.images = []
                self.num_pred = 0
                self.num_images = 0
                self.num_channel = 0
                self.num_height = 0
                self.num_width = 0
                # self.dtype = None

        elif isinstance(images, None):
            self.images = []
            self.num_pred = 0
            self.num_images = 0
            self.num_channel = 0
            self.num_height = 0
            self.num_width = 0
            # self.dtype = None

        else:
            assert isinstance(images, SparseImage3D), \
            'error: an image should SparseImage3D class'

            self.images = [images]
            self.num_pred = 1
            self.num_images = 1
            self.num_channel = images.num_channel
            self.num_height = images.num_height
            self.num_width = images.num_width
            # self.dtype = images.dtype
    
    def append(self, image):
        assert isinstance(image, SparseImage3D), \
        'error: input image should be a SparseImage4D class'
        assert image.pred < self.num_pred, \
        'error: predicate number of input image exceed number of predicate variables'
        self.images.append(image)
        self.num_images += 1

    def extend(self, images):
        assert isinstance(images, list) and isinstance(images[0], SparseImage3D), \
        'error: images should a list containing SparseImage4D class images'
        self.images.extend(images)
        self.num_images += len(images)

    def to_dense(self):
        dense = np.zeros([self.num_height, self.num_width, self.num_channel, self.num_images])
        for n_ in range(self.num_images):
            for c_ in range(self.num_channel):
                im = self.images[n_].channel[c_]
                if im is None:
                    continue
                dense[:, :, c_, n_] = im.toarray()
        return dense

    def nbytes(self):
        nbt = 0
        for n_ in range(self.num_images):
            for c_ in range(self.num_channel):
                im = self.images[n_].channel[c_]
                if im is None:
                    continue
                nbt += im.data.nbytes + im.row.nbytes + im.col.nbytes
        return nbt
    
    def size(self):
        return np.array([self.num_height, self.num_width, self.num_channel, self.num_pred])
    
    def __str__(self, to_dense=False):
        print('number of images: ', self.num_images)
        # print('number of channels: ', self.num_channel)
        for i in range(self.num_images):
            print(self.images[i].__str__(to_dense=to_dense))
        return ''
    
    def __len__(self):
        return 1
    
    
    def nnz(self):
        nnz = 0
        for n_ in range(self.num_images):
            for c_ in range(self.num_channel):
                im = self.images[n_].channel[c_]
                if im is None:
                    continue
                nnz += im.nnz
        return nnz
    
    def density(self):
        num_pixel = self.num_height * self.num_width * self.num_channel * self.num_pred
        return self.nnz() / num_pixel
        
    @staticmethod
    def from_numpy(input):

        in_dim = input.ndim

        assert isinstance(input, np.ndarray), \
        'error: input should be a numpy ndarray'
        assert in_dim >= 2 and in_dim <= 4, \
        'error: input should be 2D, 3D, or 4D numpy ndarray'

        input = copy.deepcopy(input)
        
        if in_dim == 2:
            input = input[:, :, None, None]
        elif in_dim == 3:
            input = input[:, :, :, None]

        h, w, c, n = input.shape

        sp_images = []
        for n_ in range(n):
            im3d = SparseImage3D(h, w, c, n_)

            for c_ in range(c):
                im2d = sp.coo_array(input[:, :, c_, n_])
                im3d.append(im2d, c_, n_)

            sp_images.append(im3d)

        return SparseImage(sp_images, n)
    
    @staticmethod
    def from_numpy_array(input, num_height, num_width, num_channel, num_pred):
        assert isinstance(input, np.ndarray) and input.ndim == 1, \
        'error: input should be a 1D numpy ndarray'
        
        input = input.reshape(num_height, num_width, num_channel, num_pred)
        return SparseImage.from_numpy(input, num_pred)

    @staticmethod
    def rand(h, w, c, n, density=0.1):
        sp_images = []
        for n_ in range(n):
            im3d = SparseImage3D(h, w, c, n_)

            for c_ in range(c):
                im2d = sp.random(m=h, n=w, density=density)
                im3d.append(im2d, c_, n_)

            sp_images.append(im3d)

        return SparseImage(sp_images, n)

    def to_2D(self, num_pred):
        """ Converts V[h, w, c, n] (SparseImage) into V[h*w*c, n] (scipy.sparse.coo_matrix)"""

        n = self.num_height * self.num_width * self.num_channel
        V = sp.coo_matrix((n, num_pred))

        for n_ in range(self.num_images):
            for c_ in range(self.num_channel):
                im = self.images[n_].channel[c_]
                if im is None:
                    continue

                row = self.hwc_to1D(im.row, im.col, c_)
                col = self.images[n_].pred * np.ones(len(row))
                data = im.data
                
                V += sp.coo_array(
                        (data, (row, col)), shape=(n, num_pred)
                    )
        return V
    
    def flatten(self, num_pred):
        """ Converts V[h, w, c, n] (SparseImage) into V[h*w*c*n] numpy"""
        V = self.to_2D(num_pred).toarray().flatten()
        return V

    def index_to3D(self, index):
        # V is in [height, width, channel] order

        index = copy.deepcopy(index)
        num = self.num_width * self.num_channel
        h = index // num
        index -= h * num
        w = index // self.num_channel
        c = index % self.num_channel
        return h, w, c

    def hwc_to1D(self, h, w, c):
        return h * self.num_width * self.num_channel + w * self.num_channel + c
    
    def getRow(self, index):
        """Gets a row elements along the predicate dimension in current SparseImage shape"""
        h, w, c = self.index_to3D(index)
        return self.getRow_hwc(h, w, c)
    
    def getRows(self, map):
        """Gets multiple rows elements along the predicate dimension in current SparseImage shape"""
        h_map, w_map, c_map = self.index_to3D(map)
        return self.getRows_hwc(h_map, w_map, c_map)
    
    def getRow_2D(self, index, num_pred):
        """Equivalent to getRows().to_2D() but returns matrix in shape (len(map), num_pred)"""

        h, w, c = self.index_to3D(index)
        return self.getRow_hwc_2D(h, w, c, num_pred)
    
    def getRows_2D(self, map, num_pred):
        """Equivalent to getRows().to_2D() but returns matrix in shape (len(map), num_pred)"""

        h_map, w_map, c_map = self.index_to3D(map)
        return self.getRows_hwc_2D(h_map, w_map, c_map, num_pred)
    
    def getRow_hwc(self, h, w, c):
        """Get a row elements along the predicate dimension in shape (len(map), num_pred)"""

        assert h >= 0 and h <  self.num_height, \
        'error: invalid vertical index should be between {} and {}'.format(0, self.num_height - 1)
        assert w >= 0 and w < self.num_width, \
        'error: invalid horizontal index should be between {} and {}'.format(0, self.num_width - 1)
        assert c >= 0 and c < self.num_channel, \
        'error: invalid channel index should be between {} and {}'.format(0, self.num_channel - 1)

        V = []
        for n_ in range(self.num_images):
            im = self.images[n_].channel[c]
            if im is None:
                continue

            row = im.row
            col = im.col
            data = im.data
            
            indx = np.argwhere((row == h) & (col == w)).flatten()

            row = row[indx]
            col = col[indx]
            data = data[indx]
            im2d = sp.coo_array(
                    (data, (row, col)), shape=(im.shape[0], im.shape[1])
                )
            im3d = SparseImage3D(self.num_height, self.num_width, self.num_channel, n_)
            im3d.append(im2d, c, n_)
            V.append(im3d)
        return SparseImage(V, self.num_pred)
    
    def getRows_hwc(self, h_map, w_map, c_map):
        """Get multiple rows elements along the predicate dimension  in shape (len(map), num_pred)"""
        
        assert isinstance(h_map, list) or (isinstance(h_map, np.ndarray) and h_map.ndim == 1), \
        'error: h_map should a list or 1D numpy array'
        assert isinstance(w_map, list) or (isinstance(w_map, np.ndarray) and h_map.ndim == 1), \
        'error: w_map should a list or 1D numpy array'
        assert isinstance(c_map, list) or (isinstance(c_map, np.ndarray) and c_map.ndim == 1), \
        'error: c_map should a list or 1D numpy array'
        assert len(h_map) == len(w_map) == len(c_map), \
        'error: inconsistent number of elements in h_map, w_map, and c_map'

        V = []
        for n_ in range(self.num_images):
            for k_ in range(len(h_map)):
                im = self.images[n_].channel[c_map[k_]]
                if im is None:
                    continue

                row = im.row
                col = im.col
                data = im.data
                
                indx = np.argwhere((row == h_map[k_]) & (col == w_map[k_])).flatten()

                row = row[indx]
                col = col[indx]
                data = data[indx]
                im2d = sp.coo_array(
                    (data, (row, col)), shape=(im.shape[0], im.shape[1])
                )
                im3d = SparseImage3D(self.num_height, self.num_width, self.num_channel, n_)
                im3d.append(im2d, c_map[k_], n_)
                V.append(im3d)
        return SparseImage(V, self.num_pred)
    
    
    def getRow_hwc_2D(self, h, w, c, num_pred):
        """Get a row elements along the predicate dimension"""

        assert h >= 0 and h <  self.num_height, \
        'error: invalid vertical index should be between {} and {}'.format(0, self.num_height - 1)
        assert w >= 0 and w < self.num_width, \
        'error: invalid horizontal index should be between {} and {}'.format(0, self.num_width - 1)
        assert c >= 0 and c < self.num_channel, \
        'error: invalid channel index should be between {} and {}'.format(0, self.num_channel - 1)

        V = sp.coo_matrix((1, num_pred))

        for n_ in range(self.num_images):
            im = self.images[n_].channel[c]
            if im is None:
                continue

            indx = np.argwhere((im.row == h) & (im.col == w)).flatten()

            n = len(indx)
            # row = self.hwc_to1D(im.row[indx], im.col[indx], c)
            row = np.zeros(n)
            col = self.images[n_].pred * np.ones(n)
            data = im.data[indx]

            V += sp.coo_array(
                    (data, (row, col)), shape=(1, num_pred)
                )
        return V
    
    def getRows_hwc_2D(self, h_map, w_map, c_map, num_pred):
        """Get elements along the predicate dimension"""
        
        assert isinstance(h_map, list) or (isinstance(h_map, np.ndarray) and h_map.ndim == 1), \
        'error: h_map should a list or 1D numpy array'
        assert isinstance(w_map, list) or (isinstance(w_map, np.ndarray) and h_map.ndim == 1), \
        'error: w_map should a list or 1D numpy array'
        assert isinstance(c_map, list) or (isinstance(c_map, np.ndarray) and c_map.ndim == 1), \
        'error: c_map should a list or 1D numpy array'
        assert len(h_map) == len(w_map) == len(c_map), \
        'error: inconsistent number of elements in h_map, w_map, and c_map'

        # n = self.num_height * self.num_width * self.num_channel
        n = len(h_map)
        V = sp.coo_matrix((n, num_pred))

        for n_ in range(self.num_images):
            for k_ in range(len(h_map)):
                im = self.images[n_].channel[c_map[k_]]
                if im is None:
                    continue

                indx = np.argwhere((im.row == h_map[k_]) & (im.col == w_map[k_])).flatten()

                # row = self.hwc_to1D(im.row[indx], im.col[indx], c_map[k_])
                row = k_ * np.ones(len(indx))
                col = self.images[n_].pred * np.ones(len(indx))
                data = im.data[indx]

                V += sp.coo_array(
                    (data, (row, col)), shape=(n, num_pred)
                )
        return V

    def resetRow(self, index):
        h, w, c = self.V.index_to3D(index)
        return self.resetRow_hwc(h, w, c)

    def resetRows(self, map):
        h_map, w_map, c_map = self.V.index_to3D(map)
        return self.resetRows_hwc(h_map, w_map, c_map)
    
    def resetRow_hwc(self, h, w, c):
        """Reset elements along the predicate dimension"""

        assert h >= 0 and h <  self.num_height, \
        'error: invalid vertical index should be between {} and {}'.format(0, self.num_height - 1)
        assert w >= 0 and w < self.num_width, \
        'error: invalid horizontal index should be between {} and {}'.format(0, self.num_width - 1)
        assert c >= 0 and c < self.num_channel, \
        'error: invalid channel index should be between {} and {}'.format(0, self.num_channel - 1)

        V = copy.deepcopy(self)
        for n_ in range(V.num_images):
            im = V.images[n_].channel[c]
            if im is None:
                continue
            
            row = im.row
            col = im.col
            data = im.data
            
            remove_indx = np.argwhere((row == h) & (col == w)).flatten()

            if len(row) == len(remove_indx):
                V.images[n_].channel[c] = None
                continue

            row = np.delete(row, remove_indx)
            col = np.delete(col, remove_indx)
            data = np.delete(data, remove_indx)
            
            V.images[n_].channel[c] = sp.coo_array(
                    (data, (row, col)), shape=(im.shape[0], im.shape[1])
                )
        return V
    
    def resetRows_hwc(self, h_map, w_map, c_map):
        """Reset elements along the predicate dimension"""
        
        assert isinstance(h_map, list) or (isinstance(h_map, np.ndarray) and h_map.ndim == 1), \
        'error: h_map should a list or 1D numpy array'
        assert isinstance(w_map, list) or (isinstance(w_map, np.ndarray) and h_map.ndim == 1), \
        'error: w_map should a list or 1D numpy array'
        assert isinstance(c_map, list) or (isinstance(c_map, np.ndarray) and c_map.ndim == 1), \
        'error: c_map should a list or 1D numpy array'
        assert len(h_map) == len(w_map) == len(c_map), \
        'error: inconsistent number of elements in h_map, w_map, and c_map'

        V = copy.deepcopy(self)
        for n_ in range(V.num_images):
            for k_ in range(len(h_map)):
                im = V.images[n_].channel[c_map[k_]]
                if im is None:
                    continue
                if im.nnz == 0:
                    continue

                row = im.row
                col = im.col
                data = im.data
                
                remove_indx = np.argwhere((row == h_map[k_]) & (col == w_map[k_])).flatten()

                if len(row) == len(remove_indx):
                    V.images[n_].channel[c_map[k_]] = None
                    continue

                row = np.delete(row, remove_indx)
                col = np.delete(col, remove_indx)
                data = np.delete(data, remove_indx)
                V.images[n_].channel[c_map[k_]] = sp.coo_array(
                        (data, (row, col)), shape=(im.shape[0], im.shape[1])
                    )
        return V
    

    # def resetRows_hwcMap(self, h_map, w_map, c_map):
    #     """Reset elements along the predicate dimension"""
        
    #     assert isinstan np.zeros(self.height, self.height, self.num_channel)d a list or 1D numpy array'
    #     assert isinstance(w_map, list) or (isinstance(w_map, np.ndarray) and h_map.ndim == 1), \
    #     'error: w_map should a list or 1D numpy array'
    #     assert isinstance(c_map, list) or (isinstance(c_map, np.ndarray) and c_map.ndim == 1), \
    #     'error: c_map should a list or 1D numpy array'

    #     V = copy.deepcopy(self)
    #     for n_ in range(V.num_images):
    #         for c_ in c_map:
    #             im = V.images[n_].channel[c_]
    #             if im is None:
    #                 continue

    #             row = im.row
    #             col = im.col
    #             data = im.data

    #             remove_indx = np.argwhere(np.isin(row, h_map) & np.isin(col, w_map)).flatten()

    #             row = np.delete(row, remove_indx)
    #             col = np.delete(col, remove_indx)
    #             data = np.delete(data, remove_indx)
    #             V.images[n_].channel[c_] = sp.coo_array(
    #                     (data, (row, col)), shape=(im.shape[0], im.shape[1])
    #                 )
    #     return V
    
# def resetRow(self, h, c):
#     """Reset horizontal indexes"""

#     assert h >= 0 and h < self.height, \
#     'error: invalid vertical index should be between {} and {}'.format(0, self.height - 1)
#     assert c >= 0 and c < self.num_channel, \
#     'error: invalid channel index should be between {} and {}'.format(0, self.num_channel - 1)

#     V = copy.deepcopy(self.V)
#     for n_ in range(V.num_images):
#         im = V.images[n_].channel[c]
#         if im is None:
#             continue

#         row = im.row
#         col = im.col
#         data = im.data

#         indx = np.argwhere(row != h).flatten()

#         row = row[indx]
#         col = col[indx]
#         data = data[indx]
#         V.images[n_].channel[c] = sp.coo_array(
#                 (data, (row, col)), shape=(self.height, self.width)
#             )
#     return V

# def resetRows(self, h_map, c_map):
#     """Reset horizontal indexes"""
    
#     assert isinstance(h_map, list) or (isinstance(h_map, np.ndarray) and h_map.ndim == 1), \
#     'error: h_map should a list or 1D numpy array'
#     assert isinstance(c_map, list) or (isinstance(c_map, np.ndarray) and c_map.ndim == 1), \
#     'error: c_map should a list or 1D numpy array'

#     V = copy.deepcopy(self.V)
#     for n_ in range(V.num_images):
#         for c_ in c_map:
#             im = V.images[n_].channel[c_]
#             if im is None:
#                 continue

#             row = im.row
#             col = im.col
#             data = im.data

#             indx = np.argwhere(~np.isin(row, h_map)).flatten()

#             row = row[indx]
#             col = col[indx]
#             data = data[indx]
#             V.images[n_].channel[c_] = sp.coo_array(
#                     (data, (row, col)), shape=(self.height, self.width)
#                 )
#     return V



class SparseImageStar(object):
    """
        
        Sparse Image Star for reachability
        author: Sung Woo Choi
        date: 08/09/2022
        Representation of a SparseImageStar
        ======================= np.zeros(self.height, self.height, self.num_channel)
        H W C N
        N:batch_size, H:input_img_height, W:input_img_width, C:no.of.channels 
        https://pytorch.org/blog/accelerating-pytorch-vision-models-with-channels-last-on-cpu/
        ==========================================================================  
    """

    def __init__(self, *args):
        """
            Key Attributes:
            c = []
            V = []
            C = []
            d = []

            num_pred = 0 # number of predicate variables
            pred_lb = [] # lower bound of predicate variables
            pred_ub = [] # upper bound of predicate variables

            height = 0 # height of the image
            width = 0 # width of the image
            num_channel = 0 # number of channels of the image
            num_pred = 0 # number of predicate variables
            num_pixel = 0 # number of pixels in image
        """

        len_ = len(args)

        if len_ == 6:
            [c, V, C, d, pred_lb, pred_ub] = copy.deepcopy(args)

            assert isinstance(c, np.ndarray), \
            'error: anchor image should be a numpy array'
            assert isinstance(pred_lb, np.ndarray) and pred_lb.ndim == 1, \
            'error: lower bound vector should be a 1D numpy array'
            assert isinstance(pred_ub, np.ndarray) and pred_ub.ndim == 1, \
            'error: upper bound vector should be a 1D numpy array'
            assert pred_ub.shape == pred_lb.shape, \
            'error: inconsistent number of predicate variables between predicate lower- and upper-boud vectors'


            assert isinstance(V, SparseImage), \
            'error: generator image should be a SparseImage class'

            # if isinstance(V, SparseImage):
            #     num_pred = V.num_pred
            #     self.flatten = False
            # elif isinstance(V, sp.coo_array):
            #     num_pred = V.shape[1]
            #     self.flatten = True
            # else:
            #     raise Exception('error: generator image should be a SparseImage class or scipy.sparce.coo_array')

            if len(d) > 0:
                assert isinstance(C, sp.csr_matrix), \
                'error: linear constraints matrix should be a 2D scipy sparse csr matrix'
                assert isinstance(d, np.ndarray) and d.ndim == 1, \
                'error: linear constraints vector should be a 1D numpy array'
                assert C.shape[0] == d.shape[0], \
                'error: inconsistency between constraint matrix and constraint vector'
                assert C.shape[1] == pred_lb.shape[0], \
                'error: inconsistent number of predicate variables between constraint matrix and predicate bound vectors'
                assert C.shape[1] == V.num_pred, \
                'error: inconsistent number of predicate variables between constraint matrix and generato image'
            
            self.c = c
            self.V = V
            self.C = C
            self.d = d
            self.pred_lb = pred_lb
            self.pred_ub = pred_ub

            self.height, self.width, self.num_channel, self.num_pred = V.size()
            self.num_pixel = self.height * self.width * self.num_channel

            # if self.flatten:
            #     self.height, self.width, self.num_channel, self.num_pred = V.size()
            #     self.num_pixel = self.height * self.width * self.num_channel
            # else:
            #     self.height, self.num_pred = V.shape
            #     self.width = 1
            #     self.num_channel = 1
            #     self.num_pixel = V.shape[0]

        elif len_ == 2:
            [lb, ub] = copy.deepcopy(args)
            
            assert isinstance(lb, np.ndarray), \
            'error: lower bound image should be a numpy array'
            assert isinstance(ub, np.ndarray), \
            'error: upper bound image should be a numpy array'
            assert lb.shape == ub.shape, \
            'error: inconsistency between lower bound image and upper bound image'
            assert lb.ndim > 1 and lb.ndim <= 3, \
            'error: lower and upper bound vectors should be a 2D or 3D numpy array'

            if (ub < lb).any():
                raise Exception(
                    'error: the upper bounds must not be less than the lower bounds for all dimensions')
            
            if lb.ndim == 2:
                lb = lb[:, :, None]
                ub = ub[:, :, None]

            dtype = lb.dtype
            h, w, c = lb.shape

            nv = (ub > lb).sum()

            # n = 0
            # sp_images = []
            # # number of channels
            # for k in range(c):
            #     indx = np.argwhere(ub[:, :, k] > lb[:, :, k])
                
            #     for i in indx:
            #         row = np.array([i[0]]).astype(np.ushort)
            #         col = np.array([i[1]]).astype(np.ushort)
            #         data = np.array([1.0]).astype(dtype)

            #         im2d = sp.coo_array(
            #             (data, (row, col)), shape=(h, w)
            #         )

            #         im3d = SparseImage3D(h, w, c, n)
            #         im3d.append(im2d, k, n)
            #         sp_images.append(im3d)
            #         n += 1

            # flattens in channel -> width -> height order
            n = 0
            sp_images = []
            # number of channels
            for h_ in range(h):
                indx = np.argwhere(ub[h_, :, :] > lb[h_, :, :])
                
                for i in indx:
                    row = np.array([h_]).astype(np.ushort)
                    col = np.array([i[0]]).astype(np.ushort)
                    ch_ = i[1]
                    data = np.ones(1, dtype=dtype)

                    im2d = sp.coo_array(
                        (data, (row, col)), shape=(h, w)
                    )

                    im3d = SparseImage3D(h, w, c, n)
                    im3d.append(im2d, ch_, n)
                    sp_images.append(im3d)
                    n += 1


            self.V = SparseImage(sp_images, n)
            self.c = np.zeros(lb.shape[:3], dtype = lb.dtype)

            self.C = sp.csr_matrix((0, 0), dtype=dtype)
            self.d = np.empty([0], dtype=dtype)
            
            self.pred_lb = lb.flatten()
            self.pred_ub = ub.flatten()

            self.height, self.width, self.num_channel = lb.shape[:3]
            self.num_pred = nv
            self.num_pixel = self.height * self.width * self.num_channel
            self.flatten = False
            

        elif len_ == 0: 
            self.c = np.empty([0, 0, 0])
            self.V = SparseImage([])
            self.C = sp.csr_matrix((0, 0))
            self.d = np.empty([0])
            self.pred_lb = np.empty([0])
            self.pred_ub = np.empty([0])
            self.num_pred = 0
            self.height = 0
            self.width = 0
            self.num_channel = 0
            self.num_pixel = 0
            self.flatten = False

        else:
            raise Exception(
                'error: invalid number of input arguments (should be 0, 2, 6)')
    
    def __str__(self, toDense=False):
        print('SparseImageStar Set:')
        print('c: {}'.format(self.c))
        print('V:')
        if toDense:
            # if self.flatten:
            #     print(self.V.toarray())
            # else:
            #     print(self.V.to_dense())
            print(self.V.to_dense())
            print('C_{}: \n{}'.format(self.C.getformat(), self.C.todense()))
        else:
            print(self.V)
            print('C: \n{}'.format(self.C))
        print('d: {}'.format(self.d))
        print('pred_lb: {}'.format(self.pred_lb))
        print('pred_ub: {}'.format(self.pred_ub))

        print('height: {}'.format(self.height))
        print('width: {}'.format(self.width))
        print('num_channel: {}'.format(self.num_channel))
        print('num_pred: {}'.format(self.num_pred))
        print('num_images: {}'.format(self.V.num_images))
        return ''

    def __repr__(self):
        print('SparseImageStar Set:')
        print('c: {}, {}'.format(self.c.shape, self.c.dtype))
        # if self.flatten:
        #     print('V: {}'.format(self.V.shape))
        # else:
        #     print('V: {}'.format(self.V.size()))
        print('V: {}'.format(self.V.size()))
        print('C_{}: {}, {}'.format(self.C.getformat(), self.C.shape, self.C.dtype))
        print('d: {}, {}'.format(self.d.shape, self.d.dtype))
        print('pred_lb: {}, {}'.format(self.pred_lb.shape, self.pred_lb.dtype))
        print('pred_ub: {}, {}'.format(self.pred_ub.shape, self.pred_ub.dtype))
        
        print('height: {}'.format(self.height))
        print('width: {}'.format(self.width))
        print('num_channel: {}'.format(self.num_channel))
        print('num_pred: {}'.format(self.num_pred))
        print('num_images: {}'.format(self.V.num_images))
        print('')
        return ''
    
    def __len__(self):
        return 1
    
    def nbytes(self):
        # V and c
        nbt = self.V.nbytes() + self.c.nbytes
        # C and d
        nbt += self.C.data.nbytes + self.C.indptr.nbytes + self.C.indices.nbytes + self.d.nbytes
        # pred_lb and pred_ub
        nbt += self.pred_lb.nbytes + self.pred_ub.nbytes
        return nbt
    
    def flatten(self):
        V = self.V.to_2D(self.num_pred) # in sp.coo_matrix format
        c = self.c.flatten()
        return SparseImageStar(c, V, self.C, self.d, self.pred_lb, self.pred_ub)

    def flatten_affineMap(self, W=None, b=None):

        # assert not self.flatten, 'error: SparseImageStar is not flatten'
        
        if W is None and b is None:
            return copy.deepcopy(self)

        if W is not None:
            assert isinstance(W, np.ndarray), 'error: ' + \
            'the mapping matrix should be a 2D numpy array'
            assert W.shape[1] == self.num_pixel, 'error: ' + \
            'inconsistency between mapping matrix and SparseImageStar dimension'

            V = np.matmul(W, self.V.to_2D(self.num_pred).toarray())
            c = np.matmul(W, self.c.flatten())

        if b is not None:
            assert isinstance(b, np.ndarray), 'error: ' + \
            'the offset vector should be a 1D numpy array'
            assert len(b.shape) == 1, 'error: ' + \
            'offset vector should be a 1D numpy array'

            if W is not None:
                assert W.shape[0] == b.shape[0], 'error: ' + \
                'inconsistency between mapping matrlen(self.Cx and offset'
            else:
                assert b.shape[0] == self.num_pixel, 'error: ' + \
                'inconsistency between offset vector and SparseStar dimension'

            c += b

        V = SparseImage.from_numpy(V.reshape(self.height, self.width, self.num_channel, self.num_pred))
        c = c.reshape(self.height, self.width, self.num_channel)
        return SparseImageStar(c, V, self.C, self.d, self.pred_lb, self.pred_ub)
    
    # def affineMap(self, W=None, b=None):
        
    #     assert self.flatten, 'error: SparseImageStar is not flatten'
        
    #     if W is None and b is None:
    #         return copy.deepcopy(self)

    #     V = self.V.to_2D(self.num_pred).toarray()
    #     c = copy.deepcopy(self.c).flatten()

    #     if W is not None:
    #         assert isinstance(W, np.ndarray), 'error: ' + \
    #         'the mapping matrix should be a 2D numpy array'
    #         assert W.shape[1] == self.num_pixel, 'error: ' + \
    #         'inconsistency between mapping matrix and SparseImageStar dimension'

    #         V = np.matmul(W, V)
    #         c = np.matmul(W, c)

    #     if b is not None:
    #         assert isinstance(b, np.ndarray), 'error: ' + \
    #         'the offset vector should be a 1D numpy array'
    #         assert len(b.shape) == 1, 'error: ' + \
    #         'offset vector should be a 1D numpy array'

    #         if W is not None:
    #             assert W.shape[0] == b.shape[0], 'error: ' + \
    #             'inconsistency between mapping matrlen(self.Cx and offset'
    #         else:
    #             assert b.shape[0] == self.num_pixel, 'error: ' + \
    #             'inconsistency between offset vector and SparseStar dimension'

    #         c += b

    #     V = SparseImage.from_numpy(V.reshape(self.height, self.width, self.num_channel, self.num_pred))
    #     c = c.reshape(self.height, self.width, self.num_channel)
    #     return SparseImageStar(c, V, self.C, self.d, self.pred_lb, self.pred_ub)
    
    def getRange(self, h_indx, w_indx, c_indx, lp_solver='gurobi'):
        """Get the lower and upper bounds of x[index]"""

        if lp_solver == 'estimate':
            return self.estimateRange(h_indx, w_indx, c_indx)
        else:
            l = self.getMin(h_indx, w_indx, c_indx, lp_solver)
            u = self.getMax(h_indx, w_indx, c_indx, lp_solver)
            return l, u
        
    def getRanges(self, lp_solver='gurobi', RF=0.0, layer=None, delta=0.98):
        """Get the lower and upper bound vectors of the state
            Args:
                lp_solver: linear programming solver. e.g.: 'gurobi', 'estimate', 'linprog'
        """

        if lp_solver == 'estimate':
            l, u = self.estimateRanges()
        else:
            l = self.getMins_all()
            u = self.getMaxs_all()
        return l, u
    
    def getMin(self, *args):
        """Get the minimum value of state x[index] or x[h_indx, w_indx, c_indx] by solving LP
            lp_solver = 'gurobi', 'linprog', or 'glpk'
            h_indx: veritcial index
            w_indx: horizontal index
            c_indx: channel index
            index: flattened index
        """
        len_ = len(args)

        if len_ == 4:
            [h_indx, w_indx, c_indx, lp_solver] = args
            index = None

        elif len_ == 3:
            [h_indx, w_indx, c_indx] = args
            lp_solver = 'gurobi'
            index = None

        elif len_ == 2:
            [index, lp_solver] = args

        elif len_ == 1:
            [index] = args
            lp_solver = 'gurobi'

        else:
            raise Exception(
                'error: invalid number of input arguments (should be between 1 and 4)')

        if index is not None:
            h_indx, w_indx, c_indx = self.index_to3D(index)

        return self.getMin_hwc(h_indx, w_indx, c_indx, lp_solver)
    
    def getMax(self, *args):
        """Get the maximum value of state x[index] or x[h_indx, w_indx, c_indx] by solving LP
            lp_solver = 'gurobi', 'linprog', or 'glpk'
            h_indx: veritcial index
            w_indx: horizontal index
            c_indx: channel index
            index: flattened index
        """
        len_ = len(args)

        if len_ == 4:
            [h_indx, w_indx, c_indx, lp_solver] = args
            index = None

        elif len_ == 3:
            [h_indx, w_indx, c_indx] = args
            lp_solver = 'gurobi'
            index = None

        elif len_ == 2:
            [index, lp_solver] = args

        elif len_ == 1:
            [index] = args
            lp_solver = 'gurobi'

        else:
            raise Exception(
                'error: invalid number of input arguments (should be between 1 and 4)')

        if index is not None:
            h_indx, w_indx, c_indx = self.index_to3D(index)

        return self.getMax_hwc(h_indx, w_indx, c_indx, lp_solver)
    
    def getMin_hwc(self, h_indx, w_indx, c_indx, lp_solver='gurobi'):
        """Get the minimum value of state x[index] or x[h_indx, w_indx, c_indx] by solving LP
            lp_solver = 'gurobi', 'linprog', or 'glpk'
            h_indx: veritcial index
            w_indx: horizontal index
            c_indx: channel index
        """

        assert h_indx >= 0 and h_indx < self.height, \
        'error: invalid vertical index'
        assert w_indx >= 0 and w_indx < self.width, \
        'error: invalid horizontal index'
        assert c_indx >= 0 and c_indx < self.num_channel, \
        'error: invalid channel index'

        # f = self.V.getRows(index).to_2D().toarray()
        # f = self.V.getRow_2D(index, self.num_pred) #.toarray()
        f = self.V.getRow_hwc_2D(h_indx, w_indx, c_indx, self.num_pred)

        # if (f == 0).all():
        if f.nnz == 0:
            xmin = self.c[h_indx, w_indx, c_indx]
        else:
            if lp_solver == 'gurobi':  # using gurobi is the preferred choice

                min_ = gp.Model()
                min_.Params.LogToConsole = 0
                min_.Params.OptimalityTol = GUROBI_OPT_TOL
                if self.pred_lb.size and self.pred_ub.size:
                    x = min_.addMVar(shape=self.num_pred, lb=self.pred_lb, ub=self.pred_ub)
                else:
                    x = min_.addMVar(shape=self.num_pred)
                min_.setObjective(f @ x, GRB.MINIMIZE)
                if len(self.d) > 0:
                    C = self.C
                    d = self.d
                else:
                    C = sp.csr_matrix(np.zeros((1, self.num_pred)))
                    d = 0
                min_.addConstr(C @ x <= d)
                min_.optimize()

                if min_.status == 2:
                    xmin = min_.objVal + self.c[h_indx, w_indx, c_indx]
                else:
                    raise Exception('error: cannot find an optimal solution, ' + \
                        'exitflag = %d' % (min_.status))

            elif lp_solver == 'linprog':

                # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html
                if len(self.d) == 0:
                    A = np.zeros((1, self.num_pred))
                    b = np.zeros(1)
                else:
                    A = self.C.toarray()
                    b = self.d

                lb = self.pred_lb
                ub = self.pred_ub
                lb = lb.reshape((self.num_pred, 1))
                ub = ub.reshape((self.num_pred, 1))
                res = linprog(f, A_ub=A, b_ub=b, bounds=np.hstack((lb, ub)))

                if res.status == 0:
                    xmin = res.fun + self.c[h_indx, w_indx, c_indx]
                else:
                    raise Exception('error: cannot find an optimal solution, \
                    exitflag = {}'.format(res.status))

            elif lp_solver == 'glpk':

                #  https://pyglpk.readthedocs.io/en/latest/examples.html
                #  https://pyglpk.readthedocs.io/en/latest/

                glpk.env.term_on = False

                if len(self.d) == 0:
                    A = np.zeros((1, self.num_pred))
                    b = np.zeros(1)
                else:
                    A = self.C.toarray()
                    b = self.d

                lb = self.pred_lb
                ub = self.pred_ub
                lb = lb.reshape((self.num_pred, 1))
                ub = ub.reshape((self.num_pred, 1))

                lp = glpk.LPX()  # create the empty problem instance
                lp.obj.maximize = False
                lp.rows.add(A.shape[0])  # append rows to this instance
                for r in lp.rows:
                    r.name = chr(ord('p') + r.index)  # name rows if we want
                    lp.rows[r.index].bounds = None, b[r.index]

                lp.cols.add(self.num_pred)
                for c in lp.cols:
                    c.name = 'x%d' % c.index
                    c.bounds = lb[c.index], ub[c.index]

                lp.obj[:] = f.tolist()
                B = A.reshape(A.shape[0]*A.shape[1],)
                lp.matrix = B.tolist()
                # lp.interior()
                lp.simplex()
                # default choice, interior may have a big floating point error

                if lp.status != 'opt':
                    raise Exception('error: cannot find an optimal solution, \
                    lp.status = {}'.format(lp.status))
                else:
                    xmin = lp.obj.value + self.c[h_indx, w_indx, c_indx]
            else:
                raise Exception('error: \
                unknown lp solver, should be gurobi or linprog or glpk')
        return xmin

    def getMax_hwc(self, h_indx, w_indx, c_indx, lp_solver='gurobi'):
        """Get the maximum value of state x[h_indx, w_indx, c_indx] by solving LP
            lp_solver = 'gurobi', 'linprog', or 'glpk'
            h_indx: veritcial index
            w_indx: horizontal index
            c_indx: channel index
        """

        assert h_indx >= 0 and h_indx < self.height, \
        'error: invalid vertical index'
        assert w_indx >= 0 and w_indx < self.width, \
        'error: invalid horizontal index'
        assert c_indx >= 0 and c_indx < self.num_channel, \
        'error: invalid channel index'

        # f = self.V.getRows(index).to_2D().toarray()
        # f = self.V.getRow_2D(index, self.num_pred) #.toarray()
        f = self.V.getRow_hwc_2D(h_indx, w_indx, c_indx, self.num_pred)

        # if (f == 0).all():
        if f.nnz == 0:
            xmax = self.c[h_indx, w_indx, c_indx]
        else:
            if lp_solver == 'gurobi':  # using gurobi is the preferred choice

                max_ = gp.Model()
                max_.Params.LogToConsole = 0
                max_.Params.OptimalityTol = GUROBI_OPT_TOL
                if self.pred_lb.size and self.pred_ub.size:
                    x = max_.addMVar(shape=self.num_pred,lb=self.pred_lb, ub=self.pred_ub)
                else:
                    x = max_.addMVar(shape=self.num_pred)
                max_.setObjective(f @ x, GRB.MAXIMIZE)
                if len(self.d) > 0:
                    C = self.C
                    d = self.d
                else:
                    C = sp.csr_matrix(np.zeros((1, self.num_pred)))
                    d = 0
                max_.addConstr(C @ x <= d)
                max_.optimize()

                if max_.status == 2:
                    xmax = max_.objVal + self.c[h_indx, w_indx, c_indx]
                else:
                    raise Exception('error: cannot find an optimal solution, \
                    exitflag = %d' % (max_.status))
            elif lp_solver == 'linprog':
                # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html
                if len(self.d) == 0:
                    A = np.zeros((1, self.num_pred))
                    b = np.zeros(1)
                else:
                    A = self.C.toarray()
                    b = self.d

                lb = self.pred_lb
                ub = self.pred_ub
                lb = lb.reshape((self.num_pred, 1))
                ub = ub.reshape((self.num_pred, 1))
                res = linprog(-f, A_ub=A, b_ub=b, bounds=np.hstack((lb, ub)))
                if res.status == 0:
                    xmax = -res.fun + self.c[h_indx, w_indx, c_indx]
                else:
                    raise Exception('error: cannot find an optimal solution, \
                    exitflag = {}'.format(res.status))

            elif lp_solver == 'glpk':

                # https://pyglpk.readthedocs.io/en/latest/examples.html
                # https://pyglpk.readthedocs.io/en/latest/

                glpk.env.term_on = False  # turn off messages/display

                if len(self.d) == 0:
                    A = np.zeros((1, self.num_pred))
                    b = np.zeros(1)
                else:
                    A = self.C.toarray()
                    b = self.d

                lb = self.pred_lb
                ub = self.pred_ub
                lb = lb.reshape((self.num_pred, 1))
                ub = ub.reshape((self.num_pred, 1))

                lp = glpk.LPX()  # create the empty problem instance
                lp.obj.maximize = True
                lp.rows.add(A.shape[0])  # append rows to this instance
                for r in lp.rows:
                    r.name = chr(ord('p') + r.index)  # name rows if we want
                    lp.rows[r.index].bounds = None, b[r.index]

                lp.cols.add(self.num_pred)
                for c in lp.cols:
                    c.name = 'x%d' % c.index
                    c.bounds = lb[c.index], ub[c.index]

                lp.obj[:] = f.tolist()
                B = A.reshape(A.shape[0]*A.shape[1],)
                lp.matrix = B.tolist()

                # lp.interior()
                # default choice, interior may have a big floating point error
                lp.simplex()

                if lp.status != 'opt':
                    raise Exception('error: cannot find an optimal solution, \
                    lp.status = {}'.format(lp.status))
                else:
                    xmax = lp.obj.value + self.c[h_indx, w_indx, c_indx]
            else:
                raise Exception('error: \
                unknown lp solver, should be gurobi or linprog or glpk')
        return xmax
    
    def getMins_all(self, lp_solver='gurobi'):
        xmin = np.zeros([self.height, self.height, self.num_channel])
        for h_ in range(self.height):
            for w_ in range(self.width):
                for c_ in range(self.num_channel):
                    xmin[h_, w_, c_] = self.getMin_hwc(h_, w_, c_, lp_solver)
        return xmin.flatten()

    def getMaxs_all(self, lp_solver='gurobi'):
        xmax = np.zeros([self.height, self.height, self.num_channel])
        for h_ in range(self.height):
            for w_ in range(self.width):
                for c_ in range(self.num_channel):
                    xmax[h_, w_, c_] = self.getMax_hwc(h_, w_, c_, lp_solver)
        return xmax.flatten()
    
    def getMins(self, *args):
        """Get the maximum values of state x corresponding map indexes
        """
        len_ = len(args)

        if len_ == 4:
            [h_map, w_map, c_map, lp_solver] = args
            map = None

        elif len_ == 3:
            [h_map, w_map, c_map] = args
            lp_solver = 'gurobi'
            map = None

        elif len_ == 2:
            [map, lp_solver] = args

        elif len_ == 1:
            [map] = args
            lp_solver = 'gurobi'

        else:
            raise Exception(
                'error: invalid number of input arguments (should be between 1 and 4)')

        if map is not None:
            h_map, w_map, c_map = self.V.index_to3D(map)

        n = len(map)
        xmin = np.zeros(n)
        for i in range(n):
            xmin[i] = self.getMin_hwc(h_map[i], w_map[i], c_map[i], lp_solver)
        return xmin

    def getMaxs(self, *args):
        """Get the maximum values of state x corresponding map indexes
        """
        len_ = len(args)

        if len_ == 4:
            [h_map, w_map, c_map, lp_solver] = args
            map = None

        elif len_ == 3:
            [h_map, w_map, c_map] = args
            lp_solver = 'gurobi'
            map = None

        elif len_ == 2:
            [map, lp_solver] = args

        elif len_ == 1:
            [map] = args
            lp_solver = 'gurobi'

        else:
            raise Exception(
                'error: invalid number of input arguments (should be between 1 and 4)')

        if map is not None:
            h_map, w_map, c_map = self.V.index_to3D(map)

        n = len(map)
        xmax = np.zeros(n)
        for i in range(n):
            xmax[i] = self.getMax_hwc(h_map[i], w_map[i], c_map[i], lp_solver)
        return xmax

    def estimateRange(self, h, w, c):
        """Estimate the minimum and maximum values of a state x[index]"""

        l = self.pred_lb.reshape(self.height, self.width, self.num_channel)
        u = self.pred_ub.reshape(self.height, self.width, self.num_channel)

        xmin = copy.deepcopy(self.c[h, w, c])
        xmax = copy.deepcopy(self.c[h, w, c])
        
        for n_ in range(self.V.num_images):
            X = self.V.images[n_].channel[c]
            if X != None:
                pos_f = X.maximum(0.0).tocoo()
                neg_f = X.minimum(0.0).tocoo()
                
                for k in range(pos_f.nnz):
                    i = pos_f.row[k]
                    j = pos_f.col[k]
                    v = pos_f.data[k]
                    xmin += v * l[i, j, c]
                    xmax += v * u[i, j, c]

                for k in range(neg_f.nnz):
                    i = neg_f.row[k]
                    j = neg_f.col[k]
                    v = neg_f.data[k]
                    xmin += v * u[i, j, c]
                    xmax += v * l[i, j, c]
        return xmin, xmax
    
    # def estimateRanges(self):
    #     """Estimate the minimum and maximum values of a state x[index]"""
        
    #     xmin = copy.deepcopy(self.c)
    #     xmax = copy.deepcopy(self.c)

    #     for n_ in range(self.V.num_images):
    #         for c_ in range(self.V.num_channel):
    #             X = self.V.images[n_].channel[c_].tocsr()
    #             nP = self.V.images[n_].pred
    #             if X != None:
    #                 xmin, xmax = SparseImageStar.test_X(X, xmin, xmax, self.pred_lb[nP], self.pred_ub[nP], c_)
    #                 # pos_f = X.maximum(0.0).tocoo() #sp.coo_matrix.maximum() returns sp.csr_matrix
    #                 # neg_f = X.minimum(0.0).tocoo()

    #                 # for k in range(pos_f.nnz):
    #                 #     i = pos_f.row[k]
    #                 #     j = pos_f.col[k]
    #                 #     v = pos_f.data[k]

    #                 #     xmin[i, j, c_] += v * self.pred_lb[nP]
    #                 #     xmax[i, j, c_] += v * self.pred_ub[nP]

    #                 # for k in range(neg_f.nnz):
    #                 #     i = neg_f.row[k]
    #                 #     j = neg_f.col[k]
    #                 #     v = neg_f.data[k]
    #                 #     xmin[i, j, c_] += v * self.pred_ub[nP]
    #                 #     xmax[i, j, c_] += v * self.pred_lb[nP]
    #     return xmin.flatten(), xmax.flatten()
    
    # @staticmethod
    # @jit(nopython=True)
    # def test_X(X, xmin, xmax, pred_lb, pred_ub, c_):
    #     pos_f = X.maximum(0.0) #sp.coo_matrix.maximum() returns sp.csr_matrix
    #     neg_f = X.minimum(0.0)

    #     for k in range(pos_f.nnz):
    #         i = pos_f.row[k]
    #         j = pos_f.col[k]
    #         v = pos_f.data[k]

    #         xmin[i, j, c_] += v * pred_lb
    #         xmax[i, j, c_] += v * pred_ub

    #     for k in range(neg_f.nnz):
    #         i = neg_f.row[k]
    #         j = neg_f.col[k]
    #         v = neg_f.data[k]
    #         xmin[i, j, c_] += v * pred_ub
    #         xmax[i, j, c_] += v * pred_lb
    #     return xmin, xmax
    
    def estimateRanges(self):
        """Estimate the minimum and maximum values of a state x[index]"""
        
        xmin = copy.deepcopy(self.c)
        xmax = copy.deepcopy(self.c)

        for n_ in range(self.V.num_images):
            for c_ in range(self.V.num_channel):
                X = self.V.images[n_].channel[c_]
                nP = self.V.images[n_].pred
                if X != None:
                    pos_f = X.maximum(0.0).tocoo() #sp.coo_matrix.maximum() returns sp.csr_matrix
                    neg_f = X.minimum(0.0).tocoo()

                    for k in range(pos_f.nnz):
                        i = pos_f.row[k]
                        j = pos_f.col[k]
                        v = pos_f.data[k]

                        xmin[i, j, c_] += v * self.pred_lb[nP]
                        xmax[i, j, c_] += v * self.pred_ub[nP]

                    for k in range(neg_f.nnz):
                        i = neg_f.row[k]
                        j = neg_f.col[k]
                        v = neg_f.data[k]
                        xmin[i, j, c_] += v * self.pred_ub[nP]
                        xmax[i, j, c_] += v * self.pred_lb[nP]
        return xmin.flatten(), xmax.flatten()
    
    # def estimateRanges2(self):
    #     xmin = copy.deepcopy(self.c)
    #     xmax = copy.deepcopy(self.c)

    #     for im3d in self.V:
    #         for im2d in im3d.channel:
    #             if im2d is None:
    #                 continue
                
    #             pos_f = im2d.maximum(0.0).tocoo() #sp.coo_matrix.maximum() returns sp.csr_matrix
    #             neg_f = im2d.minimum(0.0).tocoo()

    #             nP = im3d.pred
    #             for k in range(pos_f.nnz):
    #                 i = pos_f.row[k]
    #                 j = pos_f.col[k]
    #                 v = pos_f.data[k]

    #                 xmin[i, j, c_] += v * self.pred_lb[nP]
    #                 xmax[i, j, c_] += v * self.pred_ub[nP]

    #             for k in range(neg_f.nnz):
    #                 i = neg_f.row[k]
    #                 j = neg_f.col[k]
    #                 v = neg_f.data[k]
    #                 xmin[i, j, c_] += v * self.pred_ub[nP]
    #                 xmax[i, j, c_] += v * self.pred_lb[nP]
    #     return xmin.flatten(), xmax.flatten()

    
    def getRange(self, h_indx, w_indx, c_indx, lp_solver='gurobi'):
        """Get the lower and upper bounds of x[index]"""

        if lp_solver == 'estimate':
            return self.estimateRange(h_indx, w_indx, c_indx)
        else:
            l = self.getMin(h_indx, w_indx, c_indx, lp_solver)
            u = self.getMax(h_indx, w_indx, c_indx, lp_solver)
            return l, u    
        
    def getRanges(self, lp_solver='gurobi', RF=0.0, layer=None, delta=0.98):
        """Get the lower and upper bound vectors of the state
            Args:
                lp_solver: linear programming solver. e.g.: 'gurobi', 'estimate', 'linprog'
        """

        if lp_solver == 'estimate':
            l, u = self.estimateRanges()
        else:
            l = self.getMins_all()
            u = self.getMaxs_all()
        return l, u
    
    def resetRow(self, index):
        """Reset a row with index"""

        assert index >= 0 and index < self.dim, \
        'error: invalid index, it should be between {} and {}'.format(0, self.dim - 1)
        
        h_, w_, c_ = self.V.index_to3D(index)
        
        new_V = self.V.resetRow_hwc(h_, w_, c_)
        
        new_c = copy.deepcopy(self.c)
        new_c[h_, w_, c_] = 0
        
        return SparseImageStar(new_c, new_V, self.C, self.d, self.pred_lb, self.pred_ub)

    def resetRows(self, map):
        """Reset a row with a map of indexes"""

        h_map, w_map, c_map = self.V.index_to3D(map)

        new_V =  self.V.resetRows_hwc(h_map, w_map, c_map)

        new_c = copy.deepcopy(self.c)
        for i in range(len(map)):
            new_c[h_map[i], w_map[i], c_map[i]] = 0

        return SparseImageStar(new_c, new_V, self.C, self.d, self.pred_lb, self.pred_ub)


    @staticmethod
    def inf_attack(data, epsilon=0.01, data_type='default', dtype = 'float64'):
        """Generate a SparseImageStar set by infinity norm attack on input dataset"""

        if isinstance(data, np.ndarray):
            assert data.ndim == 3, \
            'error: data should be a 3D numpy array in [height, width, channel] shape'
        
        elif isinstance(data, torch.Tensor):
            assert data.ndim == 3, \
            'error: data should be a 3D torch tensor in [channel, height, width] shape'

            data = data.permute(1, 2, 0).numpy()

        else:
            raise Exception('the data should be a 3D numpy array or 3D torch tensor')

        data = data.astype(dtype)

        lb = data - epsilon
        ub = data + epsilon

        if data_type == 'image':
            lb[lb < 0] = 0
            ub[ub > 1] = 1

        return SparseImageStar(lb, ub)

    @staticmethod
    def rand_bounds(in_height, in_width, in_channel):
        """Generate a random SpareImageStar by random bounds"""

        lb = -np.random.rand(in_height, in_width, in_channel)
        ub = np.random.rand(in_height, in_width, in_channel)
        return SparseImageStar(lb, ub)
    




    # @staticmethod
    # def scipy2coopy(a):
    #     indices = np.row_stack([a.row, a.col])
    #     return torch.sparse_coo_tensor(indices, a.data, a.shape)