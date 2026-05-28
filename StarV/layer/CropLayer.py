#########################################################################
##   This file is part of the StarV verifier                           ##
##                                                                     ##
##   Copyright (c) 2025 The StarV Team                                 ##
##   License: BSD-3-Clause                                             ##
##                                                                     ##
##   Primary contacts: Hoang Dung Tran <dungtran@ufl.edu> (UF)         ##
##                     Sung Woo Choi <sungwoo.choi@ufl.edu> (UF)       ##
##                     Yuntao Li <yli17@ufl.edu> (UF)                  ##
##                     Qing Liu <qliu1@ufl.edu> (UF)                   ##
##                                                                     ##
##   See CONTRIBUTORS for full author contacts and affiliations.       ##
##   This program is licensed under the BSD 3‑Clause License; see the  ##
##   LICENSE file in the root directory.                               ##
#########################################################################
"""
Crop Layer Class
Sung Woo Choi, 11/10/2025 
"""

import copy
from turtle import left, width
import torch
import torchvision.transforms.functional as F
import numpy as np
import scipy.sparse as sp

from StarV.set.imagestar import ImageStar
from StarV.set.sparseimagestar2dcoo import SparseImageStar2DCOO
from StarV.set.sparseimagestar2dcsr import SparseImageStar2DCSR

class CropLayer(object):
    """CropLayer class (similiar to Pytorch Crop)
        Author: Sung Woo Choi
        Date: 11/10/2025
    """

    def __init__(self, top=0, left=0, height=0, width=0):
        # assert isinstance(top, int) and top >= 0, "Top must be a non-negative integer."
        # assert isinstance(left, int) and left >= 0, "Left must be a non-negative integer."
        # assert isinstance(height, int) and height >= 0, "Height must be a non-negative integer."
        # assert isinstance(width, int) and width >= 0, "Width must be a non-negative integer."

        self.top = top
        self.left = left
        self.height = height
        self.width = width

    def evaluate(self, x):
        x = copy.deepcopy(x)
        """ crops x """
        if x.ndim == 3:
            x = x[self.top:self.top+self.height, self.left:self.left+self.width,  :]
            x = np.pad(x, ((0, max(0, self.height - x.shape[0])), (0, max(0, self.width - x.shape[1])), (0,0)), mode='constant')

        elif x.ndim == 4:
            x = x[self.top:self.top+self.height, self.left:self.left+self.width, :, :]
            x = np.pad(x, ((0, max(0, self.height - x.shape[0])), (0, max(0, self.width - x.shape[1])), (0,0), (0,0)), mode='constant')

        return x
    
    def crop_pytorch(self, V):
        # V = copy.deepcopy(V).transpose(3, 2, 0, 1)  # to pytorch format [B, C, H, W]
        # V = torch.from_numpy(V)
        # return F.crop(V, self.top, self.left, self.height, self.width).numpy().transpose(2, 3, 1, 0)  # back to original format
        
        # height = V.shape[0] - self.height if self.height < 0 else self.height
        # width = V.shape[1] - self.width if self.width < 0 else self.width


        if self.height <= 0:
            height = V.shape[0] + self.height
        else:
            height = self.height

        if self.width <= 0:
            width = V.shape[1] + self.width
        else:
            width = self.width

        # height = self.height
        # width = self.width
        V = copy.deepcopy(V).transpose(3, 2, 0, 1)  # to pytorch format [B, C, H, W]
        V = torch.from_numpy(V)
        return F.crop(V, self.top, self.left, height, width).numpy().transpose(2, 3, 1, 0)  # back to original format
    
    def crop_sparse(self, V, shape):
        V = copy.deepcopy(V)

        H, W, C = shape
        n_rows, n_cols = V.shape
        assert n_rows == np.prod(shape), "Input shape does not match the provided shape."

        height, width = self.height, self.width
        out_shape = (height, width, C)

        # height = V.shape[0] - self.height if self.height < 0 else self.height
        # width = V.shape[1] - self.width if self.width < 0 else self.width
        

        # height = self.height if self.height > 0 else H - self.top
        # width = self.width if self.width > 0 else W - self.left

        if self.height <= 0:
            height = H + self.height
        else:
            height = self.height

        if self.width <= 0:
            width = W + self.width 
        else:
            width = self.width
        
        # If cropping does nothing, just return original
        if self.top == 0 and self.left == 0 and height == H and width == W:
            return V, out_shape

        # Empty target
        if n_rows == 0:
            if V.format == 'coo':
                return sp.coo_array((height * width * C, n_cols), dtype=V.dtype), out_shape
            elif V.format == 'csr':
                return sp.csr_array((n_rows, n_cols), dtype=V.dtype), out_shape

        if V.format == 'coo':
            new_row  = V.row
            new_col  = V.col
            new_data = V.data

            if self.top > 0:
                row_per_height = W * C
                shift_top = self.top * row_per_height

                # capturing only rows within the crop height
                mask = new_row >= shift_top

                if mask.any():
                    new_row  = new_row[mask] - shift_top
                    new_col  = new_col[mask]
                    new_data = new_data[mask]
                else:
                    # If no valid mask is found, return an empty COO array
                    return sp.coo_array((height * width * C, n_cols)), out_shape
                
            if self.left > 0:
                shift_left = self.left * C

                # capturing only rows within the crop width
                mask = (new_row % row_per_height) >= shift_left

                if mask.any():
                    new_row  = new_row[mask] - shift_left
                    new_col  = new_col[mask]
                    new_data = new_data[mask]
                else:
                    # If no valid mask is found, return an empty COO array
                    return sp.coo_array((height * width * C, n_cols)), out_shape
            return sp.coo_array((new_data, (new_row, new_col)), shape=(height * width * C, n_cols), copy=False), out_shape
        
        elif V.format == 'csr':

            bottom = self.top + height
            right  = self.left + width

            indptr  = V.indptr
            indices = V.indices
            data    = V.data

            # Preallocate to the worst-case (keep everything), then trim at the end.
            cap = data.size
            new_data    = np.zeros(cap, dtype=data.dtype)
            new_indices = np.zeros(cap, dtype=indices.dtype)
            new_indptr  = np.zeros(n_rows + 1, dtype=indptr.dtype)

            write_ptr = 0
            last_filled_row = -1  # highest new row whose pointer we've set
            new_indptr[0] = 0     # will be overwritten consistently by the first fill-slice

            for r in range(n_rows):
                start, end = indptr[r], indptr[r + 1]
                # Skip empty rows
                if start == end:
                    continue

                # Decode (h, w, c) with ravel arithmetic
                hw, c = divmod(r, C)
                h,  w = divmod(hw, W)

                # Check if (h, w) is inside crop rectangle
                if (h < self.top) or (h >= bottom) or (w < self.left) or (w >= right):
                    continue

                # Compute new row index (ravel arithmetic)
                new_h = h - self.top
                new_w = w - self.left
                new_r = (new_h * width + new_w) * C + c

                # Advance indptr for any skipped new rows (including setting start for new_r)
                if new_r > last_filled_row:
                    new_indptr[last_filled_row + 1 : new_r + 1] = write_ptr
                    last_filled_row = new_r

                nnz = end - start # nnz in this row
                new_data[write_ptr : write_ptr + nnz] = data[start:end]
                new_indices[write_ptr : write_ptr + nnz] = indices[start:end]
                write_ptr += nnz

            # Finalize indptr for trailing empty rows
            new_indptr[last_filled_row + 1 : n_rows + 1] = write_ptr

            # Trim over-allocation
            new_data    = new_data[:write_ptr]
            new_indices = new_indices[:write_ptr]

            return sp.csr_array((new_data, new_indices, new_indptr), shape=(n_rows, n_cols), dtype=V.dtype, copy=False), out_shape
    
        else:
            raise NotImplementedError("crop_sparse is only implemented for COO and CSR formats.")


    # def crop_sparse_3d(self, V, shape):
    #     V = copy.deepcopy(V)

    #     if V.format == 'coo':
    #         # converting row indices to 3D indices (H, W, CH) is very memory intensive
    #         h, w, ch = np.unravel_index(V.row, shape, order='C')
    #         mask = (h >= self.top) & (h < self.top + self.height) & (w >= self.left) & (w < self.left + self.width)

    #         if not np.any(mask):
    #             return sp.coo_array((self.height * self.width * shape[2], V.shape[1]))
    #         new_h = h[mask] - self.top
    #         new_w = w[mask] - self.left
    #         new_ch = ch[mask]
    #         new_row = np.ravel_multi_index((new_h, new_w, new_ch), (self.height, self.width, shape[2]))
    #         new_col = V.col[mask]
    #         new_data = V.data[mask]
    #         return sp.coo_array((new_data, (new_row, new_col)), shape=(self.height * self.width * shape[2], V.shape[1]))
        
    #     elif V.format == 'csr':


    #     else:
    #         raise NotImplementedError("crop_sparse_3d is only implemented for COO and CSR formats.")

    # def crop_sparse(self, V, shape):
    #     V = copy.deepcopy(V)

    #     if V.format == 'coo-working':
    #         new_row = V.row
    #         new_col = V.col
    #         new_data = V.data

    #         if self.top > 0:
    #             # capturing only rows within the crop height
    #             idx = (new_row // (shape[1] * shape[2])).astype(bool) 
    #             shift_top = self.top * shape[1] * shape[2]
    #             new_row = new_row[idx]
    #             # shift up by top
    #             temp = new_row - shift_top >= 0
    #             new_row = new_row[temp] - shift_top
    #             idx[idx] = temp

    #             new_col = new_col[idx]
    #             new_data = new_data[idx]
            
    #         if self.left > 0:
    #             # capturing only rows within the crop width
    #             idx = (new_row // shape[2] % shape[1]).astype(bool)
    #             shift_left = self.left * shape[2]
    #             new_row = new_row[idx]
    #             # shift left by left
    #             temp = new_row - shift_left >= 0
    #             new_row = new_row[temp] - shift_left
    #             idx[idx] = temp
    #             new_col = new_col[idx]
    #             new_data = new_data[idx]

    #         return sp.coo_matrix((new_data, (new_row, new_col)), shape=(self.height * self.width * shape[2], V.shape[1]), copy=False)
        
    #     elif V.format == 'csr-working':
    #         if self.top > 0:
    #             # capturing only rows within the crop height
    #             shift_top = self.top * shape[1] * shape[2]
    #             row_indices = np.repeat(np.arange(V.shape[0]), np.diff(new_indptr))
    #             mask = (row_indices // (shape[1] * shape[2]) >= self.top)
    #             new_data = new_data[mask]
    #             new_indices = new_indices[mask]
    #             row_indices = row_indices[mask]
    #             # shift up by top
    #             new_row = row_indices - shift_top
    #             valid_mask = new_row >= 0
    #             new_data = new_data[valid_mask]
    #             new_indices = new_indices[valid_mask]
    #             new_row = new_row[valid_mask]

    #             # Rebuild indptr
    #             unique, counts = np.unique(new_row, return_counts=True)
    #             new_indptr = np.zeros(V.shape[0] + 1, dtype=int)
    #             new_indptr[unique + 1] = counts
    #             new_indptr = np.cumsum(new_indptr)

    #         if self.left > 0:
    #             # capturing only rows within the crop width
    #             shift_left = self.left * shape[2]
    #             row_indices = np.repeat(np.arange(V.shape[0]), np.diff(new_indptr))
    #             mask = (row_indices // shape[2] % shape[1] >= self.left)
    #             new_data = new_data[mask]
    #             new_indices = new_indices[mask]
    #             row_indices = row_indices[mask]
    #             # shift left by left
    #             new_row = row_indices - shift_left
    #             valid_mask = new_row >= 0
    #             new_data = new_data[valid_mask]
    #             new_indices = new_indices[valid_mask]
    #             new_row = new_row[valid_mask]

    #             # Rebuild indptr
    #             unique, counts = np.unique(new_row, return_counts=True)
    #             new_indptr = np.zeros(V.shape[0] + 1, dtype=int)
    #             new_indptr[unique + 1] = counts
    #             new_indptr = np.cumsum(new_indptr)
        
        # elif V.format == 'coo': # O(nnz)
        #     new_row = V.row
        #     new_col = V.col
        #     new_data = V.data

        #     if self.top > 0:
        #         row_per_height = shape[1] * shape[2]
        #         shift_top = self.top * row_per_height

        #         # capturing only rows within the crop height
        #         mask = new_row >= shift_top

        #         if mask.any():
        #             new_row = new_row[mask] - shift_top
        #             new_col = new_col[mask]
        #             new_data = new_data[mask]
        #         else:
        #             # If no valid mask is found, return an empty COO array
        #             return sp.coo_array((self.height * self.width * shape[2], V.shape[1]))
                
        #     if self.left > 0:
        #         row_per_width = shape[2]
        #         shift_left = self.left * row_per_width

        #         # capturing only rows within the crop width
        #         mask = (new_row % row_per_height) >= shift_left

        #         if mask.any():
        #             new_row = new_row[mask] - shift_left
        #             new_col = new_col[mask]
        #             new_data = new_data[mask]
        #         else:
        #             # If no valid mask is found, return an empty COO array
        #             return sp.coo_array((self.height * self.width * shape[2], V.shape[1]))

        #     return sp.coo_array((new_data, (new_row, new_col)), shape=(self.height * self.width * shape[2], V.shape[1]), copy=False)
        
        # elif V.format == 'csr-fix-left':
            
        #     new_data = V.data
        #     new_indices = V.indices
        #     new_indptr = V.indptr

        #     if self.top > 0:
        #         rows_per_height = shape[1] * shape[2]   # rows per image row (width * channels)
        #         shift_top = self.top * rows_per_height  # number of rows to shift (crop from top)

        #         if shift_top >= V.shape[0]:
        #             return sp.csr_matrix((self.height * self.width * shape[2], V.shape[1]))
        #         else:
        #             # non-zero offset of the first row after cropping
        #             start_nnz = new_indptr[shift_top]
        #             new_data = new_data[start_nnz:]
        #             new_indices = new_indices[start_nnz:]
                    
        #             kept_indptr = new_indptr[shift_top:] - start_nnz # length = (height - shift_top + 1)
        #             new_indptr = np.empty_like(new_indptr)

        #             new_indptr[: V.shape[0] - shift_top + 1] = kept_indptr
        #             new_indptr[V.shape[0] - shift_top + 1:] = kept_indptr[-1]


        #     if self.left > 0:
        #         # capturing only rows within the crop width
        #         shift_left = self.left * shape[2]
        #         row_indices = np.repeat(np.arange(V.shape[0]), np.diff(new_indptr))
        #         mask = (row_indices // shape[2] % shape[1] >= self.left)
        #         new_data = new_data[mask]
        #         new_indices = new_indices[mask]
        #         row_indices = row_indices[mask]
        #         # shift left by left
        #         new_row = row_indices - shift_left
        #         valid_mask = new_row >= 0
        #         new_data = new_data[valid_mask]
        #         new_indices = new_indices[valid_mask]
        #         new_row = new_row[valid_mask]

        #         # Rebuild indptr
        #         unique, counts = np.unique(new_row, return_counts=True)
        #         new_indptr = np.zeros(V.shape[0] + 1, dtype=int)
        #         new_indptr[unique + 1] = counts
        #         new_indptr = np.cumsum(new_indptr)

        #     # if self.left > 0:
        #     #     # number of channels to shift (crop from left)
        #     #     shift_left = self.left * shape[2]
        #     #     if shape[1] <= self.left:
        #     #         return sp.csr_matrix((self.height * self.width * shape[2], V.shape[1]))
        #     #     else:
        #     #         row_cnt = new_indptr.shape[0] - 1
        #     #         new_data_list = []
        #     #         new_indices_list = []
        #     #         new_indptr = np.zeros(row_cnt + 1, dtype=V.indptr.dtype)
        #     #         for r in range(row_cnt):
        #     #             # Compute the start and end indices for the current row
        #     #             start, end = new_indptr[r], new_indptr[r + 1]
        #     #             if start == end:
        #     #                 new_indptr[r + 1] = new_indptr[r]
        #     #                 continue
        #     #             row_indices = new_indices[start:end]
        #     #             row_data = new_data[start:end]

        #     #             # Create a mask for the current row
        #     #             mask = row_indices >= shift_left
        #     #             filtered_indices = row_indices[mask] - shift_left
        #     #             filtered_data = row_data[mask]
        #     #             # Append filtered data and indices to the lists
        #     #             new_data_list.append(filtered_data)
        #     #             new_indices_list.append(filtered_indices)
        #     #             new_indptr[r + 1] = new_indptr[r] + len(filtered_data)

        #     #         new_data = np.concatenate(new_data_list) if new_data_list else np.array([], dtype=V.data.dtype)
        #     #         new_indices = np.concatenate(new_indices_list) if new_indices_list else np.array([], dtype=V.indices.dtype)

        # elif V.format == 'csr':
        #     H, W, C = shape
        #     n_rows, n_cols = V.shape
        #     assert n_rows == H * W * C

        #     # If cropping does nothing, just return original
        #     if self.top == 0 and self.left == 0 and self.height == H and self.width == W:
        #         return V

        #     bottom = self.top + self.height
        #     right  = self.left + self.width
        #     new_H, new_W = self.height, self.width
        #     new_n_rows = new_H * new_W * C

        #     # Empty target
        #     if new_n_rows == 0:
        #         return sp.csr_array((n_rows, n_cols), dtype=V.dtype)

        #     indptr  = V.indptr
        #     indices = V.indices
        #     data    = V.data

        #     # Preallocate to the worst-case (keep everything), then trim at the end.
        #     cap = data.size
        #     new_data    = np.empty(cap, dtype=data.dtype)
        #     new_indices = np.empty(cap, dtype=indices.dtype)
        #     new_indptr  = np.empty(new_n_rows + 1, dtype=indptr.dtype)

        #     write_ptr = 0
        #     last_filled_row = -1  # highest new row whose pointer we've set
        #     new_indptr[0] = 0     # will be overwritten consistently by the first fill-slice

        #     for r in range(n_rows):
        #         start, end = indptr[r], indptr[r + 1]
        #         # Skip empty rows
        #         if start == end:
        #             continue

        #         # Decode (h, w, c) with ravel arithmetic
        #         hw, c = divmod(r, C)
        #         h,  w = divmod(hw, W)

        #         # Check if (h, w) is inside crop rectangle
        #         if (h < self.top) or (h >= bottom) or (w < self.left) or (w >= right):
        #             continue

        #         # Compute new row index
        #         new_h = h - self.top
        #         new_w = w - self.left
        #         new_r = ((new_h * new_W) + new_w) * C + c

        #         # Advance indptr for any skipped new rows (including setting start for new_r)
        #         if new_r > last_filled_row:
        #             new_indptr[last_filled_row + 1 : new_r + 1] = write_ptr
        #             last_filled_row = new_r

        #         # Copy this row's nnz block
        #         k = end - start
        #         new_data[write_ptr : write_ptr + k] = data[start:end]
        #         new_indices[write_ptr : write_ptr + k] = indices[start:end]
        #         write_ptr += k

        #     # Finalize indptr for trailing empty rows
        #     new_indptr[last_filled_row + 1 : new_n_rows + 1] = write_ptr

        #     # Trim over-allocation
        #     new_data    = new_data[:write_ptr]
        #     new_indices = new_indices[:write_ptr]

        #     return sp.csr_array((new_data, new_indices, new_indptr), shape=(new_n_rows, n_cols), dtype=V.dtype, copy=False)

        # elif V.format == 'csr':
        #     H, W, C = shape
        #     n_rows, n_cols = V.shape
        #     assert n_rows == H * W * C

        #     # If cropping does nothing, just return original
        #     if self.top == 0 and self.left == 0 and self.height == H and self.width == W:
        #         return V

        #     bottom = self.top + self.height
        #     right = self.left + self.width

        #     new_H, new_W = self.height, self.width
        #     new_n_rows = new_H * new_W * C
    
        #     data = V.data
        #     indices = V.indices
        #     indptr = V.indptr

        #     # ---------- 1st pass: count nnz per *new* row (O(#rows)) ----------
        #     row_counts = np.zeros(new_n_rows, dtype=np.int64)

        #     for r in range(n_rows):
        #         start = indptr[r]
        #         end = indptr[r + 1]
        #         if start == end:
        #             continue  # no nnz in this row

        #         # decode (h, w, c) from row index r
        #         hw, c = divmod(r, C)
        #         h, w = divmod(hw, W)

        #         # check if (h, w) is inside crop rectangle
        #         if h < self.top or h >= bottom or w < self.left or w >= right:
        #             continue

        #         new_h = h - self.top
        #         new_w = w - self.left
        #         new_r = ((new_h * new_W) + new_w) * C + c

        #         row_counts[new_r] += (end - start)

        #     # build new_indptr from row_counts
        #     new_indptr = np.empty(new_n_rows + 1, dtype=indptr.dtype)
        #     new_indptr[0] = 0
        #     np.cumsum(row_counts, out=new_indptr[1:])

        #     nnz_new = int(new_indptr[-1])

        #     # If everything got cropped away
        #     if nnz_new == 0:
        #         return sp.csr_matrix((new_n_rows, n_cols), dtype=V.dtype)

        #     # allocate new data/index arrays
        #     new_data = np.empty(nnz_new, dtype=data.dtype)
        #     new_indices = np.empty(nnz_new, dtype=indices.dtype)

        #     # working copy of write positions per new row
        #     write_pos = new_indptr[:-1].copy()

        #     # ---------- 2nd pass: actually copy data (O(#rows + nnz_kept)) ----------
        #     for r in range(n_rows):
        #         start = indptr[r]
        #         end = indptr[r + 1]
        #         if start == end:
        #             continue

        #         hw, c = divmod(r, C)
        #         h, w = divmod(hw, W)

        #         if h < self.top or h >= bottom or w < self.left or w >= right:
        #             continue

        #         new_h = h - self.top
        #         new_w = w - self.left
        #         new_r = ((new_h * new_W) + new_w) * C + c

        #         k = end - start
        #         dst = write_pos[new_r]
        #         new_data[dst:dst + k] = data[start:end]
        #         new_indices[dst:dst + k] = indices[start:end]
        #         write_pos[new_r] += k

        #     return sp.csr_array((new_data, new_indices, new_indptr), shape=(new_n_rows, n_cols), dtype=V.dtype, copy=False)
        
        # else:
        #     raise NotImplementedError("crop_sparse is only implemented for COO and CSR formats.")

    def reachSingleInput(self, In):
        if isinstance(In, ImageStar):
            V = self.evaluate(In.V)
            return ImageStar(V, In.C, In.d, In.pred_lb, In.pred_ub)
        
        elif isinstance(In, SparseImageStar2DCOO):
            V, shape = self.imcrop_sparse(In.V)
            return SparseImageStar2DCOO(V, In.C, In.d, In.pred_lb, In.pred_ub, shape)

        elif isinstance(In, SparseImageStar2DCSR):
            V, shape = self.imcrop_sparse(In.V)
            return SparseImageStar2DCSR(V, In.C, In.d, In.pred_lb, In.pred_ub, shape)
        
        else:
            raise TypeError("Unsupported input type for CropLayer reachability analysis.")
    
    def reach(self, in_sets, method=None, lp_solver='gurobi', pool=None, RF=0.0, DR=0, show=False):
        """ reachability analysis of Crop layer """

        if isinstance(in_sets, list):
            out_sets = []
            for In in in_sets:
                out_sets.append(self.reachSingleInput(In))
            return out_sets

        else:
            return self.reachSingleInput(in_sets) 