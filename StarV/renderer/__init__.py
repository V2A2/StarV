#########################################################################
##   This file is part of the StarV verifier                           ##
##                                                                     ##
##   Copyright (c) 2025 The StarV Team                                 ##
##   License: BSD-3-Clause                                             ##
#########################################################################
"""
StarV ProbStar Abstract Rendering Package (`StarV.renderer`):
  - Primitives: `PhysicalProbStar`, `AffineEnvelopeTensor`, `sin_env`, `cos_env`,
    `square_env`, `reciprocal_env`, `exp_neg_env`, `relu_env`, `leakyrelu_env`,
    `sigmoid_env`, `softplus_env`, `clamp_env`, `multiply_env`, `divide_env`
  - Pinhole + Lambertian/Phong Renderer: `PinholeCameraMeshRenderer`,
    `lift_image_to_3d_relief_scene`
  - 3D Gaussian Splatting (3DGS) Renderer: `GaussianSplatting3DRenderer`,
    `fit_3d_gaussian_splats_to_image`
  - Volumetric NeRF Ray-Marching Renderer: `VolumetricNeRFRenderer`,
    `fit_nerf_to_image`
  - Split/Exact ProbStar & End-to-End 3D-to-Vision Certification:
    `partition_physical_probstar`, `render_split_probstar_bounds`,
    `certify_3d_to_vision_probstar`, `certify_3d_to_ssnn_pixel_probstar`,
    `evaluate_3d_to_vision_mc`
"""

from StarV.renderer.primitives import (
    PhysicalProbStar,
    AffineEnvelopeTensor,
    sin_env,
    cos_env,
    square_env,
    reciprocal_env,
    exp_neg_env,
    relu_env,
    leakyrelu_env,
    sigmoid_env,
    softplus_env,
    clamp_env,
    multiply_env,
    divide_env,
)
from StarV.renderer.pinhole_renderer import (
    PinholeCameraMeshRenderer,
    lift_image_to_3d_relief_scene,
    compute_so3_rotation_concrete,
    compute_so3_rotation_env,
)
from StarV.renderer.gaussian_splatting import (
    GaussianSplatting3DRenderer,
    fit_3d_gaussian_splats_to_image,
)
from StarV.renderer.nerf_renderer import (
    VolumetricNeRFRenderer,
    fit_nerf_to_image,
)
from StarV.renderer.split_probstar import (
    partition_physical_probstar,
    render_split_probstar_bounds,
    certify_3d_to_vision_probstar,
    certify_3d_to_ssnn_pixel_probstar,
    evaluate_3d_to_vision_mc,
)

__all__ = [
    'PhysicalProbStar',
    'AffineEnvelopeTensor',
    'sin_env',
    'cos_env',
    'square_env',
    'reciprocal_env',
    'exp_neg_env',
    'relu_env',
    'leakyrelu_env',
    'sigmoid_env',
    'softplus_env',
    'clamp_env',
    'multiply_env',
    'divide_env',
    'PinholeCameraMeshRenderer',
    'lift_image_to_3d_relief_scene',
    'compute_so3_rotation_concrete',
    'compute_so3_rotation_env',
    'GaussianSplatting3DRenderer',
    'fit_3d_gaussian_splats_to_image',
    'VolumetricNeRFRenderer',
    'fit_nerf_to_image',
    'partition_physical_probstar',
    'render_split_probstar_bounds',
    'certify_3d_to_vision_probstar',
    'certify_3d_to_ssnn_pixel_probstar',
    'evaluate_3d_to_vision_mc',
]
