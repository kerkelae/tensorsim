#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Functions for analytical multidimensional dMRI simulations.

To learn more about multidimensional diffusion magnetic resonance imaging,
please see Topgaard, Daniel. "Multidimensional diffusion MRI." Journal of
Magnetic Resonance 275 (2017): 98-113.

@author: Leevi Kerkela
"""

import numpy as np


def rotate_v_about_k(v, k, angle):
    """Rotate a vector counter-clockwise about another vector.

    Parameters
    ----------
    v : array_like
        3D vector to be rotated.
    k : array_like
        3D vector about which v is rotated.
    angle : float
        Angle in radians that defines extent of rotation.

    Returns:
    --------
    rotated_v : array_like
        Rotated vector.
    """
    k = k / np.linalg.norm(k)
    rotated_v = v * np.cos(angle) + np.cross(k, v) * np.sin(angle) \
        + k * np.dot(k, v) * (1 - np.cos(angle))
    return rotated_v


def align_v_with_k(v, k):
    """Rotate vector v so that it is aligned with another vector k.

    Parameters
    ----------
    v : array_like
        3D vector that is rotated.
    k : array_like
        3D vector with which v is aligned.

    Returns
    -------
    rotated_v : array_like
        Rotated vector.
    """
    axis = np.cross(v, k)
    axis /= np.linalg.norm(axis)
    angle = np.arcsin(np.linalg.norm(np.cross(v, k)) /
                      (np.linalg.norm(k) * np.linalg.norm(v)))
    if np.dot(v, k) < 0:
        angle = np.pi - angle
    rotated_vector = rotate_v_about_k(v, axis, angle)
    return rotated_vector


def rotate_tensor_about_k(tensor, k, angle):
    """Rotate a tensor counter-clockwise about a vector.

    Parameters
    ----------
    tensor : ndarray
        3 by 3 tensor that is rotated.
    k : array_like
        3D vector about which v is rotated.
    angle : float
        Angle in radians that defines extent of rotation.

    Returns
    -------
    rotated_tensor : ndarray
        Rotated tensor.
    """
    k = k / np.linalg.norm(k)
    K = np.array([[0, -k[2], k[1]],
                  [k[2], 0, -k[0]],
                  [-k[1], k[0], 0]])
    R = np.eye(3) + \
        np.sin(angle) * K + \
        (1 - np.cos(angle)) * np.matmul(K, K)
    rotated_tensor = np.matmul(np.matmul(R, tensor), R.T)
    return rotated_tensor


def align_tensor_with_vector(tensor, v, n=1):
    """Rotate tensor so that its eigenvector corresponding to the n'th
       largest eigenvalue is aligner with vector v.

    Parameters
    ----------
    tensor : ndarray
        3 by 3 tensor that is rotated.
    v : array_like
        3D vector with which tensor is aligned.
    n : int
        Integer between 1 and 3.

    Returns
    -------
    rotated_tensor : ndarray
        Rotated tensor.
    """
    eigenvalues, eigenvectors = np.linalg.eig(tensor)
    eval_1_idx = np.argmax(eigenvalues)
    eval_3_idx = np.argmin(eigenvalues)
    eval_2_idx = np.setdiff1d(np.arange(3), [eval_1_idx,
                                             eval_3_idx])[0]
    if n == 1:
        principal_eigenvector = eigenvectors[:, eval_1_idx]
    elif n == 2:
        principal_eigenvector = eigenvectors[:, eval_2_idx]
    elif n == 3:
        principal_eigenvector = eigenvectors[:, eval_3_idx]
    else:
        raise ValueError('n has to be equal to 1, 2, or 3.')
    v = v / np.linalg.norm(v)
    axis = np.cross(principal_eigenvector, v)
    if np.linalg.norm(axis) == 0:
        return tensor
    axis = axis / np.linalg.norm(axis)
    angle = np.arcsin(np.linalg.norm(np.cross(principal_eigenvector, v)))
    if np.dot(principal_eigenvector, v) < 0:
        angle = np.pi - angle
    rotated_tensor = rotate_tensor_about_k(tensor, axis, angle)
    return rotated_tensor


def calculate_FA(D):
    """Calculate fractional anisotropy of a diffusion tensor.

    Parameters
    ----------
    D : ndarray
        3 x 3 tensor.

    Returns
    -------
    FA : float
        Fractional anisotropy.
    """
    R = D / np.trace(D)
    FA = np.sqrt(.5 * (3 - 1 / np.trace(np.matmul(R, R))))
    return FA


def calculate_MD(D):
    """Calculate mean diffusivity of a diffusion tensor.

    Parameters
    ----------
    D : ndarray
        3 x 3 tensor.

    Returns
    -------
    MD : float
        Mean diffusivity.
    """
    MD = np.trace(D) / 3
    return MD


def generate_axisymmetric_tensor(FA, MD):
    """Generate axially symmetric diffusion tensor.

    Parameters
    ----------
    FA : float
        Fractional anisotropy of the tensor.
    MD : float
        Mean diffusivity of the tensor.

    Returns
    -------
    D : ndarray
        Diffusion tensor.
    """
    y = 1
    A = 1 - 2 * FA**2
    B = -2 * y
    C = (1 - FA**2) * y**2
    x1 = (-B + np.sqrt(B**2 - 4 * A * C)) / (2 * A)
    x2 = (-B - np.sqrt(B**2 - 4 * A * C)) / (2 * A)
    tensor_1 = np.array([[y, 0, 0],
                         [0, x1, 0],
                         [0, 0, x1]])
    tensor_2 = np.array([[y, 0, 0],
                         [0, x2, 0],
                         [0, 0, x2]])
    if np.all(tensor_1 >= 0) and (calculate_FA(tensor_1) - FA) < 1e-5:
        D = tensor_1
    else:
        D = tensor_2
    D /= np.trace(D)
    D *= MD * 3
    return D


def generate_b_tensor(type, b):
    """Return a linear, planar, or spherical b tensor.

    Parameters
    ----------
    type : str
        Define tensor shape. Options: 'linear', 'planar', and 'spherical'.
    b : float
        b-value in SI units

    Returns
    -------
    bten : ndarray
        Measurement tensor.
    """
    if type == 'linear':
        bten = np.array([[1, 0, 0],
                         [0, 0, 0],
                         [0, 0, 0]]) * b
    elif type == 'planar':
        bten = np.array([[1, 0, 0],
                         [0, 1, 0],
                         [0, 0, 0]]) * b / 2
    elif type == 'spherical':
        bten = np.array([[1, 0, 0],
                         [0, 1, 0],
                         [0, 0, 1]]) * b / 3
    return bten


def add_noise_to_data(data, SNR):
    """Add Rician noise to noiseless data to lower SNR to a desired level.

    Parameters
    ----------
    data : ndarray
        Data array.
    SNR : float
        Signal-to-noise ratio.

    Returns
    -------
    noisy_data : ndarray
        Noisy data.
    """
    sigma = 1 / SNR
    noisy_data = np.abs(data + np.random.normal(size=data.shape, scale=sigma,
                                                loc=0)
                        + 1j * np.random.normal(size=data.shape, scale=sigma,
                                                loc=0))
    return noisy_data


def synthetic_measurement(b_tensor, diffusion_tensor):
    """Generate synthetic dMRI signal.

    Parameters
    ----------
    b_tensor : ndarray
        b-tensor.
    diffusion_tensor : ndarray
        Diffusion tensor.

    Returns
    -------
    signal : float
        Simulated signal.
    """
    signal = np.exp(-np.sum(b_tensor * diffusion_tensor))
    return signal


phi = (1 + np.sqrt(5)) / 2
icosahedron = np.array([[0, 1, phi],
                        [0, -1, -phi],
                        [0, 1, -phi],
                        [0, -1, phi],
                        [1, phi, 0],
                        [-1, -phi, 0],
                        [1, -phi, 0],
                        [-1, phi, 0],
                        [phi, 0, 1],
                        [-phi, 0, -1],
                        [phi, 0, -1],
                        [-phi, 0, 1]]) / np.linalg.norm([0, 1, -phi])
