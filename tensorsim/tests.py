#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Simple unit tests for tensorsim module.
"""

import tensorsim
import numpy as np
import numpy.testing as npt


def test_rotate_v_about_k():
    v = np.array([1, 0, 0])
    k = np.array([0, 1, 0])
    angle = np.pi / 2
    rotated_v = tensorsim.rotate_v_about_k(v, k, angle)
    npt.assert_almost_equal(rotated_v, np.array([0, 0, -1]))
    return


def test_align_v_with_k():
    N = 100
    np.random.seed(12345)
    ks = np.random.normal(size=(N, 3))
    vs = np.random.normal(size=(N, 3))
    for i in range(N):
        rotated_v = tensorsim.align_v_with_k(vs[i], ks[i])
        npt.assert_almost_equal(np.linalg.norm(rotated_v),
                                np.linalg.norm(vs[i]))
        npt.assert_almost_equal(rotated_v / np.linalg.norm(rotated_v),
                                ks[i] / np.linalg.norm(ks[i]))
    return


def test_rotate_tensor_about_k():
    tensor = np.array([[1, 0, 0],
                       [0, 0, 0],
                       [0, 0, 0]])
    k = np.array([0, 1, 0])
    angle = np.pi / 2
    rotated_tensor = tensorsim.rotate_tensor_about_k(tensor, k, angle)
    npt.assert_almost_equal(rotated_tensor, np.array([[0, 0, 0],
                                                      [0, 0, 0],
                                                      [0, 0, 1]]))
    return


def test_align_tensor_with_vector():
    N = 100
    np.random.seed(12345)
    vs = np.random.normal(size=(N, 3))
    vs /= np.linalg.norm(vs, axis=1)[:, np.newaxis]
    for i in range(N):
        tensor = np.array([[2, 0, 0],
                           [0, 1, 0],
                           [0, 0, 0]])
        for n in range(3):
            rotated_tensor = tensorsim.align_tensor_with_vector(tensor, vs[i],
                                                                n=n + 1)
            evals, evecs = np.linalg.eig(rotated_tensor)
            idx = np.argsort(evals)
            npt.assert_almost_equal(
                np.abs(evecs[:, idx[2 - n]]), np.abs(vs[i]))
    return


def test_calculate_MD():
    tensor = np.array([[2, 0, 0],
                       [0, 1, 0],
                       [0, 0, 0]])
    MD = tensorsim.calculate_MD(tensor)
    npt.assert_almost_equal(MD, np.trace(tensor) / 3)
    return


def test_calculate_FA():
    tensor = np.array([[2, 0, 0],
                       [0, 1, 0],
                       [0, 0, 0]])
    FA = tensorsim.calculate_FA(tensor)
    npt.assert_almost_equal(FA, 0.7745966692414834)
    return


def test_generate_axisymmetric_tensor():
    N = 100
    np.random.seed(12345)
    for _ in range(N):
        FA, MD = np.random.random(size=2)
        tensor = tensorsim.generate_axisymmetric_tensor(FA, MD)
        npt.assert_almost_equal(tensorsim.calculate_FA(tensor), FA)
        npt.assert_almost_equal(tensorsim.calculate_MD(tensor), MD)
    return


def test_generate_b_tensor():
    b = 1e9
    linear_bten = tensorsim.generate_b_tensor('linear', b)
    npt.assert_almost_equal(linear_bten, np.array([[1e9, 0, 0],
                                                   [0, 0, 0],
                                                   [0, 0, 0]]))
    planar_bten = tensorsim.generate_b_tensor('planar', b)
    npt.assert_almost_equal(planar_bten, np.array([[5e8, 0, 0],
                                                   [0, 5e8, 0],
                                                   [0, 0, 0]]))
    spherical_bten = tensorsim.generate_b_tensor('spherical', b)
    npt.assert_almost_equal(spherical_bten, np.array([[1e9, 0, 0],
                                                      [0, 1e9, 0],
                                                      [0, 0, 1e9]]) / 3)
    return


def test_add_noise_to_data():
    N = int(1e5)
    data = np.ones(N)
    for SNR in np.arange(5, 101, 5):
        noisy_data = tensorsim.add_noise_to_data(data, SNR)
        npt.assert_equal(
            np.round(
                np.mean(noisy_data) /
                np.std(noisy_data)),
            SNR)
    return


def test_synthetic_measurement():
    N = int(1e3)
    bs = np.linspace(0, 5e9, N)
    MD = 1e-9
    FA = .75
    D = tensorsim.generate_axisymmetric_tensor(FA, MD)
    sim_S = np.zeros(N)
    expected_S = np.exp(-bs * MD)
    for i, b in enumerate(bs):
        bten = tensorsim.generate_b_tensor('spherical', b)
        sim_S[i] = tensorsim.synthetic_measurement(bten, D)
    npt.assert_almost_equal(sim_S, expected_S)
    return

def test_all():
    test_rotate_v_about_k()
    test_align_v_with_k()
    test_rotate_tensor_about_k()
    test_align_tensor_with_vector()
    test_calculate_MD()
    test_calculate_FA()
    test_generate_axisymmetric_tensor()
    test_generate_b_tensor()
    test_add_noise_to_data()
    test_synthetic_measurement()
    print('All tests passed!')
    return


if __name__ == '__main__':
    test_all()
