from exact_laws.mathematical_tools.derivation import *
from .. import not_implemented_warning as NIW
import pytest
import numpy as np


class TestCdiff:
        
    def test_sin1D_periodic_pres4_point(self):
        x = np.arange(0, 100) / 100 * 2 * np.pi
        fx = np.sin(x)
        cdiff_on_fx = []
        N = len(x)
        for i in range(N):
            tab = [fx[i - 2], fx[i - 1], fx[(i + 1) % N], fx[(i + 2) % N]]
            cdiff_on_fx.append(cdiff(tab, length_case=x[1], precision=4, point=True))
        dfx = np.cos(x)
        precision = x[1] * x[1] * x[1] * x[1]
        assert np.max(np.abs(cdiff_on_fx - dfx)) < precision, f"error on the derivative of a sin"

    def test_sin1D_periodic_pres4(self):
        x = np.arange(0, 100) / 100 * 2 * np.pi
        fx = np.sin(x)
        cdiff_on_fx = cdiff(fx, length_case=x[1], precision=4)
        dfx = np.cos(x)
        precision = x[1] * x[1] * x[1] * x[1]
        assert np.max(np.abs(cdiff_on_fx - dfx)) < precision, f"error on the derivative of a sin"

    def test_sin1D_noperiodic_pres4(self):
        x = np.arange(0, 100) / 100 * np.pi
        fx = np.sin(x)
        cdiff_on_fx = cdiff(fx, length_case=x[1], period=False, precision=4)
        dfx = np.cos(x)
        precision = x[1] * x[1]  # Calculation of edge with precision of 2
        assert np.max(np.abs(cdiff_on_fx - dfx)) < precision, f"error on the derivative of a sin"

    def test_sin1D_periodic_pres2_point(self):
        x = np.arange(0, 100) / 100 * 2 * np.pi
        fx = np.sin(x)
        cdiff_on_fx = []
        N = len(x)
        for i in range(N):
            tab = [fx[i - 1], fx[(i + 1) % N]]
            cdiff_on_fx.append(cdiff(tab, length_case=x[1], precision=2, point=True))
        dfx = np.cos(x)
        precision = x[1] * x[1]
        assert np.max(np.abs(cdiff_on_fx - dfx)) < precision, f"error on the derivative of a sin"

    def test_sin1D_periodic_pres2(self):
        x = np.arange(0, 100) / 100 * 2 * np.pi
        fx = np.sin(x)
        cdiff_on_fx = cdiff(fx, length_case=x[1], precision=2)
        dfx = np.cos(x)
        precision = x[1] * x[1]
        assert np.max(np.abs(cdiff_on_fx - dfx)) < precision, f"error on the derivative of a sin"

    def test_sin1D_noperiodic_pres2(self):
        x = np.arange(0, 100) / 100 * np.pi
        fx = np.sin(x)
        cdiff_on_fx = cdiff(fx, length_case=x[1], period=False, precision=2)
        dfx = np.cos(x)
        precision = x[1] * x[1]
        assert np.max(np.abs(cdiff_on_fx - dfx)) < precision, f"error on the derivative of a sin"

    def test_sin2D_periodic_pres2(self):
        x = np.arange(0, 10)
        y = np.arange(0, 100) / 100 * 2 * np.pi
        fy = np.sin(y)
        f = np.array([fy for i in x])
        cdiff_on_fy = cdiff(f, length_case=y[1], dirr=1, precision=2)
        cdiff_on_fx = cdiff(f, length_case=x[1], dirr=0, precision=2)
        dfy = np.cos(y)
        precision = y[1] * y[1]
        zero_num = 1e-16
        assert np.max(np.abs(cdiff_on_fy[0, :] - dfy)) < precision, f"error on the derivative of dimension 1"
        assert np.max(np.abs(cdiff_on_fx - np.zeros_like(f))) < zero_num, f"error on the derivative of dimension 0"

    def test_sin2D_periodic_pres4(self):
        x = np.arange(0, 10)
        y = np.arange(0, 100) / 100 * 2 * np.pi
        fy = np.sin(y)
        f = np.array([fy for i in x])
        cdiff_on_fy = cdiff(f, length_case=y[1], dirr=1, precision=4)
        cdiff_on_fx = cdiff(f, length_case=x[1], dirr=0, precision=4)
        dfy = np.cos(y)
        precision = y[1] * y[1] * y[1] * y[1]
        zero_num = 1e-16
        assert np.max(np.abs(cdiff_on_fy[0, :] - dfy)) < precision, f"error on the derivative of dimension 1"
        assert np.max(np.abs(cdiff_on_fx - np.zeros_like(f))) < zero_num, f"error on the derivative of dimension 0"


class TestDiv:
        
    def test_sin3D_periodic_pres4(self):
        x = np.arange(0, 100) / 100 * 2 * np.pi
        sinx = np.sin(x)
        cosx = np.cos(x)
        fx = np.array([sinx[i] for i in range(100) for j in range(100) for k in range(100)]).reshape((100, 100, 100))
        fy = np.array([2 * sinx[j] for i in range(100) for j in range(100) for k in range(100)]).reshape(
            (100, 100, 100))
        fz = np.array([3 * sinx[k] for i in range(100) for j in range(100) for k in range(100)]).reshape(
            (100, 100, 100))
        expected_result = np.array(
            [cosx[i] + 2 * cosx[j] + 3 * cosx[k] for i in range(100) for j in range(100) for k in range(100)]).reshape(
            (100, 100, 100))
        result = div([fx, fy, fz], case_vec=[x[1], x[1], x[1]])
        precision = x[1] * x[1] * x[1] * x[1]
        assert np.max(np.abs(result - expected_result)) < precision, f"error on div computation"


class TestRot:
        
    def test_lin3D_noperiodic_pres4(self):
        x = np.arange(0, 100) / 100 * 2 * np.pi
        fx = np.array([x[i] + 2 * x[j] + 4 * x[k] for i in range(100) for j in range(100) for k in range(100)]).reshape(
            (100, 100, 100))
        fy = np.array([x[i] + 2 * x[j] + 4 * x[k] for i in range(100) for j in range(100) for k in range(100)]).reshape(
            (100, 100, 100))
        fz = np.array([x[i] + 2 * x[j] + 4 * x[k] for i in range(100) for j in range(100) for k in range(100)]).reshape(
            (100, 100, 100))
        expected_result_x = np.array([-2 for i in range(100) for j in range(100) for k in range(100)]).reshape(
            (100, 100, 100))
        expected_result_y = np.array([3 for i in range(100) for j in range(100) for k in range(100)]).reshape(
            (100, 100, 100))
        expected_result_z = np.array([-1 for i in range(100) for j in range(100) for k in range(100)]).reshape(
            (100, 100, 100))
        result_x, result_y, result_z = rot([fx, fy, fz], case_vec=[x[1], x[1], x[1]], period=False)
        precision = x[1] * x[1]
        assert np.max(np.abs(result_x - expected_result_x)) < precision, f"error on rot computation direction x"
        assert np.max(np.abs(result_y - expected_result_y)) < precision, f"error on rot computation direction y"
        assert np.max(np.abs(result_z - expected_result_z)) < precision, f"error on rot computation direction z"

    def test_sin3D_periodic_pres4(self):
        npoint = 100
        x = np.arange(0, npoint) / npoint * 2 * np.pi
        sinx = np.sin(x)
        cosx = np.cos(x)
        fx = np.array(
            [sinx[i] * sinx[j] * sinx[k] for i in range(npoint) for j in range(npoint) for k in range(npoint)]).reshape(
            (npoint, npoint, npoint))
        fy = np.array([2 * sinx[i] * sinx[j] * sinx[k] for i in range(npoint) for j in range(npoint) for k in
                       range(npoint)]).reshape((npoint, npoint, npoint))
        fz = np.array([3 * sinx[i] * sinx[j] * sinx[k] for i in range(npoint) for j in range(npoint) for k in
                       range(npoint)]).reshape((npoint, npoint, npoint))
        expected_result_x = np.array(
            [3 * sinx[i] * cosx[j] * sinx[k] - 2 * sinx[i] * sinx[j] * cosx[k] for i in range(npoint) for j in
             range(npoint) for k in range(npoint)]).reshape((npoint, npoint, npoint))
        expected_result_y = np.array(
            [sinx[i] * sinx[j] * cosx[k] - 3 * cosx[i] * sinx[j] * sinx[k] for i in range(npoint) for j in range(npoint)
             for k in range(npoint)]).reshape((npoint, npoint, npoint))
        expected_result_z = np.array(
            [2 * cosx[i] * sinx[j] * sinx[k] - sinx[i] * cosx[j] * sinx[k] for i in range(npoint) for j in range(npoint)
             for k in range(npoint)]).reshape((npoint, npoint, npoint))
        result_x, result_y, result_z = rot([fx, fy, fz], case_vec=[x[1], x[1], x[1]], precision=4)
        precision = x[1] * x[1] * x[1] * x[1]
        assert np.max(np.abs(result_x - expected_result_x)) < precision, f"error on rot computation direction x"
        assert np.max(np.abs(result_y - expected_result_y)) < precision, f"error on rot computation direction y"
        assert np.max(np.abs(result_z - expected_result_z)) < precision, f"error on rot computation direction z"


class TestGrad:
        
    def test_sin3D_periodic_pres4(self):
        x = np.arange(0, 100) / 100 * 2 * np.pi
        sinx = np.sin(x)
        cosx = np.cos(x)
        f = np.array([sinx[i] * sinx[j] * sinx[k] for i in range(100) for j in range(100) for k in range(100)]).reshape(
            (100, 100, 100))
        expected_result_x = np.array(
            [cosx[i] * sinx[j] * sinx[k] for i in range(100) for j in range(100) for k in range(100)]).reshape(
            (100, 100, 100))
        expected_result_y = np.array(
            [sinx[i] * cosx[j] * sinx[k] for i in range(100) for j in range(100) for k in range(100)]).reshape(
            (100, 100, 100))
        expected_result_z = np.array(
            [sinx[i] * sinx[j] * cosx[k] for i in range(100) for j in range(100) for k in range(100)]).reshape(
            (100, 100, 100))
        result_x, result_y, result_z = grad(f, case_vec=[x[1], x[1], x[1]])
        precision = x[1] * x[1] * x[1] * x[1]
        assert np.max(np.abs(result_x - expected_result_x)) < precision, f"error on grad computation direction x"
        assert np.max(np.abs(result_y - expected_result_y)) < precision, f"error on grad computation direction y"
        assert np.max(np.abs(result_z - expected_result_z)) < precision, f"error on grad computation direction z"
