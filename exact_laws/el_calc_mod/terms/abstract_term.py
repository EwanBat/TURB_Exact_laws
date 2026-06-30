from typing import List
from numba import njit, prange, get_num_threads, set_num_threads
import numpy as np

class AbstractTerm:
    def __init__(self):
        pass

    def calc(self, *args, **kwargs) -> (float or List[float]):
        raise NotImplementedError("You have to reimplement this method")
    
    def calc_fourier(self, *args, **kwargs) -> (float or List[float]):
        raise NotImplementedError("You have to reimplement this method")

    def variables(self) -> List[str]:
        raise NotImplementedError("You have to reimplement this method")

    def calc_incr_traj(self, *args, **kwargs) -> (float or List[float]):
        raise NotImplementedError("You have to reimplement this method")
    
    def calc_filter(self, *args, **kwargs) -> (float or List[float]):
        raise NotImplementedError("You have to reimplement this method")

def load():
    return AbstractTerm()


@njit(parallel=True)
def calc_source_with_numba(funct, dx, dy, dz, Nx, Ny, Nz, *quantities):
    acc = 0.0

    for i in prange(Nx):
        for j in prange(Ny):
            for k in range(Nz):
                ip = i + dx - Nx * (i + dx >= Nx)
                jp = j + dy - Ny * (j + dy >= Ny)
                kp = k + dz - Nz * (k + dz >= Nz)
                acc += funct(i, j, k, ip, jp, kp, *quantities)

    return acc / (Nx * Ny * Nz)


@njit(parallel=True)
def calc_flux_with_numba(funct, dx, dy, dz, Nx, Ny, Nz, *quantities):
    acc_x = 0.0
    acc_y = 0.0
    acc_z = 0.0

    for i in prange(Nx):
        for j in prange(Ny):
            for k in range(Nz):
                ip = i + dx - Nx * (i + dx >= Nx)
                jp = j + dy - Ny * (j + dy >= Ny)
                kp = k + dz - Nz * (k + dz >= Nz)
                x, y, z = funct(i, j, k, ip, jp, kp, *quantities)
                acc_x += x
                acc_y += y
                acc_z += z

    return [acc_x / (Nx * Ny * Nz), acc_y / (Nx * Ny * Nz), acc_z / (Nx * Ny * Nz)]

@njit(parallel=True)
def calc_source_with_numba_traj(funct, n_points, n_trajectories, *quantities):
    acc = np.zeros((n_trajectories, n_points))
    for dl in prange(n_points):
        for t in prange(n_points):
            tp = t + (n_points + dl) - n_points * (t + n_points + dl >= 2 * n_points)
            acc[:,dl] += funct(t, tp, *quantities)

    return acc / n_points

@njit(parallel=True)
def calc_flux_with_numba_traj(funct, n_points, n_trajectories, *quantities):
    acc = np.zeros((3, n_trajectories, n_points))  # 3 for x, y, z components
    for dl in prange(n_points):
        for t in prange(n_points):
            tp = t + (n_points + dl) - n_points * (t + n_points + dl >= 2 * n_points)
            x, y, z = funct(t, tp, *quantities)
            acc[0,:,dl] += x
            acc[1,:,dl] += y
            acc[2,:,dl] += z

    return acc / n_points

@njit(parallel=True)
def calc_source_with_numba_traj_filter(funct, dl, n_points, n_trajectories, *quantities):
    acc = np.zeros((n_trajectories))
    for t in prange(n_points):
        tp = t + (n_points + dl) - n_points * (t + n_points + dl >= 2 * n_points)
        acc += funct(t, tp, *quantities)

    return acc / n_points

@njit(parallel=True)
def calc_flux_with_numba_traj_filter(funct, dl, n_points, n_trajectories, *quantities):
    acc = np.zeros((3, n_trajectories))  # 3 for x, y, z components
    for t in prange(n_points):
        tp = t + (n_points + dl) - n_points * (t + n_points + dl >= 2 * n_points)
        x, y, z = funct(t, tp, *quantities)
        acc[0,:] += x
        acc[1,:] += y
        acc[2,:] += z
    return acc / n_points

@njit(parallel=True)
def calc_source_with_numba_traj_split(funct, n_points, n_trajectories, *quantities):
    acc = np.zeros((9, n_trajectories, n_points))
    for dl in prange(n_points):
        for t in prange(n_points):
            tp = t + (n_points + dl) - n_points * (t + n_points + dl >= 2 * n_points)
            xx, xy, xz, yx, yy, yz, zx, zy, zz = funct(t, tp, *quantities)
            acc[0,:,dl] += xx
            acc[1,:,dl] += xy
            acc[2,:,dl] += xz
            acc[3,:,dl] += yx
            acc[4,:,dl] += yy
            acc[5,:,dl] += yz
            acc[6,:,dl] += zx
            acc[7,:,dl] += zy
            acc[8,:,dl] += zz

    return acc / n_points
