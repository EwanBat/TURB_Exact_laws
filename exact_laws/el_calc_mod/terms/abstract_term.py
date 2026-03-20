from typing import List
from numba import njit, prange


class AbstractTerm:
    def __init__(self):
        pass

    def calc(self, *args, **kwargs) -> (float or List[float]):
        raise NotImplementedError("You have to reimplement this method")
    
    def calc_fourier(self, *args, **kwargs) -> (float or List[float]):
        raise NotImplementedError("You have to reimplement this method")

    def variables(self) -> List[str]:
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
