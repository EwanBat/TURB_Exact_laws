from typing import List
from numba import njit, prange
import numpy as np
import concurrent.futures


class AbstractTerm:
    def __init__(self):
        pass

    def calc(self, *args, **kwargs) -> (float or List[float]):
        raise NotImplementedError("You have to reimplement this method")
    
    def calc_fourier(self, *args, **kwargs) -> (float or List[float]):
        raise NotImplementedError("You have to reimplement this method")
    
    def _calc_incremental_trajectories_loop(self, merged_quantities, n_trajectories, n_points):
        
        def process_single_trajectory(i): # Process a single trajectory and return its results
            trajectory_results = [None] * n_points
            for dl in range(n_points):
                trajectory_results[dl] = self.calc([dl+n_points], [2*n_points], **merged_quantities[i], traj=True)
            return i, trajectory_results
        
        results = [None] * n_trajectories

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(process_single_trajectory, i): i for i in range(n_trajectories)}
            for future in concurrent.futures.as_completed(futures):
                index, trajectory_results = future.result()
                results[index] = trajectory_results
        
        results = np.array(results)
        if np.shape(results) != (n_trajectories, n_points):
            results = np.moveaxis(results, -1, 0)  # Move points axis to the end
        return 2 * results

    def calc_incremental_trajectories(self, dic_quantities: dict, traj_param: dict, sat1:str, sat2:str) -> (float or List[float]):
        n_trajectories = traj_param["n_trajectories"]
        n_points = traj_param["n_points"]

        merged_quantities = []
        for i in range(n_trajectories):
            merged_quantities.append({})
            for quantity in dic_quantities[sat1].keys():
                if quantity in dic_quantities[sat2].keys():   
                    merged_quantities[i].update({quantity: np.concatenate((dic_quantities[sat1][quantity][i,:], dic_quantities[sat2][quantity][i,:]), axis=0)})      
        
        return self._calc_incremental_trajectories_loop(merged_quantities, n_trajectories, n_points)        

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

@njit(parallel=True)
def calc_source_with_numba_traj(funct, dl, Nt, *quantities):
    acc = 0.0

    for t in prange(Nt):
        tp = t + dl - Nt * (t + dl >= Nt)
        acc += funct(t, tp, *quantities)

    return acc / Nt

@njit(parallel=True)
def calc_flux_with_numba_traj(funct, dl, Nt, *quantities):
    acc_x = 0.0
    acc_y = 0.0
    acc_z = 0.0

    for t in prange(Nt):
        tp = t + dl - Nt * (t + dl >= Nt)
        x, y, z = funct(t, tp, *quantities)
        acc_x += x
        acc_y += y
        acc_z += z

    return [acc_x / Nt, acc_y / Nt, acc_z / Nt]