from typing import List
from numba import njit, prange
import numpy as np


class AbstractTerm:
    def __init__(self):
        pass

    def calc(self, *args, **kwargs) -> (float or List[float]):
        raise NotImplementedError("You have to reimplement this method")
    
    def calc_fourier(self, *args, **kwargs) -> (float or List[float]):
        raise NotImplementedError("You have to reimplement this method")
    
    def calc_incremental_trajectories(self, args_array, num_trajectories, length_traj):
        """
        Calcule les incréments temporels/spatiaux pour une série de trajectoires.
        Utilise le multi-threading pour paralléliser sur num_trajectories.
        """
        import concurrent.futures

        def process_single_trajectory(i):
            """Fonction exécutée par chaque thread pour une trajectoire donnée"""
            args_traj = args_array[:, i, :]
            traj_results = []
            
            for j in range(length_traj):
                traj_results.append(self.calc([j], [length_traj], *args_traj, traj=True))
                
            return traj_results

        # Initialisation d'une liste vide de la bonne taille pour garder l'ordre
        result_list = [None] * num_trajectories

        # Création du pool de threads (utilise le nombre de coeurs de votre machine)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # On lance un thread par trajectoire
            futures = {executor.submit(process_single_trajectory, i): i for i in range(num_trajectories)}
            
            # Au fur et à mesure qu'ils terminent, on stocke le résultat à la bonne place
            for future in concurrent.futures.as_completed(futures):
                index = futures[future]
                result_list[index] = future.result()

        # Transformation en array NumPy
        result = np.array(result_list)

        # Rangement des axes si le résultat est tridimensionnel (vecteurs)
        if result.ndim == 3:
            result = np.moveaxis(result, -1, 0)

        return result

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