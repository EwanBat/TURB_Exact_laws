"""
Geometry and sampling helpers for satellite trajectories.

Contains satellite-offset construction, tangent/arc-length computation,
interpolation and extraction of 3D fields along trajectories, and the
combine_multiple_trajectories workflow driver.
"""
import numpy as np
from typing import Callable, Tuple
import logging
from scipy.interpolate import Rbf


# ====================== Satellite formation geometry ======================

def _get_satellite_offsets(nbsatellite: int,
                           gap_satellite: float,
                           grid_param: dict,
                           formation: str) -> dict:
    """Build the satellite offsets used to sample the mesh around the trajectory."""
    c = np.asarray(grid_param['c'])
    satellite_offsets = {'sat_0': np.zeros(3, dtype=float)}

    if nbsatellite == 1:
        return satellite_offsets

    if nbsatellite == 4:
        satellite_offsets.update({
            'sat_1': np.array([gap_satellite, 0, 0], dtype=float) * c,
            'sat_2': np.array([0, gap_satellite, 0], dtype=float) * c,
            'sat_3': np.array([0, 0, gap_satellite], dtype=float) * c,
        })
        return satellite_offsets

    if nbsatellite == 9 and formation == 'star':
        satellite_offsets.update({
            'sat_1': np.array([gap_satellite, 0, 0], dtype=float) * c,
            'sat_2': np.array([-gap_satellite, 0, 0], dtype=float) * c,
            'sat_3': np.array([0, gap_satellite, 0], dtype=float) * c,
            'sat_4': np.array([0, -gap_satellite, 0], dtype=float) * c,
            'sat_5': np.array([0, 0, gap_satellite], dtype=float) * c,
            'sat_6': np.array([0, 0, -gap_satellite], dtype=float) * c,
            'sat_7': np.array([gap_satellite, gap_satellite, gap_satellite], dtype=float) * c,
            'sat_8': np.array([-gap_satellite, -gap_satellite, -gap_satellite], dtype=float) * c,
        })
        return satellite_offsets

    if nbsatellite == 9 and formation == 'cross':
        satellite_offsets.update({
                    'sat_1': np.array([-2*gap_satellite, 0, 0], dtype=float) * c,
                    'sat_2': np.array([-gap_satellite, 0, 0], dtype=float) * c,
                    'sat_3': np.array([0, gap_satellite, 0], dtype=float) * c,
                    'sat_4': np.array([0, -2*gap_satellite, 0], dtype=float) * c,
                    'sat_5': np.array([0, -gap_satellite, 0], dtype=float) * c,
                    'sat_6': np.array([0, gap_satellite, 0], dtype=float) * c,
                    'sat_7': np.array([0, 0, -gap_satellite], dtype=float) * c,
                    'sat_8': np.array([0, 0, gap_satellite], dtype=float) * c,
                })
        return satellite_offsets
    raise ValueError(f"nbsatellite must be 1, 4 or 9, got {nbsatellite}")


# ====================== Trajectory geometry helpers ======================

def _compute_trajectory_coordinates(trajectory: np.ndarray,
                                    grid_param: dict,
                                    tangents: np.ndarray) -> np.ndarray:
    """
    Compute physical arc length coordinates along trajectory.
    
    Uses tangent vectors to project coordinates onto the tangent direction,
    giving a 1D coordinate along the trajectory path.
    
    Parameters:
    -----------
    trajectory : np.ndarray
        Trajectory indices, shape (n_points, 3)
    grid_param : dict
        Grid parameters with L (domain size) and c (cell spacing) arrays
    tangents : np.ndarray
        Unit tangent vectors, shape (n_points, 3)
    
    Returns:
    -------
    np.ndarray
        Physical arc length coordinates, shape (n_points,)
    """
    lx = np.arange(grid_param['N'][0]) * grid_param['c'][0] - grid_param['L'][0] / 2
    ly = np.arange(grid_param['N'][1]) * grid_param['c'][1] - grid_param['L'][1] / 2
    lz = np.arange(grid_param['N'][2]) * grid_param['c'][2] - grid_param['L'][2] / 2

    ltraj = tangents[:, 0] * lx[trajectory[:, 0]] + tangents[:, 1] * ly[trajectory[:, 1]] + tangents[:, 2] * lz[trajectory[:, 2]]
    
    return ltraj

def interpolation_along_trajectory(trajectory: np.ndarray, array_data: np.ndarray, grid_param: dict) -> dict:
    """
    Interpolate data along the trajectory.

    Parameters:
    -----------
    trajectory : np.ndarray
        Trajectory in indices (n_points, 3)
    array_data : np.ndarray
        Data to interpolate (3D array)
    grid_param : dict
        Dictionary containing simulation parameters

    Returns:
    -------
    np.ndarray
        Interpolated data along the trajectory (n_points,)
    """
    x, y, z = np.arange(grid_param['N'][0]), np.arange(grid_param['N'][1]), np.arange(grid_param['N'][2])
    rbf_interpolator = Rbf(x, y, z, array_data, function='thin_plate')

    # Interpolate along the trajectory
    interpolated_values = rbf_interpolator(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2])
    return interpolated_values

def _compute_tangent_vectors(trajectory_func: Callable,
                             t_array: np.ndarray,
                             grid_param: dict,
                             traj_param: dict,
                             **kwargs) -> np.ndarray:
    """
    Compute normalized tangent vectors along a trajectory IN PHYSICAL SPACE.
    
    Parameters:
    -----------
    trajectory_func : Callable
        Function generating trajectory coordinates
    t_array : np.ndarray
        Trajectory parameter values
    grid_param : dict
        Grid parameters (N, L, c) for physical coordinate conversion
    traj_param : dict
        Trajectory-specific parameters (N, Ninterp)
    **kwargs : dict
        Additional parameters passed to trajectory_func
    
    Returns:
    -------
    np.ndarray
        Normalized tangent vectors in PHYSICAL SPACE, shape (n_points, 3)
    """
    
    # Physical coordinates arrays
    lx = np.arange(grid_param['N'][0]) * grid_param['c'][0]
    ly = np.arange(grid_param['N'][1]) * grid_param['c'][1]
    lz = np.arange(grid_param['N'][2]) * grid_param['c'][2]
    
    # Step 1: Get trajectory in INDEX space
    all_traj_indices = trajectory_func(t_array, N=grid_param['N'], Ninterp=traj_param.get('Ninterp', 1), **kwargs)  # Shape (n_points, 3)
    
    # Step 2: Convert INDICES → PHYSICAL COORDINATES
    all_traj_physical = np.zeros_like(all_traj_indices, dtype=float)
    all_traj_physical[:, 0] = lx[all_traj_indices[:, 0].astype(int)]
    all_traj_physical[:, 1] = ly[all_traj_indices[:, 1].astype(int)]
    all_traj_physical[:, 2] = lz[all_traj_indices[:, 2].astype(int)]
    
    # Step 3: Compute finite differences in PHYSICAL SPACE
    tangents = np.zeros_like(all_traj_physical, dtype=float)
    tangents[0,:] = all_traj_physical[1,:] - all_traj_physical[0,:]  # [0.001m, 0.005m, 0]
    tangents[-1,:] = all_traj_physical[-1,:] - all_traj_physical[-2,:]
    tangents[1:-1,:] = (all_traj_physical[2:,:] - all_traj_physical[:-2,:]) / 2

    # Step 4: Normalize in PHYSICAL SPACE
    norms = np.linalg.norm(tangents, axis=1)  # √(0.001² + 0.005²) = 0.0051m
    norms[norms == 0] = 1  # Avoid division by zero
    tangents = tangents / norms[:,np.newaxis]  # Vecteur unitaire sans dimension

    return tangents


# ====================== Extraction and combination ======================

def extract_quantities_along_trajectory(dic_datas: dict, trajectory: np.ndarray, 
                                       traj_param: dict,
                                       grid_param: dict) -> dict:
    """
    Extract quantities along one or multiple trajectories (indices).
    
    Parameters:
    -----------
    dic_datas : dict
        Dictionary of 3D quantities
    trajectory : np.ndarray
        Central trajectory (n_points, 3) in indices
    traj_param : dict
        Number of satellites (1, 4 or 9) and gap between satellites (if needed).
        For 9 satellites the offsets are: sat_0 center, sat_1/2 along +-x,
        sat_3/4 along +-y, sat_5/6 along +-z, sat_7/8 at (+++) and (---).
    grid_param : dict
        Grid parameters (N, L, c) for physical coordinate conversion
    Returns:
    -------
    dict : Data organized as {sat_name: {quantity_name: array(n_points,)}}
           Structure is uniform regardless of nbsatellite value
    """
    n_points = len(trajectory)
    N = grid_param['N']
    nbsatellite = traj_param['nbsatellite']

    satellite_offsets = traj_param.get('satellite_offsets')
    if satellite_offsets is None:
        satellite_offsets = _get_satellite_offsets(
            nbsatellite,
            traj_param.get('gap_satellite', 1),
            grid_param,
            traj_param.get('formation', None)
        )

    trajectory_data = {sat_name: {} for sat_name in satellite_offsets.keys()}
    trajectories = {}

    for sat_name, offset in satellite_offsets.items():
        offset_index = np.rint(offset / grid_param['c']).astype(int)
        trajectories[sat_name] = trajectory + offset_index[np.newaxis, :]
        trajectories[sat_name][:, 0] = trajectories[sat_name][:, 0] % N[0]
        trajectories[sat_name][:, 1] = trajectories[sat_name][:, 1] % N[1]
        trajectories[sat_name][:, 2] = trajectories[sat_name][:, 2] % N[2]
        trajectories[sat_name] = trajectories[sat_name].astype(int)

    for key in dic_datas.keys():
        if isinstance(dic_datas[key], np.ndarray) and dic_datas[key].ndim == 3:
            for sat_name, traj in trajectories.items():
                trajectory_data[sat_name][key] = dic_datas[key][traj[:, 0], traj[:, 1], traj[:, 2]]
        else:
            for sat_name in satellite_offsets.keys():
                trajectory_data[sat_name][key] = dic_datas[key]
    
    return trajectory_data

def combine_multiple_trajectories(trajectory_func: Callable,
                                  dic_datas_3d: dict,
                                  traj_param: dict,
                                  grid_param: dict,
                                  verbose: bool = True) -> Tuple[dict, dict]:
    """
    Generate multiple trajectories and extract data along them.
    
    Combines trajectories with different parameters into multi-dimensional arrays.
    Data is organized as: {sat_name: {variable_name: array(n_trajectories, n_points)}}
    This structure is uniform regardless of nbsatellite value.
    
    Parameters:
    -----------
    trajectory_func : Callable
        Function that generates trajectory coordinates (e.g., trajectory_linear_x)
    dic_datas_3d : dict
        3D field data from OCA files
    traj_param : dict
        Trajectory parameters containing:
        - trajectory_kwargs_list: list of parameter dicts for each trajectory
        - Ninterp: interpolation factor
        - gap_satellite: separation between satellites (if nbsatellite > 1)
        - nbsatellite: number of satellites (1, 4 or 9). For 9, sat_0 is the
          center, sat_1/2 along +-x, sat_3/4 along +-y, sat_5/6 along +-z,
          sat_7/8 at (+++) and (---)
    grid_param : dict
        Grid parameters (N, L, c) defining the computational domain
    verbose : bool
        If True, log processing summary
    
    Returns:
    -------
    tuple : (dic_datas_combined, trajectories_list)
        - dic_datas_combined: {sat_name: {var_name: array(n_trajectories, n_points)}}
        - trajectories_list: list of generated trajectory arrays
    """
    
    N = grid_param['N']
    trajectory_kwargs_list = traj_param.get('trajectory_kwargs_list', [{}])
    n_trajectories = traj_param.get('n_trajectories', len(trajectory_kwargs_list))
    Ninterp = traj_param.get('Ninterp', 1)
    gap_satellite = traj_param.get('gap_satellite', 1)
    nbsatellite = traj_param.get('nbsatellite', 1)
    formation = traj_param.get('formation', None)

    if verbose:
        logging.info(f"  Processing {n_trajectories} trajectory/trajectories...")
    
    # Generate all trajectories and extract data
    trajectories_list = []
    tangents_list = []
    ltraj_list = []
    
    # Generate trajectory with interpolation
    if traj_param['trajectory_method'] == 'linear_x' or traj_param['trajectory_method'] == 'linear_minus_x':
        t = np.arange(Ninterp * N[0]) / Ninterp
    elif traj_param['trajectory_method'] == 'linear_y' or traj_param['trajectory_method'] == 'linear_minus_y':
        t = np.arange(Ninterp * N[1]) / Ninterp
    elif traj_param['trajectory_method'] == 'linear_z' or traj_param['trajectory_method'] == 'linear_minus_z':
        t = np.arange(Ninterp * N[2]) / Ninterp
    elif traj_param['trajectory_method'] == 'linear_xy':
        t = np.arange(Ninterp * min(N[0], N[1])) / Ninterp
    
    # Get dimensions from first trajectory
    n_points = len(t)

    satellite_offsets = _get_satellite_offsets(nbsatellite, gap_satellite, grid_param, formation)
    traj_param['satellite_offsets'] = satellite_offsets

    if nbsatellite == 4:
        traj_param['dR1'] = satellite_offsets['sat_1']
        traj_param['dR2'] = satellite_offsets['sat_2']
        traj_param['dR3'] = satellite_offsets['sat_3']

    # Initialize output structure: {sat_name: {var_name: array(n_trajectories, n_points)}}
    dic_datas_combined = {
        sat_name: {var: np.zeros((n_trajectories, n_points)) for var in dic_datas_3d.keys()}
        for sat_name in satellite_offsets.keys()
    }

    for idx, trajectory_kwargs in enumerate(trajectory_kwargs_list):
        trajectories_list.append(trajectory_func(t, N=N, Ninterp=Ninterp, **trajectory_kwargs))
           
        # Compute normalized tangent vectors and physical coordinates
        tangents_list.append(_compute_tangent_vectors(trajectory_func, t, grid_param, traj_param, **trajectory_kwargs))

        ltraj_list.append(_compute_trajectory_coordinates(trajectories_list[-1], grid_param, tangents_list[-1]))

        # Extract quantities along trajectory
        trajectory_data = extract_quantities_along_trajectory(
            dic_datas_3d,
            trajectories_list[-1],
            traj_param,
            grid_param,
        )
        for sat_name in trajectory_data.keys():
            for var_name in trajectory_data[sat_name].keys():
                if var_name not in dic_datas_combined[sat_name]:
                    dic_datas_combined[sat_name][var_name] = []
                dic_datas_combined[sat_name][var_name][idx, :] = trajectory_data[sat_name][var_name]
        del trajectory_data 
    
    # Store trajectory metadata and geometry
    traj_param['trajectories_list'] = np.stack(trajectories_list)
    traj_param['tangents_list'] = np.stack(tangents_list)
    traj_param['ltraj_list'] = np.stack(ltraj_list)
    traj_param['n_trajectories'] = n_trajectories
    traj_param['n_points'] = n_points
    
    if verbose:
        logging.info(f"  [OK] Processed {n_trajectories} trajectory/trajectories successfully")
        logging.info(f"    Data structure: {{sat_name: {{var_name: array(n_trajectories, n_points)}}}}")
        logging.info(f"    Data shape: ({n_trajectories}, {n_points})")
        logging.info(f"    Satellites: {', '.join(dic_datas_combined.keys())}")
    
    del trajectories_list, tangents_list, ltraj_list  # Free memory

    return dic_datas_combined