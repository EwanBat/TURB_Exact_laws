import numpy as np
from typing import Callable, Tuple
import logging
from scipy.interpolate import Rbf


# ====================== Type of trajectories ======================
# ========== TRAJECTORY DEFINITIONS ==========

def trajectory_linear_x(t: np.ndarray, y_pos: int, z_pos: int, 
                        N: np.ndarray, Ninterp: int) -> np.ndarray:
    """
    Linear trajectory along the x axis (indices).
    
    Parameters:
    -----------
    t : np.ndarray
        Trajectory parameter (0 to N[0]-1)
    y_pos : int
        Fixed position on y (index)
    z_pos : int
        Fixed position on z (index)
    N : np.ndarray
        Grid dimensions (N[0], N[1], N[2])
    
    Returns:
    -------
    np.ndarray
        Trajectory points (n_points, 3) with [x, y, z] indices
    """
    x = t
    y = np.full_like(t, y_pos, dtype=int)
    z = np.full_like(t, z_pos, dtype=int)
    
    # Clip to grid limits
    x = np.clip(x, 0, N[0]-1).astype(int)
    y = np.clip(y, 0, N[1]-1).astype(int)
    z = np.clip(z, 0, N[2]-1).astype(int)
    return np.array([x, y, z]).T

def trajectory_linear_y(t: np.ndarray, x_pos: int, z_pos: int,
                        N: np.ndarray, Ninterp: int) -> np.ndarray:
    """
    Linear trajectory along the y axis (indices).
    
    Parameters:
    -----------
    t : np.ndarray
        Trajectory parameter (0 to N[1]-1)
    x_pos : int
        Fixed position on x (index)
    z_pos : int
        Fixed position on z (index)
    N : np.ndarray
        Grid dimensions (N[0], N[1], N[2])
    
    Returns:
    -------
    np.ndarray
        Trajectory points (n_points, 3) with [x, y, z] indices
    """
    x = np.full_like(t, x_pos, dtype=int)
    y = t
    z = np.full_like(t, z_pos, dtype=int)
    
    # Clip to grid limits
    x = np.clip(x, 0, N[0]-1).astype(int)
    y = np.clip(y, 0, N[1]-1).astype(int)
    z = np.clip(z, 0, N[2]-1).astype(int)
    return np.array([x, y, z]).T

def trajectory_linear_z(t: np.ndarray, x_pos: int, y_pos: int,
                        N: np.ndarray, Ninterp: int) -> np.ndarray:
    """
    Linear trajectory along the z axis (indices).

    Parameters:
    -----------
    t : np.ndarray
        Trajectory parameter (0 to N[2]-1)
    x_pos : int
        Fixed position on x (index)
    y_pos : int
        Fixed position on y (index)
    N : np.ndarray
        Grid dimensions (N[0], N[1], N[2])

    Returns:
    -------
    np.ndarray
        Trajectory points (n_points, 3) with [x, y, z] indices
    """
    x = np.full_like(t, x_pos, dtype=int)
    y = np.full_like(t, y_pos, dtype=int)
    z = t

    # Clip to grid limits
    x = np.clip(x, 0, N[0]-1).astype(int)
    y = np.clip(y, 0, N[1]-1).astype(int)
    z = np.clip(z, 0, N[2]-1).astype(int)
    return np.array([x, y, z]).T

def trajectory_circular_xy(t: np.ndarray, radius: int, center_y: int, center_z: int,
                           N: np.ndarray, Ninterp: int) -> np.ndarray:
    """
    Circular trajectory in the xy plane around a center (indices).
    
    Parameters:
    -----------
    t : np.ndarray
        Trajectory parameter (index 0 to N[0]-1 for complete trajectory)
    radius : int
        Radius of the circle (in grid indices)
    center_y : int
        Center in y (index)
    center_z : int
        Fixed position in z (index)
    N : np.ndarray
        Grid dimensions
    
    Returns:
    -------
    np.ndarray
        Trajectory points (n_points, 3) with [x, y, z] indices
    """
    # Normalized angle parameter (0 to 2π)
    theta = 2 * np.pi * t / N[0]
    
    y = center_y + radius * np.cos(theta)
    x = radius * np.sin(theta)
    z = np.full_like(t, center_z, dtype=int)
    
    # Clip aux limites
    x = np.clip(x, 0, N[0]-1).astype(int)
    y = np.clip(y, 0, N[1]-1).astype(int)
    z = np.clip(z, 0, N[2]-1).astype(int)
    
    return np.array([x, y, z]).T


def trajectory_helical(t: np.ndarray, pitch: int, radius: int, center_y: int, center_z: int,
                       N: np.ndarray, Ninterp: int) -> np.ndarray:
    """
    Helical trajectory spiraling around a center (indices).
    
    Parameters:
    -----------
    t : np.ndarray
        Trajectory parameter (index 0 to N[0]-1)
    pitch : int
        Helix pitch (x progression per complete turn)
    radius : int
        Helix radius (in indices)
    center_y : int
        Center in y (index)
    center_z : int
        Center in z (index)
    N : np.ndarray
        Grid dimensions
    
    Returns:
    -------
    np.ndarray
        Trajectory points (n_points, 3) with [x, y, z] indices
    """
    # Number of complete rotations needed to cover the grid
    num_rotations = round((N[0] - 1) / pitch)

    # Angle progresses through multiple rotations
    theta = 2 * np.pi * num_rotations * t / (N[0] - 1)

    # X progresses across entire grid (0 to N[0]-1)
    x = t  

    y = center_y + radius * np.cos(theta)
    z = center_z + radius * np.sin(theta)
    
    # Clip aux limites
    x = np.clip(x, 0, N[0]-1).astype(int)
    y = np.clip(y, 0, N[1]-1).astype(int)
    z = np.clip(z, 0, N[2]-1).astype(int)
    
    return np.array([x, y, z]).T


def trajectory_diagonal(t: np.ndarray, N: np.ndarray, Ninterp: int) -> np.ndarray:
    """
    Diagonal trajectory through the cube.
    
    Parameters:
    -----------
    t : np.ndarray
        Trajectory parameter (0 to N[0]-1)
    N : np.ndarray
        Grid dimensions
    Ninterp : int
        Interpolation factor for grid dimensions
    Returns:
    -------
    np.ndarray
        Trajectory points (n_points, 3) with [x, y, z] indices
    """
    # X, y, z progression proportional
    x = t
    y = (t * N[1] / N[0]).astype(int)
    z = (t * N[2] / N[0]).astype(int)
    
    # Clip to limits
    x = np.clip(x, 0, N[0]-1).astype(int)
    y = np.clip(y, 0, N[1]-1).astype(int)
    z = np.clip(z, 0, N[2]-1).astype(int)
    
    return np.array([x, y, z]).T

# ====================== Trajectory generation and data extraction ======================

def generate_all_trajectory_kwargs_linear_x(N: np.ndarray, step: int) -> list:
    """
    Generate all possible trajectory_kwargs combinations for linear_x trajectory.
    
    Creates a trajectory for each (y_pos, z_pos) combination in the grid, covering
    all possible positions perpendicular to the x-axis propagation.
    
    Parameters:
    -----------
    N : np.ndarray
        Grid dimensions (N[0], N[1], N[2])
    step : int
        Step size for trajectory generation
    
    Returns:
    -------
    list : List of dictionaries with all (y_pos, z_pos) combinations
    """
    trajectory_kwargs_list = []
    
    for y_pos in range(0, N[1], step):
        for z_pos in range(0, N[2], step):
            trajectory_kwargs_list.append({
                'y_pos': int(y_pos),
                'z_pos': int(z_pos)
            })
    
    return trajectory_kwargs_list

def generate_all_trajectory_kwargs_linear_y(N: np.ndarray, step: int) -> list:
    """
    Generate all possible trajectory_kwargs combinations for linear_y trajectory.
    
    Creates a trajectory for each (x_pos, z_pos) combination in the grid, covering
    all possible positions perpendicular to the y-axis propagation.
    
    Parameters:
    -----------
    N : np.ndarray
        Grid dimensions (N[0], N[1], N[2])
    step : int
        Step size for trajectory generation
    
    Returns:
    -------
    list : List of dictionaries with all (x_pos, z_pos) combinations
    """
    trajectory_kwargs_list = []
    
    for x_pos in range(0, N[0], step):
        for z_pos in range(0, N[2], step):
            trajectory_kwargs_list.append({
                'x_pos': int(x_pos),
                'z_pos': int(z_pos)
            })
    
    return trajectory_kwargs_list

def generate_all_trajectory_kwargs_linear_z(N: np.ndarray, step: int) -> list:
    """
    Generate all possible trajectory_kwargs combinations for linear_z trajectory.
    
    Creates a trajectory for each (x_pos, y_pos) combination in the grid, covering
    all possible positions perpendicular to the z-axis propagation.
    
    Parameters:
    -----------
    N : np.ndarray
        Grid dimensions (N[0], N[1], N[2])
    step : int
        Step size for trajectory generation
    
    Returns:
    -------
    list : List of dictionaries with all (x_pos, y_pos) combinations
    """
    trajectory_kwargs_list = []
    
    for x_pos in range(0, N[0], step):
        for y_pos in range(0, N[1], step):
            trajectory_kwargs_list.append({
                'x_pos': int(x_pos),
                'y_pos': int(y_pos)
            })
    
    return trajectory_kwargs_list

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
    Compute normalized tangent vectors along a trajectory.
    
    For each point, uses finite differences with neighbors to compute the direction
    of motion. Edge points use one-sided differences, interior points use centered differences.
    All vectors are normalized to unit length.
    
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
        Normalized tangent vectors, shape (n_points, 3)
    """
    n_points = len(t_array)
    tangents = np.zeros((n_points, 3))
    Ninterp = traj_param.get('Ninterp', 1)
    N = grid_param['N']
    
    lx = np.arange(N[0]) * grid_param['c'][0]
    ly = np.arange(N[1]) * grid_param['c'][1]
    lz = np.arange(N[2]) * grid_param['c'][2]
    
    all_traj = trajectory_func(t_array, N=N, Ninterp=Ninterp, **kwargs)  # Une seule fois
    tangents = np.zeros_like(all_traj, dtype=float)
    tangents[0,:] = all_traj[1,:] - all_traj[0,:]  # Forward difference at start
    tangents[-1,:] = all_traj[-1,:] - all_traj[-2,:]  # Backward difference at end
    tangents[1:-1,:] = (all_traj[2:,:] - all_traj[:-2,:]) / 2  # Centered difference

    # Normalize tangent vectors
    norms = np.linalg.norm(tangents, axis=1)
    norms[norms == 0] = 1  # Avoid division by zero
    tangents = tangents / norms[:,np.newaxis]

    return tangents


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
        Number of satellites (1 or 4) and gap between satellites (if nbsatellite=4)
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

    if nbsatellite == 1:
        # ===== SINGLE SATELLITE =====
        trajectory_data = {'sat_0': {}}
        for key in dic_datas.keys():
            if isinstance(dic_datas[key], np.ndarray) and dic_datas[key].ndim == 3:
                # Extract values along the trajectory
                trajectory_data['sat_0'][key] = dic_datas[key][trajectory[:, 0], trajectory[:, 1], trajectory[:, 2]]
            else:
                trajectory_data['sat_0'][key] = dic_datas[key]
    
    elif nbsatellite == 4:
        # ========== QUAD SATELLITE =========
        # Define 4 satellites in a centered square
        satellites = {
            'sat_0': np.array([0, 0, 0]),   # Center
            'sat_1': traj_param['dR1'] / grid_param['c'][0],  # x positive
            'sat_2': traj_param['dR2'] / grid_param['c'][1],  # y positive
            'sat_3': traj_param['dR3'] / grid_param['c'][2]   # z positive
        }
        # Generate 4 trajectories
        trajectories = {}
        for sat_name, offset in satellites.items():
            trajectories[sat_name] = trajectory + offset[np.newaxis, :]  # Apply offset to trajectory
            # Using periodic boundary conditions to wrap around the grid
            trajectories[sat_name][:, 0] = trajectories[sat_name][:, 0] % N[0]
            trajectories[sat_name][:, 1] = trajectories[sat_name][:, 1] % N[1]
            trajectories[sat_name][:, 2] = trajectories[sat_name][:, 2] % N[2]
            trajectories[sat_name] = trajectories[sat_name].astype(int)  # Ensure indices are integers
        # Extract data for each satellite - reorganized structure
        trajectory_data = {sat_name: {} for sat_name in satellites.keys()}
        
        for key in dic_datas.keys():
            if isinstance(dic_datas[key], np.ndarray) and dic_datas[key].ndim == 3:
                for sat_name, traj in trajectories.items():
                    # Extract values along this satellite trajectory
                    trajectory_data[sat_name][key] = dic_datas[key][traj[:,0], traj[:,1], traj[:,2]]
            else:
                # Keep scalars for all satellites
                for sat_name in satellites.keys():
                    trajectory_data[sat_name][key] = dic_datas[key]
    else:
        raise ValueError(f"nbsatellite must be 1 or 4, got {nbsatellite}")
    
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
        - gap_satellite: separation between satellites (if nbsatellite=4)
        - nbsatellite: number of satellites (1 or 4)
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

    if verbose:
        logging.info(f"\n  Processing {n_trajectories} trajectory/trajectories...")
    
    # Generate all trajectories and extract data
    trajectories_list = []
    tangents_list = []
    ltraj_list = []
    trajectory_data_list = []
    
    # Generate trajectory with interpolation
    if traj_param['trajectory_method'] == 'linear_x':
        t = np.arange(Ninterp * N[0]) / Ninterp
    elif traj_param['trajectory_method'] == 'linear_y':
        t = np.arange(Ninterp * N[1]) / Ninterp
    elif traj_param['trajectory_method'] == 'linear_z':
        t = np.arange(Ninterp * N[2]) / Ninterp
    
    if nbsatellite == 4: # Define satellite offsets for 4-satellite configuration
        dR1 = np.array([gap_satellite, 0, 0]) * grid_param['c'][0]  # Convert gap from indices to physical units
        dR2 = np.array([0, gap_satellite, 0]) * grid_param['c'][1]
        dR3 = np.array([0, 0, gap_satellite]) * grid_param['c'][2]
        traj_param['dR1'] = dR1
        traj_param['dR2'] = dR2
        traj_param['dR3'] = dR3

    for idx, trajectory_kwargs in enumerate(trajectory_kwargs_list):
        trajectory = trajectory_func(t, N=N, Ninterp=Ninterp, **trajectory_kwargs)
        trajectories_list.append(trajectory)
           
        # Compute normalized tangent vectors and physical coordinates
        tangents = _compute_tangent_vectors(trajectory_func, t, grid_param, traj_param, **trajectory_kwargs)
        tangents_list.append(tangents)
        
        ltraj = _compute_trajectory_coordinates(trajectory, grid_param, tangents)
        ltraj_list.append(ltraj)

        # Extract quantities along trajectory
        trajectory_data = extract_quantities_along_trajectory(
            dic_datas_3d,
            trajectory,
            traj_param,
            grid_param,
        )
        trajectory_data_list.append(trajectory_data)

    # Get dimensions from first trajectory
    first_data = trajectory_data_list[0]
    first_satellite = list(first_data.keys())[0]  # Get first satellite name (e.g., 'sat_0')
    first_var = list(first_data[first_satellite].keys())[0]  # Get first variable
    n_points = len(trajectory_data_list[0][first_satellite][first_var])
    
    # Initialize output structure: {sat_name: {var_name: array(n_trajectories, n_points)}}
    dic_datas_combined = {}
    
    # Get all satellite names
    satellite_names = list(first_data.keys())
    
    # Get all variable names (same for all satellites)
    var_names = list(first_data[first_satellite].keys())
    
    # Initialize arrays for each satellite and variable
    for sat_name in satellite_names:
        dic_datas_combined[sat_name] = {}
        for var_name in var_names:
            dtype = trajectory_data_list[0][sat_name][var_name].dtype
            dic_datas_combined[sat_name][var_name] = np.zeros((n_trajectories, n_points), dtype=dtype)
    
    # Fill arrays with data from all trajectories
    for traj_idx, trajectory_data in enumerate(trajectory_data_list):
        for sat_name in satellite_names:
            for var_name in var_names:
                dic_datas_combined[sat_name][var_name][traj_idx] = trajectory_data[sat_name][var_name]
    
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
        logging.info(f"    Satellites: {satellite_names}")
    
    return dic_datas_combined, trajectories_list
