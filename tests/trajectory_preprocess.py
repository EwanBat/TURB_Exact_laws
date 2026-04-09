"""
Preprocessing module for satellite trajectories.
Encapsulates loading OCA data and retrieving required quantities.
Support for custom trajectories - simple indexing in the cube.
"""

import logging
import numpy as np
from scipy.interpolate import Rbf
import h5py
import configparser
from pathlib import Path
from datetime import datetime
from typing import Callable, Tuple
import json

from exact_laws.preprocessing.process_on_oca_files import (
    extract_quantities_from_OCA_file,
    extract_simu_param_from_OCA_file
)

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


# ========== UTILITY FUNCTIONS ==========

def setup_logging(config_name="trajectory_preprocess"):
    """Create a logger with timestamp."""
    log_filename = f"{config_name}_{datetime.now().strftime('%d%m%Y_%H%M%S')}.log"
    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-7s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging


def param_to_txt(grid_param: dict, traj_param: dict, physical_param: dict, 
                 output_file: str = "parameters_summary.txt"):
    """
    Save grid, trajectory, and physical parameters to a JSON file (with .txt extension).
    Handles non-serializable objects (numpy arrays, functions) appropriately.
    
    Parameters:
    -----------
    grid_param : dict
        Grid parameters (N, L, c)
    traj_param : dict
        Trajectory parameters
    physical_param : dict
        Physical parameters
    output_file : str
        Output filename (default: "parameters_summary.txt")
    """
    
    def convert_to_serializable(obj):
        """Convert non-JSON-serializable objects to serializable format."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        elif callable(obj):
            return str(obj.__name__)
        else:
            return obj
    
    # Convert parameters to serializable format
    grid_param_serializable = convert_to_serializable(grid_param)
    traj_param_serializable = convert_to_serializable(traj_param)
    physical_param_serializable = convert_to_serializable(physical_param)
    
    # Create output dictionary
    output_data = {
        'grid_param': grid_param_serializable,
        'traj_param': traj_param_serializable,
        'physical_param': physical_param_serializable
    }
    
    # Save to JSON file
    try:
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=4)
        logging.info(f"Parameters saved to {output_file}")
    except Exception as e:
        logging.error(f"Error saving parameters to {output_file}: {e}")
        raise

def load_oca_data(input_folder, cycle, sim_type):
    """
    Loads all required OCA data.
    
    Parameters:
    -----------
    input_folder : str
        Path to folder containing 3Dfields_*.h5 files
    cycle : str
        Name of the cycle (ex: "cycle_0")
    sim_type : str
        Simulation type (ex: "CGL5") to identify parameters
    
    Returns:
    -------
    tuple : (dic_datas, dic_param) dictionaries of quantities and parameters
    """
    logging.info("\n" + "="*70)
    logging.info("LOADING OCA DATA")
    
    dic_datas = {}
    grid_param = {}
    
    # Load velocity
    with h5py.File(f"{input_folder}/3Dfields_v.h5", "r") as fv:
        param_key = "3Dgrid" if sim_type.endswith(("CGL3", "CGL5")) else "Simulation_Parameters"
        grid_param = extract_simu_param_from_OCA_file(fv, grid_param, param_key)
        (dic_datas["vx"],
         dic_datas["vy"],
         dic_datas["vz"]) = extract_quantities_from_OCA_file(fv, ["vx", "vy", "vz"], cycle)
    logging.info(f"  [OK] Velocity loaded:         {dic_datas['vx'].shape}")
    
    # Load density
    with h5py.File(f"{input_folder}/3Dfields_rho.h5", "r") as frho:
        dic_datas["rho"] = extract_quantities_from_OCA_file(frho, ["rho"], cycle)[0]
    logging.info(f"  [OK] Density loaded:          {dic_datas['rho'].shape}")
    
    # Load magnetic field
    with h5py.File(f"{input_folder}/3Dfields_b.h5", "r") as fb:
        (dic_datas["bx"],
         dic_datas["by"],
         dic_datas["bz"]) = extract_quantities_from_OCA_file(fb, ["bx", "by", "bz"], cycle)
    logging.info(f"  [OK] Magnetic field loaded:   {dic_datas['bx'].shape}")
    
    # Load pressure components
    with h5py.File(f"{input_folder}/3Dfields_pi.h5", "r") as fp:
        (dic_datas["ppar"],
         dic_datas["pperp"]) = extract_quantities_from_OCA_file(fp, ["pparli", "pperpi"], cycle)
        dic_datas["ppar"] /= 2
        dic_datas["pperp"] /= 2
    logging.info(f"  [OK] Pressure loaded:         {dic_datas['ppar'].shape}")
    
    # Try loading force amplitude (optional)
    try:
        with h5py.File(f"{input_folder}/3Dfields_forcl_ampl.h5", "r") as ff:
            (dic_datas["fp"],
             dic_datas["fm"]) = extract_quantities_from_OCA_file(ff, ["forcl_ampl_plus", "forcl_ampl_mins"], cycle)
        logging.info(f"  [OK] Force amplitude loaded:  {dic_datas['fp'].shape}")
    except:
        logging.warning("  [SKIP] Force amplitude not loaded")
    
    # Print summary
    logging.info("\n" + "-"*70)
    logging.info("DATA LOADING SUMMARY")
    logging.info(f"  Grid dimensions (N):  {grid_param['N']}")
    logging.info(f"  Domain size (L):      {grid_param['L']}")
    logging.info(f"  Cell spacing (c):     {grid_param['c']}")
    logging.info(f"  Data fields:          {len(dic_datas)} fields loaded")
    for field in sorted(dic_datas.keys()):
        logging.info(f"    - {field}")
    
    return dic_datas, grid_param


def load_config_from_ini(config_file, input_folder: str = ""):
    """
    Load all parameters from a .ini file
    
    Parameters:
    -----------
    config_file : str
        Path to the configuration .ini file
    input_folder : str
        Default path to folder containing OCA data (can be overridden in INI)
    
    Returns:
    -------
    tuple : (laws, terms, quantities, physical_params, trajectory_kwargs_list,
             nbsatellite, gap_satellite, input_folder, cycle, sim_type, di,
             trajectory_method, Ninterp)
    """
    config = configparser.ConfigParser()
    config.read(config_file)
    
    # Check required sections
    required_sections = ['RUN_PARAMS', 'INPUT_DATA']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Required section missing in INI file: [{section}]")
    
    # Load OUTPUT_DATA section
    laws = eval(config["OUTPUT_DATA"].get("laws", "[]"))
    terms = eval(config["OUTPUT_DATA"].get("terms", "[]"))
    quantities = eval(config["OUTPUT_DATA"].get("quantities", "[]"))
    name_output = config["OUTPUT_DATA"].get("name_output", "trajectory_output")
    
    # Load PHYSICAL_PARAMS section
    physical_param = {}
    if "PHYSICAL_PARAMS" in config:
        for key in config["PHYSICAL_PARAMS"].keys():
            try:
                physical_param[key] = float(eval(config["PHYSICAL_PARAMS"][key]))
            except:
                physical_param[key] = config["PHYSICAL_PARAMS"][key]
    
    # Load trajectory_kwargs as list of dictionaries
    trajectory_kwargs_str = config["RUN_PARAMS"].get("trajectory_kwargs", "[{}]")
    trajectory_kwargs_list = eval(trajectory_kwargs_str)
    
    # Ensure it's a list of dicts
    if isinstance(trajectory_kwargs_list, dict):
        trajectory_kwargs_list = [trajectory_kwargs_list]
    elif not isinstance(trajectory_kwargs_list, list):
        trajectory_kwargs_list = [{}]
    
    # Load RUN_PARAMS section
    try:
        nbsatellite = config["RUN_PARAMS"].getint("nbsatellite", 1)
        gap_satellite = config["RUN_PARAMS"].getfloat("gap_satellite", 1)
        trajectory_method = config["RUN_PARAMS"].get("trajectory_method", "linear_x")
        Ninterp = config["RUN_PARAMS"].getint("Ninterp", 1)
    except Exception as e:
        logging.error(f"Error reading RUN_PARAMS: {e}")
        raise
    
    # Load INPUT_DATA section
    try:
        input_folder = config["INPUT_DATA"].get("path", input_folder)
        cycle = config["INPUT_DATA"].get("cycle", "cycle_0")
        sim_type = config["INPUT_DATA"].get("sim_type", "OCA_CGL5").split("_")[-1]
    except Exception as e:
        logging.error(f"Error reading INPUT_DATA: {e}")
        raise
    
    # Load PHYSICAL_PARAMS di value
    try:
        di = config["PHYSICAL_PARAMS"].getfloat("di", 1.0)
    except Exception as e:
        logging.error(f"Error reading di from PHYSICAL_PARAMS: {e}")
        di = 1.0
    
    return (laws, terms, quantities, name_output, physical_param, trajectory_kwargs_list,
            nbsatellite, gap_satellite, input_folder, cycle, sim_type, di,
            trajectory_method, Ninterp)


def extract_quantities_along_trajectory(dic_datas: dict, trajectory: np.ndarray, 
                                       nbsatellite: int = 1, gap_satellite: int = 2) -> dict:
    """
    Extract quantities along one or multiple trajectories (indices).
    
    Parameters:
    -----------
    dic_datas : dict
        Dictionary of 3D quantities
    trajectory : np.ndarray
        Central trajectory (n_points, 3) in indices
    nbsatellite : int
        Number of satellites (1 or 4)
    gap_satellite : int
        Gap between satellites in indices (used if nbsatellite=4)
    
    Returns:
    -------
    dict : 1D quantities extracted along trajectory/trajectories
           If nbsatellite=1: {quantity_name: (n_points,)}
           If nbsatellite=4: {quantity_name: {sat_0: (n_points,), sat_1: ..., sat_2: ..., sat_3: ...}}
    """
    n_points = len(trajectory)
    
    if nbsatellite == 1:
        # ===== SINGLE SATELLITE =====
        trajectory_data = {}
        
        for key in dic_datas.keys():
            if isinstance(dic_datas[key], np.ndarray) and dic_datas[key].ndim == 3:
                # Extract values along the trajectory
                trajectory_data[key] = dic_datas[key][trajectory[:, 0], trajectory[:, 1], trajectory[:, 2]]
            else:
                trajectory_data[key] = dic_datas[key]
    
    elif nbsatellite == 4:
        # ========== QUAD SATELLITE =========
        # Define 4 satellites in a centered square
        satellites = {
            'sat_0': np.array([0, 0, 0]),   # Center
            'sat_1': np.array([gap_satellite, 0, 0]),  # x positive
            'sat_2': np.array([0, gap_satellite, 0]),  # y positive
            'sat_3': np.array([0, 0, gap_satellite])   # z positive
        }
        
        # Generate 4 trajectories
        trajectories = {}
        for sat_name, offset in satellites.items():
            traj = trajectory + offset
            trajectories[sat_name] = np.clip(traj, 0, 255).astype(int)
        
        # Extract data for each satellite
        trajectory_data = {}
        
        for key in dic_datas.keys():
            if isinstance(dic_datas[key], np.ndarray) and dic_datas[key].ndim == 3:
                trajectory_data[key] = {}
                
                for sat_name, traj in trajectories.items():
                    # Extract values along this satellite trajectory
                    trajectory_data[key][sat_name] = np.array([
                        dic_datas[key][int(traj[i, 0]), int(traj[i, 1]), int(traj[i, 2])]
                        for i in range(n_points)
                    ])
            else:
                # Keep scalars as before
                trajectory_data[key] = dic_datas[key]
    
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
    For single satellites, data is arranged as (n_trajectories, n_points).
    For 4-satellite formations, data is arranged as (n_satellites, n_trajectories, n_points).
    
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
        - dic_datas_combined: extracted quantities arranged as described above
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
    
    for idx, trajectory_kwargs in enumerate(trajectory_kwargs_list):
        # Generate trajectory with interpolation
        t = np.arange(Ninterp * N[0]) / Ninterp
        trajectory = trajectory_func(t, N=N, Ninterp=Ninterp, **trajectory_kwargs)
        trajectories_list.append(trajectory)
        n_points = len(trajectory)
        
        # Compute normalized tangent vectors and physical coordinates
        tangents = _compute_tangent_vectors(trajectory_func, t, grid_param, traj_param, **trajectory_kwargs)
        tangents_list.append(tangents)
        
        ltraj = _compute_trajectory_coordinates(trajectory, grid_param, tangents)
        ltraj_list.append(ltraj)
        
        # Extract quantities along trajectory
        trajectory_data = extract_quantities_along_trajectory(
            dic_datas_3d,
            trajectory,
            nbsatellite=nbsatellite,
            gap_satellite=gap_satellite
        )
        trajectory_data_list.append(trajectory_data)
    
    # Get dimensions from first trajectory
    first_data = trajectory_data_list[0]
    n_points = len(trajectory_data_list[0][list(first_data.keys())[0]] 
                  if nbsatellite == 1 
                  else list(first_data[list(first_data.keys())[0]].values())[0])
    
    # Allocate output arrays with proper shape
    dic_datas_combined = {}
    
    if nbsatellite == 1:
        # Shape: (n_trajectories, n_points)
        for key in first_data.keys():
            dtype = trajectory_data_list[0][key].dtype
            dic_datas_combined[key] = np.zeros((n_trajectories, n_points), dtype=dtype)
        
        dic_datas_combined = {
            key: np.stack([traj[key] for traj in trajectory_data_list], axis=0)
            for key in trajectory_data_list[0].keys()
            }
    
    else:
        # Shape: (n_satellites, n_trajectories, n_points)
        for key in first_data.keys():
            dtype = trajectory_data_list[0][key]['sat_0'].dtype
            dic_datas_combined[key] = np.zeros((nbsatellite, n_trajectories, n_points), dtype=dtype)
        
        # Fill arrays with all satellite data
        for traj_idx, trajectory_data in enumerate(trajectory_data_list):
            for key in trajectory_data.keys():
                if isinstance(trajectory_data[key], dict):
                    for sat_num, sat_name in enumerate([f'sat_{i}' for i in range(nbsatellite)]):
                        dic_datas_combined[key][sat_num, traj_idx] = trajectory_data[key][sat_name]
                else:
                    # Propagate scalar values to all satellites
                    for sat_num in range(nbsatellite):
                        dic_datas_combined[key][sat_num, traj_idx] = trajectory_data[key]
    
    # Store trajectory metadata and geometry
    traj_param['trajectories_list'] = trajectories_list
    traj_param['tangents_list'] = tangents_list
    traj_param['ltraj_list'] = ltraj_list
    traj_param['n_trajectories'] = n_trajectories
    traj_param['n_points'] = n_points
    
    if verbose:
        logging.info(f"  ✓ Processed {n_trajectories} trajectory/trajectories successfully")
        if nbsatellite == 1:
            logging.info(f"    Data shape: (n_trajectories, n_points) = ({n_trajectories}, {n_points})")
        else:
            logging.info(f"    Data shape: (n_satellites, n_trajectories, n_points) = ({nbsatellite}, {n_trajectories}, {n_points})")
    
    return dic_datas_combined, trajectories_list


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
    tangents[0] = all_traj[1] - all_traj[0]  # Forward difference at start
    tangents[-1] = all_traj[-1] - all_traj[-2]  # Backward difference at end
    tangents[1:-1] = (all_traj[2:] - all_traj[:-2]) / 2  # Centered difference
    
    # Normalize tangent vectors
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    tangents /= norms

    return tangents


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
    
    ltraj = tangents[:, 0] * lx[trajectory[:, 0]] + \
            tangents[:, 1] * ly[trajectory[:, 1]] + \
            tangents[:, 2] * lz[trajectory[:, 2]]
    
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

def preprocess_trajectory_from_ini(ini_file,
                                   input_folder: str = "",
                                   verbose: bool = True):
    """
    Load configuration from an INI file and preprocess along trajectories.
    
    Parameters:
    -----------
    ini_file : str
        Path to the configuration .ini file (ex: "traj_satellite.ini")
    input_folder : str
        Path to folder containing OCA data (can be overridden by INI)
    verbose : bool
        Display detailed information
    
    Returns:
    -------
    dict : Results containing configuration, data, parameters and trajectories
    """
    
    setup_logging(Path(ini_file).stem)
    
    if verbose:
        logging.info("\n" + "="*70)
        logging.info(f"PREPROCESSING TRAJECTORY FROM {ini_file}")
    
    # Check that the file exists
    ini_path = Path(ini_file)
    if not ini_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {ini_file}")
    
    if verbose:
        logging.info(f"INI file found: {ini_path.absolute()}")
        logging.info("\nLoading configuration from INI file...")
    
    # Load all configuration from INI file
    try:
        (laws, terms, quantities, name_output, physical_param, trajectory_kwargs_list,
         nbsatellite, gap_satellite, input_folder, cycle, sim_type, di,
         trajectory_method, Ninterp) = load_config_from_ini(ini_file, input_folder)
    except Exception as e:
        logging.error(f"Error loading configuration: {e}")
        raise
    
    if verbose:
        logging.info(f"  Laws:              {laws}")
        logging.info(f"  Terms:             {terms}")
        logging.info(f"  Quantities:        {quantities}")
        logging.info(f"  Physical params:   {physical_param}")
        logging.info(f"  Nbsatellite:       {nbsatellite}")
        logging.info(f"  Gap satellite:     {gap_satellite}")
        logging.info(f"  Input folder:      {input_folder}")
        logging.info(f"  Cycle:             {cycle}")
        logging.info(f"  Sim type:          {sim_type}")
        logging.info(f"  Trajectory method: {trajectory_method}")
        logging.info(f"  N trajectory sets: {len(trajectory_kwargs_list)}")
    
    # Load OCA data (3D)
    try:
        dic_datas_3d, grid_param = load_oca_data(input_folder, cycle, sim_type)
    except Exception as e:
        logging.error(f"Error loading OCA data: {e}")
        raise
    
    # Select trajectory function based on configuration
    if trajectory_method == "linear_x":
        trajectory_func = trajectory_linear_x
    elif trajectory_method == "circular_xy":
        trajectory_func = trajectory_circular_xy
    elif trajectory_method == "helical":
        trajectory_func = trajectory_helical
    elif trajectory_method == "diagonal":
        trajectory_func = trajectory_diagonal
    else:
        raise ValueError(f"Unsupported trajectory method: {trajectory_method}")
    
    if verbose:
        logging.info("\n" + "-"*70)
        logging.info("PROCESSING TRAJECTORIES")

    # Prepare trajectory parameters for processing
    traj_param = {}
    traj_param['Ninterp'] = Ninterp
    traj_param['gap_satellite'] = gap_satellite
    traj_param['nbsatellite'] = nbsatellite
    traj_param['trajectory_kwargs_list'] = trajectory_kwargs_list

    # Generate all trajectories and extract field data
    dic_datas, trajectories_list = combine_multiple_trajectories(
        trajectory_func,
        dic_datas_3d,
        traj_param,
        grid_param,
        verbose=verbose
    )
    
    # Compute mean parameters from extracted data
    if 'ppar' in dic_datas:
        if nbsatellite == 1:
            physical_param["meanppar"] = [np.mean(arr) for arr in dic_datas['ppar']]
        else:
            physical_param["meanppar"] = {sat: [np.mean(arr) for arr in dic_datas['ppar'][sat]] 
                                     for sat in dic_datas['ppar']}
    if 'pperp' in dic_datas:
        if nbsatellite == 1:
            physical_param["meanpperp"] = [np.mean(arr) for arr in dic_datas['pperp']]
        else:
            physical_param["meanpperp"] = {sat: [np.mean(arr) for arr in dic_datas['pperp'][sat]] 
                                      for sat in dic_datas['pperp']}
    if 'rho' in dic_datas:
        if nbsatellite == 1:
            physical_param["rho_mean"] = [np.mean(arr) for arr in dic_datas['rho']]
        else:
            physical_param["rho_mean"] = {sat: [np.mean(arr) for arr in dic_datas['rho'][sat]] 
                                     for sat in dic_datas['rho']}
        
    if verbose:
        logging.info(f"\n  [OK] Extraction complete: {len(dic_datas)} field quantities")
        logging.info(f"    Total trajectories processed: {len(trajectories_list)}")
    
    # Save parameters to file for later use
    param_to_txt(grid_param, traj_param, physical_param)

    return {
            'laws': laws,
            'terms': terms,
            'quantities': quantities,
            'dic_datas': dic_datas,
            'grid_param': grid_param,
            'traj_param': traj_param,
            'physical_param': physical_param,
            'trajectory_name': trajectory_func.__name__.split('_', 1)[-1],  
            'name_output': name_output  
        }

    