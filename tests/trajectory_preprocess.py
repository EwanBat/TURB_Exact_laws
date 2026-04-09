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
    dic_param = {}
    
    # Load velocity
    with h5py.File(f"{input_folder}/3Dfields_v.h5", "r") as fv:
        param_key = "3Dgrid" if sim_type.endswith(("CGL3", "CGL5")) else "Simulation_Parameters"
        dic_param = extract_simu_param_from_OCA_file(fv, dic_param, param_key)
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
    logging.info(f"  Grid dimensions (N):  {dic_param['N']}")
    logging.info(f"  Domain size (L):      {dic_param['L']}")
    logging.info(f"  Cell spacing (c):     {dic_param['c']}")
    logging.info(f"  Data fields:          {len(dic_datas)} fields loaded")
    for field in sorted(dic_datas.keys()):
        logging.info(f"    - {field}")
    
    return dic_datas, dic_param


def load_config_from_ini(config_file):
    """
    Load parameters from a .ini file
    
    Returns:
    -------
    tuple : (laws, terms, quantities, physical_params)
    """
    config = configparser.ConfigParser()
    config.read(config_file)
    
    laws = eval(config["OUTPUT_DATA"].get("laws", "[]"))
    terms = eval(config["OUTPUT_DATA"].get("terms", "[]"))
    quantities = eval(config["OUTPUT_DATA"].get("quantities", "[]"))
    
    physical_params = {}
    if "PHYSICAL_PARAMS" in config:
        for key in config["PHYSICAL_PARAMS"].keys():
            try:
                physical_params[key] = float(eval(config["PHYSICAL_PARAMS"][key]))
            except:
                physical_params[key] = config["PHYSICAL_PARAMS"][key]
    
    return laws, terms, quantities, physical_params

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
                trajectory_data[key] = np.array([
                    dic_datas[key][int(trajectory[i, 0]), int(trajectory[i, 1]), int(trajectory[i, 2])]
                    for i in range(n_points)
                ])
                # trajectory_data[key] = np.append(trajectory_data[key], trajectory_data[key][0])  # Add one point to have a periodic trajectory for Fourier transforms
            else:
                trajectory_data[key] = dic_datas[key]
                # trajectory_data[key] = np.append(trajectory_data[key], trajectory_data[key][0])  # Add one point to have a periodic trajectory for Fourier transforms
    
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

def interpolation_along_trajectory(trajectory: np.ndarray, array_data: np.ndarray, dic_param: dict) -> dict:
    """
    Interpolate data along the trajectory.

    Parameters:
    -----------
    trajectory : np.ndarray
        Trajectory in indices (n_points, 3)
    array_data : np.ndarray
        Data to interpolate (3D array)
    dic_param : dict
        Dictionary containing simulation parameters

    Returns:
    -------
    np.ndarray
        Interpolated data along the trajectory (n_points,)
    """
    x, y, z = np.arange(dic_param['N'][0]), np.arange(dic_param['N'][1]), np.arange(dic_param['N'][2])
    rbf_interpolator = Rbf(x, y, z, array_data, function='thin_plate')

    # Interpolate along the trajectory
    interpolated_values = rbf_interpolator(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2])
    return interpolated_values


def extract_coordinates_along_trajectory(trajectory: np.ndarray, dic_param: dict) -> dict:
    """
    Extract physical coordinates (x, y, z) along the trajectory from indices.
    
    Parameters:
    -----------
    trajectory : np.ndarray
        Trajectory in indices (n_points, 3)
    dic_param : dict
        Dictionary containing simulation parameters, particularly 'lx', 'ly', 'lz'
    
    Returns:
    -------
    dict : Dictionary with physical coordinates along the trajectory
           {'x': (n_points,), 'y': (n_points,), 'z': (n_points,)}
    """
    
    lx = dic_param['lx'] - dic_param['L'][0] / 2  # Centering coordinates around zero
    ly = dic_param['ly'] - dic_param['L'][1] / 2
    lz = dic_param['lz'] - dic_param['L'][2] / 2
    tangents = dic_param['tangents']

    ltraj = tangents[:, 0] * lx[trajectory[:, 0]] + tangents[:, 1] * ly[trajectory[:, 1]] + tangents[:, 2] * lz[trajectory[:, 2]]
    dic_param['ltraj'] = ltraj

def compute_tangent_vectors_from_parameter(trajectory_func: Callable, 
                                           t_array: np.ndarray, 
                                           dic_param: dict,
                                           **kwargs) -> np.ndarray:
    """
    Compute tangent vectors by numerical differentiation of the trajectory function.
    
    Parameters:
    -----------
    trajectory_func : Callable
        Function that generates trajectory (e.g., trajectory_linear_x)
    t_array : np.ndarray
        Parameter values (0 to N[0]-1)
    N : np.ndarray
        Grid dimensions
    dic_param : dict
        Contains 'lx', 'ly', 'lz' for physical coordinates
    **kwargs : dict
        Arguments for trajectory_func
    
    Returns:
    -------
    np.ndarray
        Normalized tangent vectors (n_points, 3)
    """
    n_points = len(t_array)
    dt = 1.0  # Index spacing
    tangents = np.zeros((n_points, 3))
    N = dic_param['N']

    lx = np.arange(dic_param['N'][0]) * dic_param['c'][0]
    ly = np.arange(dic_param['N'][1]) * dic_param['c'][1]
    lz = np.arange(dic_param['N'][2]) * dic_param['c'][2]
    
    for i in range(n_points):
        if i == 0:
            t_curr = t_array[i]
            t_next = t_array[i+1]
        elif i == n_points - 1:
            t_prev = t_array[i-1]
            t_curr = t_array[i]
        else:
            t_prev = t_array[i-1]
            t_next = t_array[i+1]
        
        if i == 0:
            traj_curr = trajectory_func(np.array([t_curr]), N=N, **kwargs)[0]
            traj_next = trajectory_func(np.array([t_next]), N=N, **kwargs)[0]
            pos_curr = np.array([lx[traj_curr[0]], ly[traj_curr[1]], lz[traj_curr[2]]])
            pos_next = np.array([lx[traj_next[0]], ly[traj_next[1]], lz[traj_next[2]]])
            tangent = pos_next - pos_curr
        elif i == n_points - 1:
            traj_prev = trajectory_func(np.array([t_prev]), N=N, **kwargs)[0]
            traj_curr = trajectory_func(np.array([t_curr]), N=N, **kwargs)[0]
            pos_prev = np.array([lx[traj_prev[0]], ly[traj_prev[1]], lz[traj_prev[2]]])
            pos_curr = np.array([lx[traj_curr[0]], ly[traj_curr[1]], lz[traj_curr[2]]])
            tangent = pos_curr - pos_prev
        else:
            traj_prev = trajectory_func(np.array([t_prev]), N=N, **kwargs)[0]
            traj_next = trajectory_func(np.array([t_next]), N=N, **kwargs)[0]
            pos_prev = np.array([lx[traj_prev[0]], ly[traj_prev[1]], lz[traj_prev[2]]])
            pos_next = np.array([lx[traj_next[0]], ly[traj_next[1]], lz[traj_next[2]]])
            tangent = pos_next - pos_prev
        
        # Normalize
        norm = np.linalg.norm(tangent,axis=0)
        if norm > 0:
            tangents[i] = tangent / norm
        else:
            tangents[i] = np.array([1.0, 0.0, 0.0])
    
    dic_param['lx'] = lx
    dic_param['ly'] = ly
    dic_param['lz'] = lz
    dic_param['tangents'] = tangents

def preprocess_trajectory_from_ini(ini_file,
                                   input_folder: str = "",
                                   verbose: bool = True):
    """
    Load configuration from an INI file and preprocess along a trajectory.
    
    Parameters:
    -----------
    ini_file : str
        Path to the configuration .ini file (ex: "traj_satellite.ini")
    input_folder : str
        Path to folder containing OCA data
    verbose : bool
        Display detailed information
    
    Returns:
    -------
    dict : Results containing:
        - 'config': Loaded configuration
        - 'required_quantities': Required quantities
        - 'dic_datas': Raw data extracted along trajectory (1D)
        - 'dic_param': Simulation parameters
        - 'trajectory': Trajectory points used (indices)
    """
    
    # Setup logging
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
    
    # Load configuration from INI file
    if verbose:
        logging.info("\nLoading configuration from INI file...")
    
    try:
        laws, terms, quantities, physical_params = load_config_from_ini(ini_file)
    except KeyError as e:
        logging.error(f"Missing key in INI file: {e}")
        raise
    except Exception as e:
        logging.error(f"Error reading INI file: {e}")
        raise
    
    # Load additional parameters from INI
    config = configparser.ConfigParser()
    
    try:
        config.read(ini_file)
    except Exception as e:
        logging.error(f"Error parsing INI file: {e}")
        raise
    
    # Check required sections
    required_sections = ['RUN_PARAMS', 'INPUT_DATA']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Required section missing in INI file: [{section}]")
    
    try:
        nbsatellite = config["RUN_PARAMS"].getint("nbsatellite", 1)
        gap_satellite = config["RUN_PARAMS"].getfloat("gap_satellite", 1)
        input_folder = config["INPUT_DATA"].get("path", input_folder)
        cycle = config["INPUT_DATA"].get("cycle", "cycle_0")
        sim_type = config["INPUT_DATA"].get("sim_type", "OCA_CGL5").split("_")[-1]
        di = config["PHYSICAL_PARAMS"].getfloat("di", 1.0)
        trajectory_method = config["RUN_PARAMS"].get("trajectory_method", "linear_x")
        trajectory_kwargs = eval(config["RUN_PARAMS"].get("trajectory_kwargs", "{}"))
        Ninterp = config["RUN_PARAMS"].getint("Ninterp", 1)
    except Exception as e:
        logging.error(f"Error reading parameters: {e}")
        raise
    
    if verbose:
        logging.info(f"  Laws:            {laws}")
        logging.info(f"  Terms:           {terms}")
        logging.info(f"  Quantities:      {quantities}")
        logging.info(f"  Physical params: {physical_params}")
        logging.info(f"  Nbsatellite:     {nbsatellite}")
        logging.info(f"  Gap satellite:   {gap_satellite}")
        logging.info(f"  Input folder:    {input_folder}")
        logging.info(f"  Cycle:           {cycle}")
        logging.info(f"  Sim type:        {sim_type}")
        logging.info(f"  Ion inertial length (di): {di}")
    
    # Load OCA data (3D)
    try:
        dic_datas_3d, dic_param = load_oca_data(input_folder, cycle, sim_type)
    except Exception as e:
        logging.error(f"Error loading OCA data: {e}")
        raise
    
    # Determine required quantities
    if verbose:
        logging.info("\n" + "-"*70)
        logging.info("DETERMINING REQUIRED QUANTITIES")
        logging.info(f"  Nbsatellite = {nbsatellite} -> gradients {'enabled' if nbsatellite > 1 else 'disabled'}")
    
    # Default trajectory if none provided
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
    
    # Generate trajectory
    if verbose:
        logging.info("\n" + "-"*70)
        logging.info("GENERATING TRAJECTORY")
        logging.info(f"  Trajectory function: {trajectory_func.__name__}")
        logging.info(f"  Trajectory kwargs:   {trajectory_kwargs}")
        logging.info(f"  Number of satellites: {nbsatellite}")
    
    try:
        t = np.arange(Ninterp * dic_param['N'][0]) / Ninterp  # Parameter for trajectory generation
        trajectory = trajectory_func(t, N=dic_param['N'], Ninterp=Ninterp, **trajectory_kwargs)
    except Exception as e:
        logging.error(f"Error generating trajectory: {e}")
        raise
    
    if verbose:
        logging.info(f"  Trajectory points: {len(trajectory)}")
        if nbsatellite == 1:
            logging.info(f"  First point (indices): {trajectory[0]}")
            logging.info(f"  Last point (indices):  {trajectory[-1]}")
        else:
            logging.info(f"  Central trajectory:")
            logging.info(f"    First point (indices): {trajectory[0]}")
            logging.info(f"    Last point (indices):  {trajectory[-1]}")
            logging.info(f"  Separation between satellites: {gap_satellite}")
    
    # Extract quantities along trajectory
    if verbose:
        logging.info("\n" + "-"*70)
        logging.info("EXTRACTING DATA ALONG TRAJECTORY")
    
    dic_datas = extract_quantities_along_trajectory(
        dic_datas_3d, 
        trajectory, 
        nbsatellite=nbsatellite, 
        gap_satellite=int(gap_satellite)
    )

    # Compute tangent vectors along the trajectory
    compute_tangent_vectors_from_parameter(trajectory_func, t, dic_param, Ninterp=Ninterp, **trajectory_kwargs)
    # Get physical coordinates along the trajectory
    extract_coordinates_along_trajectory(trajectory, dic_param) # Physical coordinates to dic_param
    
    # Add means if available
    if 'ppar' in dic_datas:
        dic_param["meanppar"] = np.mean(dic_datas['ppar'],axis=0) if nbsatellite == 1 else {sat: np.mean(dic_datas['ppar'][sat]) for sat in dic_datas['ppar']}
    if 'pperp' in dic_datas:
        dic_param["meanpperp"] = np.mean(dic_datas['pperp'],axis=0) if nbsatellite == 1 else {sat: np.mean(dic_datas['pperp'][sat]) for sat in dic_datas['pperp']}
    if 'rho' in dic_datas:
        dic_param["rho_mean"] = np.mean(dic_datas['rho'],axis=0) if nbsatellite == 1 else {sat: np.mean(dic_datas['rho'][sat]) for sat in dic_datas['rho']}
    dic_param["di"] = di

    if verbose:
        logging.info(f"  Extracted {len(dic_datas)} quantities")
        if nbsatellite == 1:
            logging.info(f"  Trajectory length: {len(trajectory)} points")
            for key in sorted(dic_datas.keys()):
                if isinstance(dic_datas[key], np.ndarray):
                    logging.info(f"    - {key}: shape={dic_datas[key].shape} | min={dic_datas[key].min():.3e} | max={dic_datas[key].max():.3e}")
        else:
            logging.info(f"  Number of satellites: {nbsatellite}")
            logging.info(f"  Trajectory length: {len(trajectory)} points")
            for key in sorted(dic_datas.keys()):
                if isinstance(dic_datas[key], dict):
                    logging.info(f"    - {key}:")
                    for sat_name in sorted(dic_datas[key].keys()):
                        data = dic_datas[key][sat_name]
                        logging.info(f"        {sat_name}: shape={data.shape} | min={data.min():.3e} | max={data.max():.3e}")
                elif isinstance(dic_datas[key], np.ndarray):
                    logging.info(f"    - {key}: shape={dic_datas[key].shape} | min={dic_datas[key].min():.3e} | max={dic_datas[key].max():.3e}")
        logging.info("="*70 + "\n")
    
    return {
        'config': {
            'laws': laws,
            'terms': terms,
            'quantities': quantities,
            'physical_params': physical_params,
            'nbsatellite': nbsatellite,
            'gap_satellite': gap_satellite
        },
        'dic_datas': dic_datas,  # 1D data extracted along trajectory/trajectories
        'dic_param': dic_param,
        'trajectory': trajectory,
        'trajectory_name': trajectory_func.__name__.split('_', 1)[-1]  # e.g. "linear_x", "circular_xy"
    }
