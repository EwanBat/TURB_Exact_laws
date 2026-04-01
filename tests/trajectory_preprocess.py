"""
Preprocessing module for satellite trajectories.
Encapsulates loading OCA data and retrieving required quantities.
Support for custom trajectories - simple indexing in the cube.
"""

import logging
import numpy as np
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
                        N: np.ndarray) -> np.ndarray:
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
                           N: np.ndarray) -> np.ndarray:
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
                       N: np.ndarray) -> np.ndarray:
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
    # X progression with pitch
    x = (t / N[0]) * pitch
    
    # Normalized angle parameter
    theta = 2 * np.pi * t / N[0]
    
    y = center_y + radius * np.cos(theta)
    z = center_z + radius * np.sin(theta)
    
    # Clip aux limites
    x = np.clip(x, 0, N[0]-1).astype(int)
    y = np.clip(y, 0, N[1]-1).astype(int)
    z = np.clip(z, 0, N[2]-1).astype(int)
    
    return np.array([x, y, z]).T


def trajectory_diagonal(t: np.ndarray, N: np.ndarray) -> np.ndarray:
    """
    Diagonal trajectory through the cube.
    
    Parameters:
    -----------
    t : np.ndarray
        Trajectory parameter (0 to N[0]-1)
    N : np.ndarray
        Grid dimensions
    
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
    logging.info("="*70)
    
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
    logging.info("-"*70)
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
    
    lx = np.arange(0, dic_param['N'][0]) * dic_param['c'][0]
    ly = np.arange(0, dic_param['N'][1]) * dic_param['c'][1]
    lz = np.arange(0, dic_param['N'][2]) * dic_param['c'][2]

    x = lx[trajectory[:, 0]]
    y = ly[trajectory[:, 1]]
    z = lz[trajectory[:, 2]]

    dic_param['lx'] = lx
    dic_param['ly'] = ly
    dic_param['lz'] = lz

def preprocess_trajectory_from_ini(ini_file, 
                                   trajectory_func: Callable = None,
                                   trajectory_kwargs: dict = None,
                                   input_folder: str = "data_oca",
                                   verbose: bool = True):
    """
    Load configuration from an INI file and preprocess along a trajectory.
    
    Parameters:
    -----------
    ini_file : str
        Path to the configuration .ini file (ex: "traj_satellite.ini")
    trajectory_func : Callable, optional
        Function returning a trajectory: f(t, N, **trajectory_kwargs) -> np.ndarray (n_points, 3)
        If None, uses trajectory_linear_x with y_pos=100, z_pos=100
    trajectory_kwargs : dict, optional
        Additional arguments for trajectory_func (all in indices, not physical units)
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
        logging.info("="*70)
    
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
        logging.info("-"*70)
        logging.info(f"  Nbsatellite = {nbsatellite} -> gradients {'enabled' if nbsatellite > 1 else 'disabled'}")
    
    # Default trajectory if none provided
    if trajectory_func is None:
        trajectory_func = trajectory_linear_x
        trajectory_kwargs = {'y_pos': 100, 'z_pos': 100}
    
    if trajectory_kwargs is None:
        trajectory_kwargs = {}
    
    # Generate trajectory
    if verbose:
        logging.info("\n" + "-"*70)
        logging.info("GENERATING TRAJECTORY")
        logging.info("-"*70)
        logging.info(f"  Trajectory function: {trajectory_func.__name__}")
        logging.info(f"  Trajectory kwargs:   {trajectory_kwargs}")
        logging.info(f"  Number of satellites: {nbsatellite}")
    
    try:
        t = np.arange(dic_param['N'][0])
        trajectory = trajectory_func(t, N=dic_param['N'], **trajectory_kwargs)
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
        logging.info("-"*70)
    
    dic_datas = extract_quantities_along_trajectory(
        dic_datas_3d, 
        trajectory, 
        nbsatellite=nbsatellite, 
        gap_satellite=int(gap_satellite)
    )

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
        'trajectory': trajectory
    }
