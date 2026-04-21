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
import json

from exact_laws.preprocessing.process_on_oca_files import (
    extract_quantities_from_OCA_file,
    extract_simu_param_from_OCA_file
)

from tools_trajectory_preprocessing import (
    trajectory_linear_x,
    trajectory_linear_y,
    trajectory_linear_z,
    trajectory_circular_xy,
    trajectory_helical,
    trajectory_diagonal,
    combine_multiple_trajectories,
    generate_all_trajectory_kwargs_linear_x,
    generate_all_trajectory_kwargs_linear_y,
    generate_all_trajectory_kwargs_linear_z,
)

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
                 filename: str = "parameters_summary.txt"):
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
    filename : str
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
        with open(filename, 'w') as f:
            json.dump(output_data, f, indent=4)
        logging.info(f"Parameters saved to {filename}")
    except Exception as e:
        logging.error(f"Error saving parameters to {filename}: {e}")
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
    
    # Check if 'all' is specified
    if trajectory_kwargs_str.strip().lower() == "'all'" or trajectory_kwargs_str.strip().lower() == '"all"':
        # Will be generated after loading grid parameters
        trajectory_kwargs_list = 'all'
    else:
        trajectory_kwargs_list = eval(trajectory_kwargs_str)
        
        # Ensure it's a list of dicts
        if isinstance(trajectory_kwargs_list, dict):
            trajectory_kwargs_list = [trajectory_kwargs_list]
        elif not isinstance(trajectory_kwargs_list, list):
            trajectory_kwargs_list = [{}]
    
    # Load RUN_PARAMS section
    try:
        method = config["RUN_PARAMS"].get("method", "fourier")
        nbsatellite = config["RUN_PARAMS"].getint("nbsatellite", 1)
        gap_satellite = config["RUN_PARAMS"].getfloat("gap_satellite", 1)
        trajectory_method = config["RUN_PARAMS"].get("trajectory_method", "linear_x")
        Ninterp = config["RUN_PARAMS"].getint("Ninterp", 1)
        step_traj = config["RUN_PARAMS"].getint("step_traj", 10)
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
    
    return (laws, terms, quantities, name_output, physical_param, method, trajectory_kwargs_list,
            nbsatellite, gap_satellite, input_folder, cycle, sim_type, di,
            trajectory_method, Ninterp, step_traj)

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
        (laws, terms, quantities, name_output, physical_param, method, trajectory_kwargs_list,
         nbsatellite, gap_satellite, input_folder, cycle, sim_type, di,
         trajectory_method, Ninterp, step_traj) = load_config_from_ini(ini_file, input_folder)
    except Exception as e:
        logging.error(f"Error loading configuration: {e}")
        raise
    
    if verbose:
        logging.info(f"  Laws:              {laws}")
        logging.info(f"  Method:            {method}")
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

    # Prepare trajectory parameters for processing
    traj_param = {}
    traj_param['Ninterp'] = Ninterp
    traj_param['gap_satellite'] = gap_satellite
    traj_param['nbsatellite'] = nbsatellite
    traj_param['step_traj'] = None

    # Generate all trajectory_kwargs if 'all' was specified
    if trajectory_kwargs_list == 'all':
        if trajectory_method == "linear_x":
            trajectory_kwargs_list = generate_all_trajectory_kwargs_linear_x(grid_param['N'], step_traj)
            name_output += f"_all_step{step_traj}"
            traj_param['step_traj'] = step_traj
            if verbose:
                logging.info(f"  Generating ALL trajectory positions for linear_x...")
                logging.info(f"    Total combinations: {len(trajectory_kwargs_list)} trajectories")
        elif trajectory_method == "linear_y":
            trajectory_kwargs_list = generate_all_trajectory_kwargs_linear_y(grid_param['N'], step_traj)
            name_output += f"_all_step{step_traj}"
            traj_param['step_traj'] = step_traj
            if verbose:
                logging.info(f"  Generating ALL trajectory positions for linear_y...")
                logging.info(f"    Total combinations: {len(trajectory_kwargs_list)} trajectories")
        elif trajectory_method == "linear_z":
            trajectory_kwargs_list = generate_all_trajectory_kwargs_linear_z(grid_param['N'], step_traj)
            name_output += f"_all_step{step_traj}"
            traj_param['step_traj'] = step_traj
            if verbose:
                logging.info(f"  Generating ALL trajectory positions for linear_z...")
                logging.info(f"    Total combinations: {len(trajectory_kwargs_list)} trajectories")
        else:
            logging.warning(f"  'all' mode not yet implemented for {trajectory_method}, using default")
            trajectory_kwargs_list = [{}]
    traj_param['trajectory_kwargs_list'] = trajectory_kwargs_list


    # Select trajectory function based on configuration
    if trajectory_method == "linear_x":
        trajectory_func = trajectory_linear_x
    elif trajectory_method == "linear_y":
        trajectory_func = trajectory_linear_y
    elif trajectory_method == "linear_z":
        trajectory_func = trajectory_linear_z
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

    return {
            'laws': laws,
            'terms': terms,
            'quantities': quantities,
            'dic_datas': dic_datas,
            'method': method,
            'grid_param': grid_param,
            'traj_param': traj_param,
            'physical_param': physical_param,
            'trajectory_name': trajectory_func.__name__.split('_', 1)[-1],  
            'name_output': name_output  
        }

