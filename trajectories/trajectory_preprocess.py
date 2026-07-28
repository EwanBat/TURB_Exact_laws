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

from trajectories.tools_trajectory_preprocessing import (
    trajectory_linear_x,
    trajectory_linear_minus_x,
    trajectory_linear_y,
    trajectory_linear_minus_y,
    trajectory_linear_z,
    trajectory_linear_minus_z,
    trajectory_linear_xy,
    combine_multiple_trajectories,
    generate_all_trajectory_kwargs_linear_x,
    generate_all_trajectory_kwargs_linear_y,
    generate_all_trajectory_kwargs_linear_z,
    generate_all_trajectory_kwargs_linear_xy,
)

logger = logging.getLogger(__name__)


# ========== UTILITY FUNCTIONS ==========

def setup_logging(config_name: str = "trajectory_preprocess"):
    """Create a logger with timestamp."""
    log_filename = f"{config_name}_{datetime.now().strftime('%d%m%Y_%H%M%S')}.log"
    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-7s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging


def param_to_txt(grid_param: dict, traj_param: dict, physical_param: dict, laws: list,
                 filename: str = "parameters_summary.txt"):
    """
    Save grid, trajectory, and physical parameters to a JSON file (with .txt extension).

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

    output_data = {
        'grid_param': convert_to_serializable(grid_param),
        'traj_param': convert_to_serializable(traj_param),
        'physical_param': convert_to_serializable(physical_param),
        'laws': convert_to_serializable(laws)
    }

    try:
        with open(filename, 'w') as f:
            json.dump(output_data, f, indent=4)
        logger.info(f"Parameters saved to {filename}")
    except Exception as e:
        logger.error(f"Error saving parameters to {filename}: {e}")
        raise


# ========== TRAJECTORY PREPROCESSOR CLASS ==========

class TrajectoryPreprocessor:
    """Preprocess satellite trajectories from OCA simulation data.

    Loads 3D field data, generates satellite trajectories, and extracts
    field quantities along those trajectories.

    Parameters:
    -----------
    verbose : bool
        Display detailed processing logs.
    """

    TRAJECTORY_METHODS = {
        "linear_x": trajectory_linear_x,
        "linear_minus_x": trajectory_linear_minus_x,
        "linear_y": trajectory_linear_y,
        "linear_minus_y": trajectory_linear_minus_y,
        "linear_z": trajectory_linear_z,
        "linear_minus_z": trajectory_linear_minus_z,
        "linear_xy": trajectory_linear_xy,
    }

    GENERATE_ALL_FUNCTIONS = {
        "linear_x": generate_all_trajectory_kwargs_linear_x,
        "linear_minus_x": generate_all_trajectory_kwargs_linear_x,
        "linear_y": generate_all_trajectory_kwargs_linear_y,
        "linear_minus_y": generate_all_trajectory_kwargs_linear_y,
        "linear_z": generate_all_trajectory_kwargs_linear_z,
        "linear_minus_z": generate_all_trajectory_kwargs_linear_z,
        "linear_xy": generate_all_trajectory_kwargs_linear_xy,
    }

    SUPPORTED_NBSATELLITE = (1, 4, 9)

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

        self.config = None
        self.grid_param = None
        self.traj_param = None
        self.physical_param = None
        self.run_params = None
        self.laws = None
        self.terms = None
        self.quantities = None
        self.name_output = None
        self.dic_datas_3d = None
        self.trajectory_kwargs_list = None
        self.trajectory_func = None
        self.trajectory_method = None

    # ========== CONFIGURATION ==========

    def load_config(self, config_file: str, input_folder: str = ""):
        """Load all parameters from a .ini configuration file.

        Parameters:
        -----------
        config_file : str
            Path to the configuration .ini file
        input_folder : str
            Default path to folder containing OCA data
        """
        if self.verbose:
            logger.info(f"Loading configuration from {config_file}...")

        config = configparser.ConfigParser()
        config.read(config_file)

        required_sections = ['RUN_PARAMS', 'INPUT_DATA']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Required section missing in INI file: [{section}]")

        # INPUT_DATA section
        input_folder = config["INPUT_DATA"].get("path", input_folder)
        cycle = config["INPUT_DATA"].get("cycle", "cycle_0")
        sim_type = config["INPUT_DATA"].get("sim_type", "OCA_CGL5").split("_")[-1]

        # OUTPUT_DATA section
        self.laws = eval(config["OUTPUT_DATA"].get("laws", "[]"))
        self.terms = eval(config["OUTPUT_DATA"].get("terms", "[]"))
        self.quantities = eval(config["OUTPUT_DATA"].get("quantities", "[]"))
        self.name_output = config["OUTPUT_DATA"].get("name_output", "trajectory_output")

        # PHYSICAL_PARAMS section
        self.physical_param = {}
        if "PHYSICAL_PARAMS" in config:
            for key in config["PHYSICAL_PARAMS"].keys():
                try:
                    self.physical_param[key] = float(eval(config["PHYSICAL_PARAMS"][key]))
                except:
                    self.physical_param[key] = config["PHYSICAL_PARAMS"][key]

        di = config["PHYSICAL_PARAMS"].getfloat("di", 1.0)
        self.physical_param["di"] = di

        # RUN_PARAMS section
        method = config["RUN_PARAMS"].get("method", None)
        Ninterp = config["RUN_PARAMS"].getint("Ninterp", None)
        max_workers = config["RUN_PARAMS"].getint("max_workers", np.nan)
        filter_enabled = config["RUN_PARAMS"].getboolean("filter", False)

        self.run_params = {
            "method": method,
            "Ninterp": Ninterp,
            "max_workers": max_workers,
            "filter_enabled": filter_enabled,
        }

        # TRAJECTORY_PARAMS section
        nbsatellite = config["TRAJECTORY_PARAMS"].getint("nbsatellite", None)
        gap_satellite = config["TRAJECTORY_PARAMS"].getfloat("gap_satellite", None)
        self.trajectory_method = config["TRAJECTORY_PARAMS"].get("trajectory_method", None)
        step_traj = config["TRAJECTORY_PARAMS"].getint("step_traj", None)

        trajectory_kwargs_str = config["TRAJECTORY_PARAMS"].get("trajectory_kwargs", "[{}]")
        if trajectory_kwargs_str.strip().lower() in ("'all'", '"all"'):
            self.trajectory_kwargs_list = 'all'
        else:
            self.trajectory_kwargs_list = eval(trajectory_kwargs_str)
            if isinstance(self.trajectory_kwargs_list, dict):
                self.trajectory_kwargs_list = [self.trajectory_kwargs_list]
            elif not isinstance(self.trajectory_kwargs_list, list):
                self.trajectory_kwargs_list = [{}]

        self.config = {
            "input_folder": input_folder,
            "cycle": cycle,
            "sim_type": sim_type,
            "nbsatellite": nbsatellite,
            "gap_satellite": gap_satellite,
            "step_traj": step_traj,
            "Ninterp": Ninterp,
        }

        if self.verbose:
            logger.info(f"  Input folder:      {input_folder}")
            logger.info(f"  Cycle:             {cycle}")
            logger.info(f"  Sim type:          {sim_type}")
            logger.info(f"  N threads:         {max_workers}")
            logger.info(f"  Laws:              {self.laws}")
            logger.info(f"  Physical params:   {self.physical_param}")
            logger.info(f"  Method:            {method}")
            logger.info(f"  N interp points:   {Ninterp}")
            logger.info(f"  Filter enabled:    {filter_enabled}")
            logger.info(f"  Nbsatellite:       {nbsatellite}")
            logger.info(f"  Gap satellite:     {gap_satellite}")
            logger.info(f"  Trajectory method: {self.trajectory_method}")

    # ========== DATA LOADING ==========

    def load_oca_data(self, input_folder: str = None, cycle: str = None, sim_type: str = None):
        """Load all required 3D OCA field data.

        Parameters:
        -----------
        input_folder : str, optional
            Path to folder containing 3Dfields_*.h5 files
        cycle : str, optional
            Name of the cycle (ex: "cycle_0")
        sim_type : str, optional
            Simulation type (ex: "CGL5")
        """
        if input_folder is None:
            input_folder = self.config["input_folder"]
        if cycle is None:
            cycle = self.config["cycle"]
        if sim_type is None:
            sim_type = self.config["sim_type"]

        if self.verbose:
            logger.info("\n" + "=" * 70)
            logger.info("LOADING OCA DATA")

        dic_datas = {}
        grid_param = {}

        with h5py.File(f"{input_folder}/3Dfields_v.h5", "r") as fv:
            param_key = "3Dgrid" if sim_type.endswith(("CGL3", "CGL5")) else "Simulation_Parameters"
            grid_param = extract_simu_param_from_OCA_file(fv, grid_param, param_key)
            (dic_datas["vx"],
             dic_datas["vy"],
             dic_datas["vz"]) = extract_quantities_from_OCA_file(fv, ["vx", "vy", "vz"], cycle)
        logger.info(f"  [OK] Velocity loaded:         {dic_datas['vx'].shape}")

        with h5py.File(f"{input_folder}/3Dfields_rho.h5", "r") as frho:
            dic_datas["rho"] = extract_quantities_from_OCA_file(frho, ["rho"], cycle)[0]
        logger.info(f"  [OK] Density loaded:          {dic_datas['rho'].shape}")

        with h5py.File(f"{input_folder}/3Dfields_b.h5", "r") as fb:
            (dic_datas["bx"],
             dic_datas["by"],
             dic_datas["bz"]) = extract_quantities_from_OCA_file(fb, ["bx", "by", "bz"], cycle)
        logger.info(f"  [OK] Magnetic field loaded:   {dic_datas['bx'].shape}")

        with h5py.File(f"{input_folder}/3Dfields_pi.h5", "r") as fp:
            (dic_datas["ppar"],
             dic_datas["pperp"]) = extract_quantities_from_OCA_file(fp, ["pparli", "pperpi"], cycle)
            dic_datas["ppar"] /= 2
            dic_datas["pperp"] /= 2
        logger.info(f"  [OK] Pressure loaded:         {dic_datas['ppar'].shape}")

        try:
            with h5py.File(f"{input_folder}/3Dfields_forcl_ampl.h5", "r") as ff:
                (dic_datas["fp"],
                 dic_datas["fm"]) = extract_quantities_from_OCA_file(ff, ["forcl_ampl_plus", "forcl_ampl_mins"], cycle)
            logger.info(f"  [OK] Force amplitude loaded:  {dic_datas['fp'].shape}")
        except Exception:
            logger.warning("  [SKIP] Force amplitude not loaded")

        if self.verbose:
            logger.info("\n" + "-" * 70)
            logger.info("DATA LOADING SUMMARY")
            logger.info(f"  Grid dimensions (N):  {grid_param['N']}")
            logger.info(f"  Domain size (L):      {grid_param['L']}")
            logger.info(f"  Cell spacing (c):     {grid_param['c']}")
            logger.info(f"  Data fields:          {len(dic_datas)} fields loaded")
            for field in sorted(dic_datas.keys()):
                logger.info(f"    - {field}")

        self.dic_datas_3d = dic_datas
        self.grid_param = grid_param

    # ========== TRAJECTORY SETUP ==========

    def _validate_nbsatellite(self):
        nbsatellite = self.config["nbsatellite"]
        if nbsatellite not in self.SUPPORTED_NBSATELLITE:
            raise ValueError(
                f"Unsupported nbsatellite value: {nbsatellite}. "
                f"Expected {self.SUPPORTED_NBSATELLITE}."
            )

    def _select_trajectory_func(self):
        method = self.trajectory_method
        if method not in self.TRAJECTORY_METHODS:
            raise ValueError(f"Unsupported trajectory method: {method}")
        self.trajectory_func = self.TRAJECTORY_METHODS[method]

    def _setup_traj_param(self):
        Ninterp = self.run_params["Ninterp"]
        gap_satellite = self.config["gap_satellite"]
        nbsatellite = self.config["nbsatellite"]
        step_traj = self.config["step_traj"]

        self.traj_param = {
            'Ninterp': Ninterp,
            'gap_satellite': gap_satellite,
            'nbsatellite': nbsatellite,
            'step_traj': None,
            'trajectory_method': self.trajectory_method,
            'trajectory_func': self.trajectory_func,
        }

        if self.trajectory_kwargs_list == 'all':
            self._generate_all_kwargs(step_traj)

        self.traj_param['trajectory_kwargs_list'] = self.trajectory_kwargs_list

    def _generate_all_kwargs(self, step_traj: int):
        generate_func = self.GENERATE_ALL_FUNCTIONS.get(self.trajectory_method)
        if generate_func is None:
            logger.warning(
                f"  'all' mode not yet implemented for {self.trajectory_method}, using default"
            )
            self.trajectory_kwargs_list = [{}]
            return

        self.trajectory_kwargs_list = generate_func(self.grid_param['N'], step_traj)
        self.name_output += f"_all_step{step_traj}"
        self.traj_param['step_traj'] = step_traj

        if self.verbose:
            logger.info(f"  Generating ALL trajectory positions for {self.trajectory_method}...")
            logger.info(f"    Total combinations: {len(self.trajectory_kwargs_list)} trajectories")

    # ========== MEAN PARAMETERS ==========

    def _compute_mean_params(self, dic_datas: dict):
        first_sat = list(dic_datas.keys())[0] if dic_datas else 'sat_0'

        if 'ppar' in dic_datas.get(first_sat, {}):
            self.physical_param["meanppar"] = {
                sat: [np.mean(dic_datas[sat]['ppar'][traj_idx])
                      for traj_idx in range(self.traj_param['n_trajectories'])]
                for sat in dic_datas.keys()
            }

        if 'pperp' in dic_datas.get(first_sat, {}):
            self.physical_param["meanpperp"] = {
                sat: [np.mean(dic_datas[sat]['pperp'][traj_idx])
                      for traj_idx in range(self.traj_param['n_trajectories'])]
                for sat in dic_datas.keys()
            }

        if 'rho' in dic_datas.get(first_sat, {}):
            self.physical_param["rho_mean"] = {
                sat: [np.mean(dic_datas[sat]['rho'][traj_idx])
                      for traj_idx in range(self.traj_param['n_trajectories'])]
                for sat in dic_datas.keys()
            }

    # ========== MAIN WORKFLOW ==========

    def run(self):
        """Execute the full preprocessing workflow.

        Returns:
        -------
        dict
            Results containing configuration, data, parameters and trajectories.
        """
        if self.verbose:
            logger.info("\n" + "=" * 70)
            logger.info("PREPROCESSING TRAJECTORY")

        self._validate_nbsatellite()
        self._select_trajectory_func()
        self._setup_traj_param()

        if self.verbose:
            logger.info("\n" + "-" * 70)
            logger.info("PROCESSING TRAJECTORIES")

        dic_datas = combine_multiple_trajectories(
            self.trajectory_func,
            self.dic_datas_3d,
            self.traj_param,
            self.grid_param,
            verbose=self.verbose,
        )

        self._compute_mean_params(dic_datas)

        if self.verbose:
            logger.info(f"\n  [OK] Extraction complete: {len(dic_datas)} field quantities")
            logger.info(f"    Total trajectories processed: {len(self.traj_param['trajectories_list'])}")

        return {
            'laws': self.laws,
            'terms': self.terms,
            'quantities': self.quantities,
            'dic_datas': dic_datas,
            'grid_param': self.grid_param,
            'traj_param': self.traj_param,
            'physical_param': self.physical_param,
            'run_params': self.run_params,
            'trajectory_name': self.trajectory_func.__name__.split('_', 1)[-1],
            'name_output': self.name_output,
        }


# ========== CONVENIENCE FUNCTION ==========

def preprocess_trajectory_from_ini(ini_file: str, input_folder: str = "", verbose: bool = True):
    """Load configuration from an INI file and preprocess along trajectories.

    Convenience wrapper around TrajectoryPreprocessor.

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

    ini_path = Path(ini_file)
    if not ini_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {ini_file}")

    if verbose:
        logger.info(f"INI file found: {ini_path.absolute()}")

    preprocessor = TrajectoryPreprocessor(verbose=verbose)
    preprocessor.load_config(ini_file, input_folder)

    if preprocessor.config["nbsatellite"] not in TrajectoryPreprocessor.SUPPORTED_NBSATELLITE:
        raise ValueError(
            f"Unsupported nbsatellite value: {preprocessor.config['nbsatellite']}. "
            f"Expected {TrajectoryPreprocessor.SUPPORTED_NBSATELLITE}."
        )

    preprocessor.load_oca_data()
    result = preprocessor.run()

    return result
