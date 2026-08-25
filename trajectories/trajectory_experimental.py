# trajectory_experimental.py
"""
Module for loading and processing experimental satellite trajectory data.
Loads data from numpy arrays (1 or 4 satellites) and computes derived quantities.

Data structure:
    - Input: Lists of numpy arrays for each satellite and field
    - Output: {sat_0: {x: array(1, n_points), y, z, bx, by, bz, vx, vy, vz, rho, ppar, pperp}, ...}
    - Supports: 1 satellite or 4 satellites (for multi-point measurements)

Key features:
    - Computes tangent vectors (trajectory direction) from positions
    - Computes trajectory length and relative positions (4-satellite case)
    - Structures data uniformly with trajectory processing pipeline
"""

import logging
import numpy as np
import json
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from datetime import datetime
import configparser

logger = logging.getLogger(__name__)


# ========== UTILITY FUNCTIONS ==========

def setup_logging(config_name: str = "trajectory_experimental"):
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
# ========== EXPERIMENTAL TRAJECTORY DATA LOADER CLASS ==========

class ExperimentalTrajectoryDataLoader:
    """
    Load and process experimental satellite trajectory data from CSV files.
    
    Handles single and multi-satellite configurations, computes missing derived
    quantities (like tangent vectors), and structures data uniformly with the
    rest of the trajectory processing pipeline.
    """
    
    # ========== INITIALIZATION ==========
    
    def __init__(self, verbose: bool = False):
        """
        Initialize the experimental trajectory data loader.
        
        Parameters:
        -----------
        verbose : bool
            Enable detailed logging
        """
        self.verbose = verbose
        self.data = {}
        self.grid_param = {}
        self.trajectories = {}
        self.traj_param = {}
        self.physical_param = {}
        self.dic_datas = {}
        self.config = None
        
    # ========== PUBLIC METHODS ==========
    
    def load_datas_dict(self,
                x: List[np.ndarray]=None, y: List[np.ndarray]=None, z: List[np.ndarray]=None,
                bx: List[np.ndarray]=None, by: List[np.ndarray]=None, bz: List[np.ndarray]=None,
                vx: List[np.ndarray]=None, vy: List[np.ndarray]=None, vz: List[np.ndarray]=None,
                rho: List[np.ndarray]=None, ppar: List[np.ndarray]=None, pperp: List[np.ndarray]=None,
                nbsatellite: int = None) -> Tuple[Dict, Dict]:
        """
        Load experimental trajectory data from numpy arrays.
        
        Accepts lists of numpy arrays (one array per satellite) and structures them
        uniformly for processing.
        
        Parameters:
        -----------
        x, y, z : List[np.ndarray]
            Position arrays, one per satellite. Shape: (n_points,)
        bx, by, bz : List[np.ndarray]
            Magnetic field arrays, one per satellite. Shape: (n_points,)
        vx, vy, vz : List[np.ndarray]
            Velocity arrays, one per satellite. Shape: (n_points,)
        rho : List[np.ndarray]
            Density arrays, one per satellite. Shape: (n_points,)
        ppar, pperp : List[np.ndarray]
            Parallel and perpendicular pressure arrays, one per satellite. Shape: (n_points,)
        nbsatellite : int
            Number of satellites (1 or 4). Must match length of input arrays.
        
        Returns:
        -------
        None (stores internally in self.dic_datas)
        """
        
        if self.verbose:
            logging.info("\n" + "="*70)
            logging.info("LOADING EXPERIMENTAL TRAJECTORY DATA FROM ARRAY")
        
        try:
            dic_datas = self._process_satellite_data(x, y, z, bx, by, bz, vx, vy, vz, rho, ppar, pperp, nbsatellite)
        except Exception as e:
            logging.error(f"Error processing data: {e}")
            raise
        
        # Create minimal grid_param (no mesh)
        self.grid_param = {
            'N': np.array([1, 1, 1]),  # Not a mesh
            'L': np.array([1.0, 1.0, 1.0]),  # Not applicable
            'c': np.array([1.0, 1.0, 1.0]),  # Not applicable
            'source': 'experimental_satellite_data'
        }
        
        if self.verbose:
            logging.info("\n" + "-"*70)
            logging.info("DATA LOADING SUMMARY")
            for sat_name, sat_data in dic_datas.items():
                logging.info(f"  Satellite {sat_name}:")
                for field_name, field_data in sat_data.items():
                    logging.info(f"    - {field_name}: {field_data.shape}")
        
        self.dic_datas = dic_datas
    
    def compute_derived_quantities(self, compute_tangent: bool = True) -> Dict:
        """
        Compute derived quantities: tangent vectors, trajectory length, relative positions.
        
        **IMPORTANT:**
        - Tangent vectors: unit vectors along trajectory path (computed via finite differences)
        - Trajectory length: cumulative distance along trajectory
        - Relative positions (4-sat only): mean relative position of sat_1,2,3 w.r.t. sat_0
        
        Parameters:
        -----------
        compute_tangent : bool
            Whether to compute tangent vectors from positions. Default: True
        """
        
        if self.verbose:
            logging.info("\n" + "="*70)
            logging.info("COMPUTING DERIVED QUANTITIES")
        
        for sat_name, sat_data in self.dic_datas.items():
            
            # === TANGENT VECTORS ===
            # Unit vectors pointing along trajectory direction
            # Computed from finite differences: (r[i+1] - r[i-1]) / 2
            if compute_tangent and 'x' in sat_data and 'y' in sat_data and 'z' in sat_data:
                positions = np.array([sat_data['x'][0], sat_data['y'][0], sat_data['z'][0]]).T  # (n_points, 3)
                
                # Finite differences for interior points
                tangents = np.zeros_like(positions)
                tangents[1:-1] = (positions[2:] - positions[:-2]) / 2
                tangents[0] = positions[1] - positions[0]  # Forward difference at start
                tangents[-1] = positions[-1] - positions[-2]  # Backward difference at end
                
                # Normalize to unit vectors
                norms = np.linalg.norm(tangents, axis=1, keepdims=True)
                norms[norms == 0] = 1.0  # Avoid division by zero
                tangents_normalized = tangents / norms
                
                self.traj_param['tangents_list'] = np.array([tangents_normalized])
                
                if self.verbose:
                    logging.info(f"  [{sat_name}] Tangent vectors computed")
        
        # Compute ltraj_list for compatibility (one trajectory per satellite)
        ltraj = []
        x = self.dic_datas['sat_0']['x']
        y = self.dic_datas['sat_0']['y']
        z = self.dic_datas['sat_0']['z']
        ltraj_dist = 0
        ltraj.append(ltraj_dist)
        for i in range(1, len(x)):
            ltraj_dist += np.linalg.norm(np.array([x[i-1] - x[i], y[i-1] - y[i], z[i-1] - z[i]]))
            ltraj.append(ltraj_dist)
        
        self.traj_param['ltraj_list'] = np.array([np.array(ltraj)])

        # Compute relative positions of satellite if nbsatellite = 4
        if len(self.dic_datas) == 4:
            for i in range(1, 4):
                dRx = self.dic_datas[f'sat_{i}']['x'][0,:] - self.dic_datas['sat_0']['x'][0,:]
                dRy = self.dic_datas[f'sat_{i}']['y'][0,:] - self.dic_datas['sat_0']['y'][0,:]
                dRz = self.dic_datas[f'sat_{i}']['z'][0,:] - self.dic_datas['sat_0']['z'][0,:]
                dR = np.array([dRx, dRy, dRz])
                dR = np.mean(dR, axis=1)  # Average over time to get mean relative position
                self.traj_param[f'dR{i}'] = dR  # Store

        if self.verbose:
            logging.info(f"  Relative positions computed for 4 satellites")

        if self.verbose:
            logging.info(f"  [OK] Derived quantities computed for {len(self.dic_datas)} satellites")
            
    # ========== CONFIGURATION ==========

    def load_config(self, config_file: str):
        """
        Load all parameters from a .ini configuration file.

        Parameters:
        -----------
        config_file : str
            Path to the configuration .ini file (ex: "trajectory_experimental.ini")
            Must contain [RUN_PARAMS], [OUTPUT_DATA], [PHYSICAL_PARAMS] and [INPUT_DATA] sections.
        """
        ini_path = Path(config_file)
        if not ini_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")

        if self.verbose:
            logging.info(f"INI file found: {ini_path.absolute()}")

        config = configparser.ConfigParser()
        config.read(config_file)

        self.config = {
            "max_workers": config["INPUT_DATA"].getint("max_workers", 2),
            "nbsatellite": config["RUN_PARAMS"].getint("nbsatellite", None),
            "compute_tangent": config["RUN_PARAMS"].getboolean("compute_tangent", True),
            "laws": eval(config["OUTPUT_DATA"].get("laws", "[]")),
            "terms": eval(config["OUTPUT_DATA"].get("terms", "[]")),
            "quantities": eval(config["OUTPUT_DATA"].get("quantities", "[]")),
            "name_output": config["OUTPUT_DATA"].get("name_output", None),
            "method": config["RUN_PARAMS"].get("method", "fourier"),
            "Ninterp": config["RUN_PARAMS"].getint("Ninterp", 1),
        }

        if "PHYSICAL_PARAMS" in config:
            for key in config["PHYSICAL_PARAMS"].keys():
                try:
                    self.physical_param[key] = float(eval(config["PHYSICAL_PARAMS"][key]))
                except:
                    self.physical_param[key] = config["PHYSICAL_PARAMS"][key]

        if self.verbose:
            logging.info(f"  Nbsatellite:       {self.config['nbsatellite']}")
            logging.info(f"  Laws:              {self.config['laws']}")
            logging.info(f"  Method:            {self.config['method']}")
            logging.info(f"  Compute tangents:  {self.config['compute_tangent']}")

    # ========== MAIN WORKFLOW ==========

    def run(self,
            x: List[np.ndarray] = None, y: List[np.ndarray] = None, z: List[np.ndarray] = None,
            bx: List[np.ndarray] = None, by: List[np.ndarray] = None, bz: List[np.ndarray] = None,
            vx: List[np.ndarray] = None, vy: List[np.ndarray] = None, vz: List[np.ndarray] = None,
            rho: List[np.ndarray] = None, ppar: List[np.ndarray] = None, pperp: List[np.ndarray] = None) -> Dict:
        """
        Execute the full experimental preprocessing workflow.

        Loads the provided satellite data arrays, computes derived quantities
        (tangents, trajectory length, relative positions) and structures the
        trajectory/physical parameters for downstream computation.

        Parameters:
        -----------
        x, y, z : List[np.ndarray]
            Position arrays, one per satellite. Shape: (n_points,)
        bx, by, bz : List[np.ndarray]
            Magnetic field arrays, one per satellite. Shape: (n_points,)
        vx, vy, vz : List[np.ndarray]
            Velocity arrays, one per satellite. Shape: (n_points,)
        rho : List[np.ndarray]
            Density arrays, one per satellite. Shape: (n_points,)
        ppar, pperp : List[np.ndarray]
            Parallel and perpendicular pressure arrays, one per satellite. Shape: (n_points,)

        Returns:
        -------
        dict : Processing results with keys:
            - 'dic_datas': {sat_0: {x, y, z, bx, by, bz, vx, vy, vz, rho, ppar, pperp}, ...}
            - 'grid_param': Placeholder for compatibility
            - 'traj_param': {nbsatellite, n_points, tangents_list, ltraj_list, dR1, dR2, dR3, ...}
            - 'physical_param': {meanppar, meanpperp, rho_mean, ...}
            - 'laws', 'terms', 'quantities': From INI [OUTPUT_DATA]
            - 'method': Interpolation method (from INI)
            - 'name_output': Output prefix
            - 'max_workers': Number of parallel workers

        Raises:
        -------
        ValueError : If nbsatellite not 1 or 4
        """
        if self.config is None:
            raise RuntimeError("Configuration not loaded. Call load_config() before run().")

        nbsatellite = self.config["nbsatellite"]

        self.load_datas_dict(
            x=x, y=y, z=z, bx=bx, by=by, bz=bz,
            vx=vx, vy=vy, vz=vz, rho=rho, ppar=ppar, pperp=pperp,
            nbsatellite=nbsatellite,
        )
        self.compute_derived_quantities(compute_tangent=self.config["compute_tangent"])

        # Structure trajectory parameters (following preprocess_components format)
        first_sat = list(self.dic_datas.keys())[0] if self.dic_datas else 'sat_0'
        n_points = self.dic_datas[first_sat][list(self.dic_datas[first_sat].keys())[0]].shape[1]

        self.traj_param.update({
            'nbsatellite': nbsatellite,
            'Ninterp': self.config["Ninterp"],
            'n_trajectories': 1,  # One trajectory per satellite
            'n_points': n_points,
            'trajectory_kwargs_list': [{}],  # No kwargs for experimental data
            'trajectory_method': 'experimental',
            'source': 'experimental_satellite_measurements'
        })

        self._compute_stats()

        if self.verbose:
            logging.info("\n" + "-"*70)
            logging.info("DATA SUMMARY")
            logging.info(f"  Number of points per satellite: {n_points}")
            logging.info(f"  Number of satellites: {len(self.dic_datas)}")
            for sat_name, sat_data in self.dic_datas.items():
                logging.info(f"    {sat_name}: {len(sat_data)} fields")

        return {
            'laws': self.config["laws"],
            'terms': self.config["terms"],
            'quantities': self.config["quantities"],
            'dic_datas': self.dic_datas,
            'method': self.config["method"],
            'grid_param': self.grid_param,
            'traj_param': self.traj_param,
            'physical_param': self.physical_param,
            'trajectory_name': 'experimental',
            'name_output': self.config["name_output"],
            'max_workers': self.config["max_workers"]
        }

    # ========== PRIVATE METHODS ==========
    
    
    def _compute_stats(self):
        """Compute mean pressure and density statistics from the loaded data."""
        first_sat = list(self.dic_datas.keys())[0] if self.dic_datas else 'sat_0'

        if 'ppar' in self.dic_datas.get(first_sat, {}):
            self.physical_param["meanppar"] = {
                sat: [np.mean(self.dic_datas[sat]['ppar'])]
                for sat in self.dic_datas.keys()
            }

        if 'pperp' in self.dic_datas.get(first_sat, {}):
            self.physical_param["meanpperp"] = {
                sat: [np.mean(self.dic_datas[sat]['pperp'])]
                for sat in self.dic_datas.keys()
            }

        if 'rho' in self.dic_datas.get(first_sat, {}):
            self.physical_param["rho_mean"] = {
                sat: [np.mean(self.dic_datas[sat]['rho'])]
                for sat in self.dic_datas.keys()
            }

    def _process_satellite_data(self, x: List[np.ndarray], y: List[np.ndarray], z: List[np.ndarray],
                                bx: List[np.ndarray], by: List[np.ndarray], bz: List[np.ndarray],
                                vx: List[np.ndarray], vy: List[np.ndarray], vz: List[np.ndarray],
                                rho: List[np.ndarray], ppar: List[np.ndarray], pperp: List[np.ndarray],
                                nbsatellite: int = None) -> Dict:
        """
        Structure data from numpy arrays into dictionary format.
        Reshapes each 1D array to (1, n_points) for pipeline compatibility.
        
        **Supported configurations:**
        - nbsatellite=1: Single satellite
        - nbsatellite=4: Four satellites (multi-point measurement)
        
        Parameters:
        -----------
        x, y, z, bx, by, bz, vx, vy, vz, rho, ppar, pperp : List[np.ndarray]
            Data arrays, length = nbsatellite. Each array has shape (n_points,)
        nbsatellite : int
            Must be 1 or 4
        
        Returns:
        -------
        dict : {sat_0: {x: array(1,n), y: array(1,n), ...}, sat_1: {...}, ...}
        
        Raises:
        -------
        ValueError : If nbsatellite not in {1, 4}
        """
        
        dic_datas = {}
        
        if nbsatellite == 1:
            # Single satellite: reshape from (n_points,) to (1, n_points)
            dic_datas['sat_0'] = {
                'x': x[0][np.newaxis, :],
                'y': y[0][np.newaxis, :],
                'z': z[0][np.newaxis, :],
                'bx': bx[0][np.newaxis, :],
                'by': by[0][np.newaxis, :],
                'bz': bz[0][np.newaxis, :],
                'vx': vx[0][np.newaxis, :],
                'vy': vy[0][np.newaxis, :],
                'vz': vz[0][np.newaxis, :],
                'rho': rho[0][np.newaxis, :],
                'ppar': ppar[0][np.newaxis, :],
                'pperp': pperp[0][np.newaxis, :]
            }
            logging.info(f"  [OK] 1 satellite: {len(dic_datas['sat_0'])} fields")
            
        elif nbsatellite == 4:
            # 4 satellites: reshape each satellite's data
            dic_datas = {}
            for i in range(4):
                dic_datas[f'sat_{i}'] = {
                    'x': x[i][np.newaxis, :],
                    'y': y[i][np.newaxis, :],
                    'z': z[i][np.newaxis, :],
                    'bx': bx[i][np.newaxis, :],
                    'by': by[i][np.newaxis, :],
                    'bz': bz[i][np.newaxis, :],
                    'vx': vx[i][np.newaxis, :],
                    'vy': vy[i][np.newaxis, :],
                    'vz': vz[i][np.newaxis, :],
                    'rho': rho[i][np.newaxis, :],
                    'ppar': ppar[i][np.newaxis, :],
                    'pperp': pperp[i][np.newaxis, :]
                }
            logging.info(f"  [OK] 4 satellites: {len(dic_datas['sat_0'])} fields each")
            
        else:
            raise ValueError(f"Unsupported nbsatellite={nbsatellite}. Only 1 or 4 supported.")

        
        return dic_datas
