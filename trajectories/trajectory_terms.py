# trajectory_terms.py
"""
Module to compute terms along a trajectory.
Analog to trajectory_quantities.py but for terms.
Uses calc_fourier() methods from terms for trajectories.
Encapsulated in TrajectoryTermsComputer class for better parameter management.
"""

import numpy as np
import logging
import h5py
from exact_laws.el_calc_mod.laws import LAWS
from exact_laws.el_calc_mod.terms import TERMS

logger = logging.getLogger(__name__)


# ========== TRAJECTORY TERMS COMPUTER CLASS ==========

class TrajectoryTermsComputer:
    """
    Compute physics terms along trajectories.
    
    Encapsulates term computation logic with parameter storage as instance attributes
    to reduce repeated parameter passing.
    Handles both single satellite and 4-satellite formation configurations.
    """
    
    # ========== CLASS CONSTANTS ==========
    
    # Mapping of abstract variables to their concrete components
    VARIABLE_COMPONENTS = {
        # === Raw data ===
        'v': ['vx', 'vy', 'vz'],
        'Iv': ['Ivx', 'Ivy', 'Ivz'],
        'b': ['bx', 'by', 'bz'],
        'Ib': ['Ibx', 'Iby', 'Ibz'],
        'w': ['wx', 'wy', 'wz'],  # Vitesse de vent (compressible)
        'j': ['jx', 'jy', 'jz'],  # Courant
        'Ij': ['Ijx', 'Ijy', 'Ijz'],  # Courant incompressible
        'f': ['fx', 'fy', 'fz'],  # Force
        
        # === Scalaires directs ===
        'rho': ['rho'],
        'Irho': ['Irho'],
        'pm': ['pm'],  # Magnetic pressure
        'Ipm': ['Ipm'],  # Incompressible magnetic pressure
        'pgyr': ['pgyr'],  # Gyrotropic pressure
        'Ipgyr': ['Ipgyr'],  # Incompressible gyrotropic pressure
        'piso': ['piso'],  # Isotropic pressure
        'Ipiso': ['Ipiso'],  # Incompressible isotropic pressure
        'ppol': ['ppol'],  # Poloidal pressure
        'Ippol': ['Ippol'],  # Incompressible poloidal pressure
        'pcgl': ['pcgl'],  # CGL pressure
        'Ipcgl': ['Ipcgl'],  # Incompressible CGL pressure
        'ugyr': ['ugyr'],  # Gyrotropic velocity
        'Iugyr': ['Iugyr'],  # Incompressible gyrotropic velocity
        'uiso': ['uiso'],  # Isotropic velocity
        'Iuiso': ['Iuiso'],  # Incompressible isotropic velocity
        'upol': ['upol'],  # Poloidal velocity
        'Iupol': ['Iupol'],  # Incompressible poloidal velocity
        'ucgl': ['ucgl'],  # CGL velocity
        'Iucgl': ['Iucgl'],  # Incompressible CGL velocity
        
        # === Divergences ===
        'divv': ['divvx', 'divvy', 'divvz'],  # Velocity divergence
        'divb': ['divbx', 'divby', 'divbz'],  # Magnetic field divergence
        'divj': ['divjx', 'divjy', 'divjz'],  # Current divergence
        
        # === Gradients ===
        'gradrho': ['dxrho', 'dyrho', 'dzrho'],  # Density gradient
        'gradv': ['grad_v_x', 'grad_v_y', 'grad_v_z'],  # Velocity gradient
        'graduiso': ['grad_uiso_x', 'grad_uiso_y', 'grad_uiso_z'],  # Isotropic velocity gradient
        'gradupol': ['grad_upol_x', 'grad_upol_y', 'grad_upol_z'],  # Poloidal velocity gradient
        
        # === Hyperdissipation ===
        'hdk': ['hdkx', 'hdky', 'hdkz'],  # Kinetic hyperdissipation
        'hdm': ['hdmx', 'hdmy', 'hdmz'],  # Magnetic hyperdissipation
        'hdk2': ['hdk2x', 'hdk2y', 'hdk2z'],  # Kinetic hyperdissipation order 2
    }
    
    SATELLITE_NAMES = ['sat_0', 'sat_1', 'sat_2', 'sat_3']
    
    # ========== INITIALIZATION ==========
    
    def __init__(self, verbose: bool = False, physical_param: dict = None, traj_param: dict = None):
        """
        Initialize the trajectory terms computer.
        
        Parameters:
        -----------
        verbose : bool
            Enable detailed logging
        physical_param : dict, optional
            Physical parameters
        traj_param : dict, optional
            Trajectory parameters
        """
        self.verbose = verbose
        self.physical_param = physical_param or {}
        self.traj_param = traj_param or {}
        self.nbsatellite = self.traj_param.get('nbsatellite', 1)
    
    # ========== PUBLIC METHODS ==========
    
    def list_required_terms(self, laws: list = None):
        """
        Returns the list of required terms to compute the given laws.
        
        Parameters:
        -----------
        laws : list[str]
            List of law names
        
        Returns:
        -------
        set : Set of required terms
        """
        if laws is None:
            laws = []
        
        terms = set()
        
        if not laws:
            return terms
        
        # Prepare parameters for terms_and_coeffs (convert lists to scalars)
        params_clean = self._prepare_dic_param_for_terms_and_coeffs(self.physical_param)
        
        # Add terms from laws
        for law_name in laws:
            if law_name in LAWS:
                law_obj = LAWS[law_name]
                # terms_and_coeffs() returns (terms_list, coeffs_dict)
                law_terms, _ = law_obj.terms_and_coeffs(params_clean)
                terms.update(law_terms)
        
        return terms
    
    def compute_term_from_TERMS(self, term_name: str, dic_quant: dict, method: str = None):
        """
        Compute a single term using the calc_fourier method from TERMS.
        
        Adapted for 1D trajectory data. Handles single satellite or 4-satellite 
        formations by extracting satellite-specific data before computation.
        
        Parameters:
        -----------
        term_name : str
            Name of the term to compute (e.g., "flux_dvdvdv", "bg17_vwv")
        dic_quant : dict
            Dictionary of data with uniform structure:
            {sat_name: {var_name: array(n_trajectories, n_points)}, ...}
        method : str
            Computation method ("fourier" or "incremental")
        
        Returns:
        -------
        dict : Computed term with satellite structure
               {sat_name: array(n_trajectories, n_points)}
        """
        
        if term_name not in TERMS:
            raise ValueError(f"Term '{term_name}' not found in TERMS")
        
        term_obj = TERMS[term_name]
        result = {}
        
        try:
            # Get abstract variables required by the term
            abstract_vars = term_obj.variables()
            
            # Compute for each satellite
            satellite_names = list(dic_quant.keys())
            
            for sat_name in satellite_names:
                # Extract data for this satellite
                dic_quant_sat = dic_quant[sat_name]
                
                # Extract satellite-specific parameters
                dic_param_sat = self._extract_sat_parameters(sat_name)
                
                # Convert abstract variables to concrete components for this satellite
                args_sat = self._get_concrete_variables(abstract_vars, dic_quant_sat)
                
                # Compute term for this satellite
                if method == "fourier":
                    result_sat = term_obj.calc_fourier(*args_sat, dic_param=dic_param_sat, traj=True)
                elif method == "incremental":
                    num_trajs = len(self.traj_param['trajectories_list'])
                    length_traj = len(self.traj_param['trajectories_list'][0])
                    args_array = np.array(args_sat)
                    result_sat = term_obj.calc_incremental_trajectories(args_array, num_trajs, length_traj)
                else:
                    raise ValueError(f"Unsupported method: {method}")

                if type(result_sat) != np.ndarray:
                    result_sat = np.asarray(result_sat)
                
                result[sat_name] = result_sat
            
        except Exception as e:
            if self.verbose:
                logger.error(f"Failed to compute {term_name}: {e}")
            raise
        
        return result
    
    def compute_all_terms_for_laws(self, dic_quantities: dict = None, laws: list = None, method: str = None):
        """
        Compute all terms required for the given laws.
        
        Determines the set of terms needed from law specifications, then computes
        them from the provided quantities. Accumulated per-satellite for uniform structure.
        
        Parameters:
        -----------
        dic_quantities : dict
            Dictionary of computed quantities with uniform structure:
            {sat_name: {var_name: array(n_trajectories, n_points)}, ...}
        laws : list[str]
            List of law names to compute terms for
        method : str
            Computation method ("fourier" or "incremental")
        
        Returns:
        -------
        dict : Dictionary of computed terms with uniform structure:
               {sat_name: {term_name: array(n_trajectories, n_points)}, ...}
        """
        
        if laws is None:
            laws = []
        
        # Get required terms
        required_terms = self.list_required_terms(laws)
        
        if self.verbose:
            logging.info("\n" + "-"*70)
            logging.info("FLUX AND SOURCE TERMS COMPUTATION")
            logging.info(f"  Computing {len(required_terms)} terms for {len(laws)} law(s)")
            logging.info(f"  Structure: {{sat_name: {{term_name: array(n_trajectories, n_points)}}}}")
        
        # Initialize result structure with satellite names
        satellite_names = list(dic_quantities.keys())
        result = {sat_name: {} for sat_name in satellite_names}
        
        for term_name in required_terms:
            try:
                computed = self.compute_term_from_TERMS(term_name, dic_quantities, method=method)
                
                # Store in uniform structure {sat_name: {term_name: array}}
                for sat_name in satellite_names:
                    result[sat_name][term_name] = computed[sat_name]
            except Exception as e:
                if self.verbose:
                    logger.error(f"Failed to compute {term_name}: {str(e)}")
        
        if self.verbose:
            logging.info(f"  [OK] All {len(required_terms)} terms computed successfully:")
            logging.info(required_terms)

        return result
    
    def terms_to_h5(self, result_terms: dict, filename: str = "terms_trajectory.h5"):
        """
        Save computed terms to an HDF5 file.
        
        Expects uniform structure: {sat_name: {term_name: array(n_trajectories, n_points)}}
        
        Parameters:
        -----------
        result_terms : dict
            Dictionary of computed terms with structure:
            {sat_name: {term_name: array(...)}, ...}
        filename : str
            Output filename for the HDF5 file
        """
        
        with h5py.File(filename, 'w') as f:
            for sat_name, terms_dict in result_terms.items():
                # Create a group for each satellite
                sat_group = f.create_group(sat_name)
                for term_name, term_value in terms_dict.items():
                    sat_group.create_dataset(term_name, data=term_value)
    
    # ========== PRIVATE METHODS ==========
    
    def _prepare_dic_param_for_terms_and_coeffs(self, dic_param: dict):
        """
        Prepare dic_param for terms_and_coeffs() by converting list-based parameters to scalars.
        
        Parameters:
        -----------
        dic_param : dict
            Dictionary with potentially list-based parameters
        
        Returns:
        -------
        dict : Cleaned dictionary with scalar parameters
        """
        params_clean = {}
        
        for key, value in dic_param.items():
            if isinstance(value, list):
                # Extract first element from list (same parameter for all trajectories)
                params_clean[key] = value[0]
            elif isinstance(value, dict):
                # For nbsatellite=4, extract first satellite and first trajectory
                if 'sat_0' in value:
                    first_sat_value = value['sat_0']
                    if isinstance(first_sat_value, list):
                        params_clean[key] = first_sat_value[0]
                    else:
                        params_clean[key] = first_sat_value
                else:
                    params_clean[key] = value
            else:
                params_clean[key] = value
        
        return params_clean
    
    def _extract_sat_parameters(self, sat_name: str):
        """
        Extract satellite-specific physical parameters.
        
        Parameters:
        -----------
        sat_name : str
            Satellite name (e.g., 'sat_0')
        
        Returns:
        -------
        dict : Satellite-specific parameters
        """
        dic_param_sat = {}
        for key, value in self.physical_param.items():
            if isinstance(value, dict) and sat_name in value:
                dic_param_sat[key] = value[sat_name]
            elif isinstance(value, list):
                dic_param_sat[key] = value[0]  # Use first element
            else:
                dic_param_sat[key] = value
        
        return dic_param_sat
    
    def _get_concrete_variables(self, abstract_vars: list, dic_quant: dict):
        """
        Convert abstract variables to concrete components.
        
        Parameters:
        -----------
        abstract_vars : list[str]
            List of abstract variables (ex: ['v', 'b'])
        dic_quant : dict
            Dictionary containing the data for one satellite
        
        Returns:
        -------
        list : List of np.ndarray corresponding to concrete variables
        """
        concrete_data = []
        for var in abstract_vars:
            components = self.VARIABLE_COMPONENTS.get(var, [var])  # Default to var itself
            for comp in components:
                if comp not in dic_quant:
                    raise ValueError(f"Component '{comp}' (from '{var}') not found")
                concrete_data.append(dic_quant[comp])
        return concrete_data


# ========== BACKWARD COMPATIBILITY FUNCTIONS ==========

def compute_all_terms_for_laws(dic_quantities: dict = None, traj_param: dict = None, physical_param: dict = None, laws: list = None, method: str = None, verbose: bool = False):
    """
    Backward compatibility wrapper for compute_all_terms_for_laws.
    
    Deprecated: Use TrajectoryTermsComputer.compute_all_terms_for_laws instead.
    """
    computer = TrajectoryTermsComputer(verbose=verbose, 
                                      physical_param=physical_param, 
                                      traj_param=traj_param)
    return computer.compute_all_terms_for_laws(dic_quantities, laws, method)

def terms_to_h5(result_terms: dict, filename: str = "terms_trajectory.h5"):
    """
    Backward compatibility wrapper for terms_to_h5.
    
    Deprecated: Use TrajectoryTermsComputer.terms_to_h5 instead.
    """
    computer = TrajectoryTermsComputer()
    computer.terms_to_h5(result_terms, filename)