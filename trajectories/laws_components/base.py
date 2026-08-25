"""
Base class for trajectory laws computation.
Contains logger, initialization, parameter helper, and I/O.
"""
import logging
import h5py

logger = logging.getLogger('trajectories.laws_components')


class TrajectoryLawsComputerBase:
    """
    Compute law terms with coefficients along trajectories.

    Applies law coefficients to computed terms, handles divergence calculations,
    and manages both single-satellite and 4-satellite configurations.
    All data maintains structure: {sat_name: {term_name: array(n_traj, n_pts)}}
    """

    def __init__(self, verbose: bool = False, physical_param: dict = None, traj_param: dict = None, grid_param: dict = None):
        """
        Initialize the trajectory laws computer.

        Parameters:
        -----------
        verbose : bool
            Enable detailed logging
        physical_param : dict, optional
            Physical parameters (can be set later)
        traj_param : dict, optional
            Trajectory parameters (can be set later)
        grid_param : dict, optional
            Grid parameters (can be set later)
        """
        self.verbose = verbose
        self.physical_param = physical_param or {}
        self.traj_param = traj_param or {}
        self.grid_param = grid_param or {}
        self.nbsatellite = self.traj_param.get('nbsatellite', 1)

    def _prepare_dic_param_for_terms_and_coeffs(self, dic_param: dict):
        """
        Extract scalar values from parameter dictionary for law.terms_and_coeffs().

        The law computation expects scalar parameters, but dic_param may contain
        arrays or dictionaries (one value per trajectory or per satellite).
        Extract the first value uniformly for all trajectories.

        Parameters:
        -----------
        dic_param : dict
            Physical parameters (potentially list or dict values)

        Returns:
        -------
        dict : Cleaned dictionary with scalar values
        """
        params_clean = {}

        for key, value in dic_param.items():
            if isinstance(value, list):
                params_clean[key] = value[0]
            elif isinstance(value, dict):
                # For nbsatellite=4: extract first satellite and first trajectory
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

    def laws_to_h5(self, dic_law_terms, dic_coefficients, filename: str = "laws_terms.h5"):
        """
        Save law terms and coefficients to HDF5 file.

        Structure:
            /law_terms/sat_0/term_key -> array(n_traj, n_pts)
            /law_terms/sat_1/term_key -> array(n_traj, n_pts)
            /coefficients/law_term_key -> scalar value

        Parameters:
        -----------
        dic_law_terms : dict
            {sat_name: {term_key: array(n_traj, n_pts)}}
        dic_coefficients : dict
            {law_term_key: coefficient_value}
        filename : str
            Output HDF5 file path
        """
        with h5py.File(filename, 'w') as f:
            # Save law terms with satellite groups
            law_terms_group = f.create_group('law_terms')
            for sat_name, terms_dict in dic_law_terms.items():
                sat_group = law_terms_group.create_group(sat_name)
                for term_key, value in terms_dict.items():
                    sat_group.create_dataset(term_key, data=value, compression="gzip", compression_opts=9)

            # Save coefficients
            coeffs_group = f.create_group('coefficients')
            for coeff_key, coeff_value in dic_coefficients.items():
                coeffs_group.create_dataset(coeff_key, data=coeff_value)

        logging.info(f"  [OK] Saved law terms for {len(dic_law_terms)} satellite(s) to {filename}")