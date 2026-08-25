"""
Coefficient application mixin for trajectory laws.
Contains the public entry point and the per-satellite-count coefficient methods.
"""
import logging

from exact_laws.el_calc_mod.laws import LAWS
from trajectories.derivation_satellite import divergence_1satellite, divergence_4satellite, divergence_9satellite
from .base import logger


class TrajectoryLawsCoefficientsMixin:
    """
    Mixin providing law coefficient application methods.
    Expects self with: verbose, nbsatellite, physical_param, traj_param,
    grid_param, _prepare_dic_param_for_terms_and_coeffs(), laws_to_h5()
    """

    # ========== PUBLIC METHODS ==========

    def compute_laws_terms(self, dic_terms: dict, laws=None, filename="laws_terms.h5", method: str = None):
        """
        Compute law terms with coefficients for all satellites.

        Processes each law by extracting its term requirements and coefficients,
        then applying coefficients to available computed terms. Divergence terms
        are computed on-demand using satellite geometry.

        Parameters:
        -----------
        dic_terms : dict
            {sat_name: {term_name: array(n_trajectories, n_points)}}
            Example: {sat_0: {v: array(...), b: array(...), ...}}
        laws : list[str]
            Law names to process (e.g., ['PP98', 'BG17'])
        filename : str
            Output HDF5 file path
        method : str, optional
            Computation method ('incremental' or 'fourier')

        Returns:
        -------
        tuple : (dic_law_terms, dic_coefficients)
                - dic_law_terms: {sat_name: {term_coeff_key: array(n_traj, n_pts)}}
                - dic_coefficients: {law_term_key: coefficient_value}
        """

        if self.verbose:
            logging.info("\n" + "="*70)
            logging.info("COMPUTING LAW TERMS WITH COEFFICIENTS")
            logging.info(f"  Nbsatellite:  {self.nbsatellite}")

        if laws is None:
            laws = []

        dic_law_terms = {'sat_'+str(i): {} for i in range(self.nbsatellite)}
        dic_coefficients = {}

        for law_name in laws:
            if law_name not in LAWS:
                if self.verbose:
                    logger.warning(f"Law '{law_name}' not found")
                continue

            if self.verbose:
                logging.info(f"Processing law: {law_name}")

            try:
                law_obj = LAWS[law_name]

                if self.nbsatellite == 1:
                    law_terms, law_coeffs = self._apply_law_coefficients_1satellite(
                        dic_terms['sat_0'], law_obj
                    )
                elif self.nbsatellite == 4:
                    law_terms, law_coeffs = self._apply_law_coefficients_4satellite(
                        dic_terms, law_obj, method=method
                    )
                elif self.nbsatellite == 9:
                    law_terms, law_coeffs = self._apply_law_coefficients_9satellite(
                        dic_terms, law_obj, method=method
                    )

                dic_law_terms['sat_0'].update(law_terms)

                for term_key, coeff_value in law_coeffs.items():
                    dic_coefficients[f"{law_name}_{term_key}"] = coeff_value

                if self.verbose:
                    logging.info(f"  [OK] Terms computed for {len(list(dic_terms.keys()))} satellite(s)")
                    logging.info(f"    Applied terms: {list(law_terms.keys())}")

            except Exception as e:
                logger.error(f"Failed to process {law_name}: {e}")

        self.laws_to_h5(dic_law_terms, dic_coefficients, filename=filename)
        return dic_law_terms, dic_coefficients

    # ========== PRIVATE METHODS ==========

    def _apply_law_coefficients_1satellite(self, dic_terms_sat: dict, law_obj):
        """
        Apply law coefficients to computed terms for a single satellite.

        Processes three types of term coefficients:
        1. Divergence terms (div_*): compute via divergence_1satellite()
        2. Source terms (source_*): use directly from dic_terms_sat
        3. Simple terms (other): use directly from dic_terms_sat

        Parameters:
        -----------
        dic_terms_sat : dict
            {term_name: array(n_trajectories, n_points)}
        law_obj : AbstractLaw
            Has terms_and_coeffs() method

        Returns:
        -------
        tuple : (result_dict, coefficients_dict)
        """

        params_clean = self._prepare_dic_param_for_terms_and_coeffs(self.physical_param)
        law_terms, coeffs = law_obj.terms_and_coeffs(params_clean)
        result = {}

        # Partition coefficients by type: divergence, source, or simple terms
        # This categorization happens once; terms are then matched with data
        div_coeffs = {k: v for k, v in coeffs.items() if k.startswith('div_')}
        source_coeffs = {k: v for k, v in coeffs.items() if k.startswith('source_')}
        simple_coeffs = {k: v for k, v in coeffs.items()
                         if not k.startswith(('div_', 'source_'))}

        incomputable_terms = []

        for coeff_key, coeff_value in div_coeffs.items():
            term_name = coeff_key.replace('div_', '')
            if term_name in dic_terms_sat:
                try:
                    result[coeff_key] = divergence_1satellite(dic_terms_sat[term_name], self.traj_param)
                    if self.verbose:
                        logger.info(f"  [OK] Divergence {coeff_key} computed for sat_0")
                except Exception as e:
                    incomputable_terms.append((coeff_key, f"divergence failed: {e}"))
                    if self.verbose:
                        logger.error(f"  [ERROR] Divergence {coeff_key} failed: {e}")
            else:
                incomputable_terms.append((coeff_key, f"term '{term_name}' not in dic_terms_sat"))

        for coeff_key, coeff_value in source_coeffs.items():
            if coeff_key in dic_terms_sat:
                result[coeff_key] = dic_terms_sat[coeff_key]
                if self.verbose:
                    logger.info(f"  [OK] Source term {coeff_key} used directly")
            else:
                incomputable_terms.append((coeff_key, f"term '{coeff_key}' not in dic_terms_sat"))

        for coeff_key, coeff_value in simple_coeffs.items():
            if coeff_key in dic_terms_sat:
                result[coeff_key] = dic_terms_sat[coeff_key]
                if self.verbose:
                    logger.info(f"  [OK] Simple term {coeff_key} used directly")
            else:
                incomputable_terms.append((coeff_key, f"term '{coeff_key}' not in dic_terms_sat"))

        if incomputable_terms:
            for key, reason in incomputable_terms:
                logger.warning(f"  [WARNING] {key}: {reason}")

        return result, coeffs

    def _apply_law_coefficients_4satellite(self, dic_terms: dict, law_obj, method: str = None):
        """
        Apply law coefficients to computed terms for four satellites.

        Similar to 1-satellite case but:
        - Divergence terms computed using all 4 satellites (satellite geometry)
        - Method parameter ('incremental' or 'fourier') controls divergence calculation
        - Results stored in sat_0 (gradient-based terms)

        Parameters:
        -----------
        dic_terms : dict
            {sat_0: {...}, sat_1: {...}, sat_2: {...}, sat_3: {...}}
        law_obj : AbstractLaw
            Has terms_and_coeffs() method
        method : str, optional
            'incremental' or 'fourier' for divergence calculation

        Returns:
        -------
        tuple : (result_dict, coefficients_dict)
        """
        params_clean = self._prepare_dic_param_for_terms_and_coeffs(self.physical_param)
        law_terms, coeffs_sat_0 = law_obj.terms_and_coeffs(params_clean)
        result = {}

        # Partition coefficients by type: divergence, source, or simple terms
        div_coeffs = {k: v for k, v in coeffs_sat_0.items() if k.startswith('div_')}
        source_coeffs = {k: v for k, v in coeffs_sat_0.items() if k.startswith('source_')}
        simple_coeffs = {k: v for k, v in coeffs_sat_0.items()
                         if not k.startswith(('div_', 'source_'))}

        incomputable_terms = []

        for coeff_key, coeff_value in div_coeffs.items():
            term_name = coeff_key.replace('div_', '')
            if term_name in dic_terms['sat_0']:
                try:
                    if method == "incremental":
                        result[coeff_key] = divergence_4satellite(dic_terms, term_name, self.traj_param)
                    elif method == "fourier":
                        result[coeff_key] = dic_terms['sat_0'][term_name]
                    if self.verbose:
                        logger.info(f"  [OK] Divergence {coeff_key} computed")
                except Exception as e:
                    incomputable_terms.append((coeff_key, f"divergence failed: {e}"))
                    if self.verbose:
                        logger.error(f"  [ERROR] Divergence {coeff_key} failed: {e}")
            else:
                incomputable_terms.append((coeff_key, f"term '{term_name}' not in dic_terms['sat_0']"))

        for coeff_key, coeff_value in source_coeffs.items():
            if coeff_key in dic_terms['sat_0']:
                result[coeff_key] = dic_terms['sat_0'][coeff_key]
                if self.verbose:
                    logger.info(f"  [OK] Source term {coeff_key} used directly")
            else:
                incomputable_terms.append((coeff_key, f"term '{coeff_key}' not in dic_terms['sat_0']"))

        for coeff_key, coeff_value in simple_coeffs.items():
            if coeff_key in dic_terms['sat_0']:
                result[coeff_key] = dic_terms['sat_0'][coeff_key]
                if self.verbose:
                    logger.info(f"  [OK] Simple term {coeff_key} used directly")
            else:
                incomputable_terms.append((coeff_key, f"term '{coeff_key}' not in dic_terms['sat_0']"))

        if incomputable_terms:
            for key, reason in incomputable_terms:
                logger.warning(f"  [WARNING] {key}: {reason}")

        return result, coeffs_sat_0

    def _apply_law_coefficients_9satellite(self, dic_terms: dict, law_obj, method: str = None):
        """
        Apply law coefficients to computed terms for 9-satellite cube configuration.

        For divergence terms: computed at second order precision using the
        axis-aligned satellite pairs (incremental method) or passed through sat_0
        (fourier). Source and simple terms used directly from sat_0.

        Parameters:
        -----------
        dic_terms : dict
            {sat_0: {...}, sat_1: {...}, ..., sat_8: {...}}
        law_obj : AbstractLaw
            Has terms_and_coeffs() method
        method : str, optional
            'incremental' or 'fourier' for divergence calculation

        Returns:
        -------
        tuple : (result_dict, coefficients_dict)
        """
        params_clean = self._prepare_dic_param_for_terms_and_coeffs(self.physical_param)
        law_terms, coeffs = law_obj.terms_and_coeffs(params_clean)
        result = {}

        div_coeffs = {k: v for k, v in coeffs.items() if k.startswith('div_')}
        source_coeffs = {k: v for k, v in coeffs.items() if k.startswith('source_')}
        simple_coeffs = {k: v for k, v in coeffs.items()
                         if not k.startswith(('div_', 'source_'))}

        incomputable_terms = []

        for coeff_key, coeff_value in div_coeffs.items():
            term_name = coeff_key.replace('div_', '')
            if term_name in dic_terms['sat_0']:
                try:
                    if method == "incremental":
                        result[coeff_key] = divergence_9satellite(dic_terms, term_name, self.traj_param, self.grid_param)
                        if self.verbose:
                            logger.info(f"  [OK] Divergence {coeff_key} computed at second order from satellite pairs")
                    elif method == "fourier":
                        result[coeff_key] = dic_terms['sat_0'][term_name]
                        if self.verbose:
                            logger.info(f"  [OK] Divergence {coeff_key} passed through (fourier)")
                except Exception as e:
                    incomputable_terms.append((coeff_key, f"divergence failed: {e}"))
                    if self.verbose:
                        logger.error(f"  [ERROR] Divergence {coeff_key} failed: {e}")
            else:
                incomputable_terms.append((coeff_key, f"term '{term_name}' not in dic_terms['sat_0']"))

        for coeff_key, coeff_value in source_coeffs.items():
            if coeff_key in dic_terms['sat_0']:
                result[coeff_key] = dic_terms['sat_0'][coeff_key]
                if self.verbose:
                    logger.info(f"  [OK] Source term {coeff_key} used directly")
            else:
                incomputable_terms.append((coeff_key, f"term '{coeff_key}' not in dic_terms['sat_0']"))

        for coeff_key, coeff_value in simple_coeffs.items():
            if coeff_key in dic_terms['sat_0']:
                result[coeff_key] = dic_terms['sat_0'][coeff_key]
                if self.verbose:
                    logger.info(f"  [OK] Simple term {coeff_key} used directly")
            else:
                incomputable_terms.append((coeff_key, f"term '{coeff_key}' not in dic_terms['sat_0']"))

        if incomputable_terms:
            for key, reason in incomputable_terms:
                logger.warning(f"  [WARNING] {key}: {reason}")

        return result, coeffs