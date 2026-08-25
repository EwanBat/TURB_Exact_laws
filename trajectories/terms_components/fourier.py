"""
Fourier computation mixin for trajectory terms.
Contains methods for Fourier term computation for 1 and multi-satellite.
"""
import numpy as np
import logging
from exact_laws.el_calc_mod.terms import TERMS
from .base import logger


class TrajectoryTermsFourierMixin:
    """
    Mixin providing Fourier term computation methods.
    Expects self with: _sat_param_cache, verbose, FLUX_TERMS, SOURCE_TERMS
    """

    def _compute_terms_fourier_1sat(self, dic_quantities, required_terms):
        """
        Compute terms using Fourier method for single satellite.

        Parameters:
        -----------
        dic_quantities : dict
            {sat_name: {var_name: array(n_trajectories, n_points)}}
        required_terms : list[str]
            Names of terms to compute

        Returns:
        -------
        dict : {'sat_0': {term_name: array(n_trajectories, n_points)}}
        """
        result = {'sat_0': {}}
        computed = []
        missing = []

        for term_name in required_terms:
            try:
                term_obj = TERMS[term_name]
                result['sat_0'][term_name] = term_obj.calc_fourier(
                    **dic_quantities['sat_0'], dic_param=self._sat_param_cache, traj=True)
                if not isinstance(result['sat_0'][term_name], np.ndarray):
                    result['sat_0'][term_name] = np.asarray(result['sat_0'][term_name])
                computed.append(term_name)
            except Exception as e:
                missing.append(term_name)
                if self.verbose:
                    logger.error(f"  [ERROR] Failed to compute term {term_name}: {e}")

        if self.verbose:
            for t in computed:
                logger.info(f"  [OK] Term {t} computed for sat_0")
            for t in missing:
                logger.warning(f"  [WARNING] Term {t} NOT computed for sat_0")

        return result

    def _compute_terms_fourier_multi(self, dic_quantities, required_terms):
        """
        Compute terms using Fourier method for multi-satellite configuration (4 or 9).

        Flux terms use calc_with_fourier_4sat, source terms use calc_fourier.

        Parameters:
        -----------
        dic_quantities : dict
            {sat_name: {var_name: array(n_trajectories, n_points)}}
        required_terms : list[str]
            Names of terms to compute

        Returns:
        -------
        dict : {'sat_0': {term_name: array(n_trajectories, n_points)}}
        """
        result = {'sat_0': {}}
        computed = []
        missing = []
        for term_name in required_terms:
            try:
                term_obj = TERMS[term_name]
                if term_name in self.FLUX_TERMS:
                    result['sat_0'][term_name] = term_obj.calc_with_fourier_4sat(
                        **dic_quantities['sat_0'], dic_param=self._sat_param_cache, traj=True)
                elif term_name in self.SOURCE_TERMS:
                    result['sat_0'][term_name] = term_obj.calc_fourier(
                        **dic_quantities['sat_0'], dic_param=self._sat_param_cache, traj=True)
                else:
                    missing.append(term_name)
                    continue
                if not isinstance(result['sat_0'][term_name], np.ndarray):
                    result['sat_0'][term_name] = np.asarray(result['sat_0'][term_name])
                computed.append(term_name)
            except Exception as e:
                missing.append(term_name)
                if self.verbose:
                    logger.error(f"  [ERROR] Failed to compute term {term_name}: {e}")

        if self.verbose:
            for t in computed:
                logger.info(f"  [OK] Term {t} computed for sat_0")
            for t in missing:
                logger.warning(f"  [WARNING] Term {t} NOT computed for sat_0")

        return result
