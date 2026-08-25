"""
Computation mixin for trajectory quantities.
Contains the public entry point, quantity requirement listing, dispatch,
and the per-satellite-count computation methods.
"""
import logging

from exact_laws.el_calc_mod.laws import LAWS
from exact_laws.el_calc_mod.terms import TERMS
from .base import logger


class TrajectoryQuantitiesComputeMixin:
    """
    Mixin providing trajectory quantity computation methods.
    Expects self with: verbose, traj_param, nbsatellite, quantities_registry,
    QUANTITY_DEPENDENCIES, GRADIENT_QUANTITIES, _compute_quantity_vectorized(),
    quantities_to_h5()
    """

    # ========== PUBLIC METHODS ==========

    def extract_and_compute(self, dic_datas: dict,
                            laws=None, terms=None, quantities=None, method: str = None,
                            filename: str = None):
        """Compute all required quantities from laws/terms specifications.

        Parameters:
        -----------
        dic_datas : dict
            {sat_name: {var_name: array(n_trajectories, n_points)}}
        laws : list, optional
            Law names; required variables extracted via LAWS[name].variables()
        terms : list, optional
            Term names; required variables extracted via TERMS[name].variables()
        quantities : list, optional
            Explicit list of quantities to compute
        filename : str, optional
            If set, save results to HDF5

        Returns:
        -------
        dict : Same structure as input, with computed quantities added
        """
        if self.verbose:
            logger.info("\n" + "=" * 70)
            logger.info("TRAJECTORY QUANTITIES COMPUTATION (VECTORIZED)")
            logger.info(f"  Nbsatellite:   {self.nbsatellite}")
            logger.info(f"  Gap satellite: {self.traj_param.get('gap_satellite', 1)}")

        available_quantities = self._list_required_quantities(
            laws, terms, quantities, method=method
        )

        if self.verbose:
            logger.info(f"  Quantities to compute: {len(available_quantities)}")
            logger.info(f"  {available_quantities}")

        return self._compute_all_quantities(dic_datas, available_quantities, filename)

    def _list_required_quantities(self, laws=None, terms=None,
                                  quantities=None, method: str = None):
        """
        Collect all quantities required by the given laws and terms.

        Parameters:
        -----------
        laws : list[str], optional
            Law names; their variable requirements are fetched via LAWS[name].variables()
        terms : list[str], optional
            Term names; their variable requirements are fetched via TERMS[name].variables()
        quantities : list[str], optional
            Explicit list of additional quantities
        method : str, optional
            Computation method ('incremental' or 'fourier'), passed to .variables()

        Returns:
        -------
        list[str] : Unique set of required quantity names
        """
        quantities = list(quantities) if quantities else []

        if terms:
            for term_name in terms:
                if term_name in TERMS:
                    quantities.extend(
                        TERMS[term_name].variables(nbsatellite=self.nbsatellite, method=method)
                    )

        if laws:
            for law_name in laws:
                if law_name in LAWS:
                    quantities.extend(
                        LAWS[law_name].variables(nbsatellite=self.nbsatellite, method=method)
                    )

        return list(set(quantities))

    # ========== PRIVATE COMPUTATION METHODS ==========

    def _compute_all_quantities(self, dic_datas: dict, available_quantities: list, filename: str):
        """
        Dispatch to the correct computation path based on satellite count.

        For nbsatellite=1: single pass over all quantities.
        For nbsatellite=4: non-gradient quantities per satellite + gradient from 4-sat.
        For nbsatellite=9: non-gradient quantities per satellite + gradient from
        face-based tetrahedrons averaged over 24 sub-tuples.

        Parameters:
        -----------
        dic_datas : dict
            {sat_name: {var_name: array(n_trajectories, n_points)}}
        available_quantities : list[str]
            Quantity names to compute
        filename : str or None
            If set, save results to HDF5

        Returns:
        -------
        dict : {sat_name: {var_name: array(n_trajectories, n_points)}}
        """
        dic_quantities = {sat_name: {} for sat_name in dic_datas.keys()}
    
        all_quantity_names = list(set(
            [req for deps in self.QUANTITY_DEPENDENCIES.values() for req in deps["requires"] if deps in available_quantities]
            + [req for deps in self.GRADIENT_QUANTITIES.values() for req in deps["requires"] if deps in available_quantities]
        ))

        if self.nbsatellite == 1:
            self._compute_all_single_pass(dic_datas, dic_quantities, available_quantities)
            if self.verbose:
                missing = [q for q in all_quantity_names if q not in dic_quantities["sat_0"].keys()]
                if missing:

                    logger.error(f"  [ERROR] Missing quantities in sat_0 for nbsatellite=1: {missing}")

                else:
                    logger.info(f"  [OK] All quantities computed successfully for nbsatellite=1.")
                    logger.info(f"  dic_quantities['sat_0'].keys(): {list(dic_quantities['sat_0'].keys())}")
        elif self.nbsatellite == 4:
            self._compute_non_gradient_quantities(dic_datas, dic_quantities, available_quantities)
            self._merge_raw_data(dic_datas, dic_quantities)
            self._compute_gradient_4satellite(dic_quantities, available_quantities)
            if self.verbose:
                missing = [q for q in all_quantity_names if q not in dic_quantities["sat_0"].keys()]
                if missing:

                    logger.error(f"  [ERROR] Missing quantities in sat_0 for nbsatellite=4: {missing}")


                else:
                    logger.info(f"  [OK] All quantities computed successfully for nbsatellite=4.")
                    logger.info(f"  dic_quantities['sat_0'].keys(): {list(dic_quantities['sat_0'].keys())}")
        elif self.nbsatellite == 9:
            self._compute_non_gradient_quantities(dic_datas, dic_quantities, available_quantities)
            self._merge_raw_data(dic_datas, dic_quantities)
            self._compute_gradient_9satellite(dic_quantities, available_quantities)
            if self.verbose:
                missing = [q for q in all_quantity_names if q not in dic_quantities["sat_0"].keys()]
                if missing:

                    logger.error(f"  [ERROR] Missing quantities in sat_0 for nbsatellite=9: {missing}")

                else:
                    logger.info(f"  [OK] All quantities computed successfully for nbsatellite=9.")
                    logger.info(f"  dic_quantities['sat_1'].keys(): {list(dic_quantities['sat_1'].keys())}")

        if filename:
            self.quantities_to_h5(dic_quantities, filename)

        return dic_quantities

    def _compute_all_single_pass(self, dic_datas: dict, dic_quantities: dict, quantities: list):
        """
        Compute all quantities in one pass (nbsatellite=1).

        Iterates over all required quantities and computes them via the
        QUANTITIES registry, storing results into dic_quantities.

        Parameters:
        -----------
        dic_datas : dict
            Raw input data {sat_name: {var_name: array}}
        dic_quantities : dict
            Output dict to populate {sat_name: {var_name: array}}
        quantities : list[str]
            Quantity names to compute
        """
        for quantity_name in quantities:
            try:
                for sat_name in dic_datas.keys():
                    result = self._compute_quantity_vectorized(quantity_name, dic_datas[sat_name])
                    if isinstance(result, dict):
                        dic_quantities[sat_name].update(result)
                    else:
                        dic_quantities[sat_name][quantity_name] = result
            except Exception as e:
                if self.verbose:
                    logger.error(f"Failed to compute {quantity_name}: {e}")

    def _compute_non_gradient_quantities(self, dic_datas: dict, dic_quantities: dict, quantities: list):
        """
        Compute non-gradient quantities independently for each satellite.

        Skips quantities listed in GRADIENT_QUANTITIES; computes all others
        per satellite using _compute_quantity_vectorized.

        Parameters:
        -----------
        dic_datas : dict
            Raw input data {sat_name: {var_name: array}}
        dic_quantities : dict
            Output dict to populate {sat_name: {var_name: array}}
        quantities : list[str]
            Quantity names to compute
        """
        for quantity_name in quantities:
            if quantity_name in self.GRADIENT_QUANTITIES:
                continue
            try:
                for sat_name in dic_datas.keys():
                    result = self._compute_quantity_vectorized(quantity_name, dic_datas[sat_name])
                    if isinstance(result, dict):
                        dic_quantities[sat_name].update(result)
                    else:
                        dic_quantities[sat_name][quantity_name] = result
            except Exception as e:
                if self.verbose:
                    logger.error(f"Failed to compute {quantity_name}: {e}")

    def _merge_raw_data(self, dic_datas: dict, dic_quantities: dict):
        """
        Copy raw input fields into dic_quantities so gradients can reference them.

        For each satellite, copies any key from dic_datas that is not already
        present in dic_quantities.

        Parameters:
        -----------
        dic_datas : dict
            Raw input data {sat_name: {var_name: array}}
        dic_quantities : dict
            Output dict to populate {sat_name: {var_name: array}}
        """
        for sat_name in dic_datas.keys():
            for key, value in dic_datas[sat_name].items():
                if key not in dic_quantities[sat_name]:
                    dic_quantities[sat_name][key] = value

    def _compute_gradient_4satellite(self, dic_quantities: dict, quantities: list):
        """
        Compute gradient/divergence quantities using all 4 satellites.

        For each quantity in GRADIENT_QUANTITIES, computes it via the
        QUANTITIES registry using the multi-satellite dic_quantities,
        storing results at sat_0.

        Parameters:
        -----------
        dic_quantities : dict
            {sat_name: {var_name: array}} with data for all 4 satellites
        quantities : list[str]
            Quantity names to compute (only GRADIENT_QUANTITIES entries are processed)
        """
        for quantity_name in quantities:
            if quantity_name not in self.GRADIENT_QUANTITIES:
                continue
            try:
                result = self._compute_quantity_vectorized(quantity_name, dic_quantities)
                if isinstance(result, dict):
                    dic_quantities['sat_0'].update(result)
                else:
                    dic_quantities['sat_0'][quantity_name] = result
            except Exception as e:
                if self.verbose:
                    logger.error(f"Failed to compute {quantity_name}: {e}")

    def _compute_gradient_9satellite(self, dic_quantities: dict, quantities: list):
        """
        Compute gradient/divergence quantities using all 9 satellites.

        Delegates to the gradient quantity itself (which for the 9-satellite
        cluster computes a second-order central first derivative along each
        axis from the opposite-pair satellites) and stores the result at sat_0.

        Parameters:
        -----------
        dic_quantities : dict
            {sat_name: {var_name: array}} with data for all 9 satellites
        quantities : list[str]
            Quantity names to compute (only GRADIENT_QUANTITIES entries are processed)
        """
        for quantity_name in quantities:
            if quantity_name not in self.GRADIENT_QUANTITIES:
                continue
            try:
                result = self._compute_quantity_vectorized(quantity_name, dic_quantities)
                if isinstance(result, dict):
                    dic_quantities['sat_0'].update(result)
                else:
                    dic_quantities['sat_0'][quantity_name] = result
            except Exception as e:
                if self.verbose:
                    logger.error(f"Failed to compute {quantity_name}: {e}")