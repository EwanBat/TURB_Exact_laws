import numpy as np
from numba import set_num_threads
import logging
from exact_laws.el_calc_mod.terms import TERMS

logger = logging.getLogger(__name__)


class IncrementalTermsMixIn:

    def _compute_terms_incremental_1sat(self, dic_quantities, required_terms):
        result = {'sat_0': {}}
        fs = self._get_incremental_fs()
        n_points = self.traj_param["n_points"]
        n_trajectories = self.traj_param["n_trajectories"]
        filter_enabled = self.run_params.get('filter_enabled', False)
        computed = []
        missing = []

        merged_quantities = {}
        for quantity in dic_quantities['sat_0'].keys():
            merged_quantities[quantity] = np.concatenate((
                dic_quantities['sat_0'][quantity],
                dic_quantities['sat_0'][quantity]
            ), axis=1)

        for term_name in required_terms:
            try:
                term_obj = TERMS[term_name]
                if filter_enabled:
                    result['sat_0'][term_name] = term_obj.calc_filter(
                        n_points, n_trajectories, fs, **merged_quantities)
                else:
                    result['sat_0'][term_name] = term_obj.calc_incr_traj(
                        n_points, n_trajectories, **merged_quantities)
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

    def _compute_terms_incremental_4sat(self, dic_quantities, required_terms):
        result = {sat_name: {} for sat_name in self._sat_names}
        fs = self._get_incremental_fs()
        sat1 = 'sat_0'
        n_points = self.traj_param["n_points"]
        n_trajectories = self.traj_param["n_trajectories"]
        filter_enabled = self.run_params.get('filter_enabled', False)

        set_num_threads(self.run_params.get('max_workers', 1))

        flux_terms = [t for t in required_terms if t in self.FLUX_TERMS]
        source_terms = [t for t in required_terms if t in self.SOURCE_TERMS]

        for sat2 in self._sat_names:
            computed = []
            missing = []

            merged_quantities = {}
            for quantity in dic_quantities[sat1].keys():
                if quantity in dic_quantities[sat2]:
                    merged_quantities[quantity] = np.concatenate((
                        dic_quantities[sat1][quantity],
                        dic_quantities[sat2][quantity]
                    ), axis=1)

            for term_name in flux_terms:
                try:
                    term_obj = TERMS[term_name]
                    if filter_enabled:
                        result[sat2][term_name] = term_obj.calc_filter(
                            n_points, n_trajectories, fs, **merged_quantities)
                    else:
                        result[sat2][term_name] = term_obj.calc_incr_traj(
                            n_points, n_trajectories, **merged_quantities)
                    computed.append(term_name)
                except Exception as e:
                    missing.append(term_name)
                    if self.verbose:
                        logger.error(f"  [ERROR] Failed to compute term {term_name} for {sat2}: {e}")

            if sat2 == 'sat_0':
                for term_name in source_terms:
                    try:
                        term_obj = TERMS[term_name]
                        if filter_enabled:
                            result[sat2][term_name] = term_obj.calc_filter(
                                n_points, n_trajectories, fs, **merged_quantities)
                        else:
                            result[sat2][term_name] = term_obj.calc_incr_traj(
                                n_points, n_trajectories, **merged_quantities)
                        computed.append(term_name)
                    except Exception as e:
                        missing.append(term_name)
                        if self.verbose:
                            logger.error(f"  [ERROR] Failed to compute term {term_name} for {sat2}: {e}")

            if self.verbose:
                for t in computed:
                    logger.info(f"  [OK] Term {t} computed for {sat2}")
                for t in missing:
                    logger.warning(f"  [WARNING] Term {t} NOT computed for {sat2}")

        return result

    def _compute_terms_incremental_9sat(self, dic_quantities, required_terms):
        result = {sat_name: {} for sat_name in self._sat_names}
        fs = self._get_incremental_fs()
        sat1 = 'sat_0'
        n_points = self.traj_param["n_points"]
        n_trajectories = self.traj_param["n_trajectories"]
        filter_enabled = self.run_params.get('filter_enabled', False)

        set_num_threads(self.run_params.get('max_workers', 1))

        flux_terms = [t for t in required_terms if t in self.FLUX_TERMS]
        source_terms = [t for t in required_terms if t in self.SOURCE_TERMS]
        other_terms = [t for t in required_terms if t not in self.FLUX_TERMS and t not in self.SOURCE_TERMS]

        for sat2 in self._sat_names:
            computed = []
            missing = []

            merged_quantities = {}
            for quantity in dic_quantities[sat1].keys():
                if quantity in dic_quantities[sat2]:
                    merged_quantities[quantity] = np.concatenate((
                        dic_quantities[sat1][quantity],
                        dic_quantities[sat2][quantity]
                    ), axis=1)

            for term_name in flux_terms + other_terms:
                try:
                    term_obj = TERMS[term_name]
                    if filter_enabled:
                        result[sat2][term_name] = term_obj.calc_filter(
                            n_points, n_trajectories, fs, **merged_quantities)
                    else:
                        result[sat2][term_name] = term_obj.calc_incr_traj(
                            n_points, n_trajectories, **merged_quantities)
                    computed.append(term_name)
                except Exception as e:
                    missing.append(term_name)
                    if self.verbose:
                        logger.error(f"  [ERROR] Failed to compute term {term_name} for {sat2}: {e}")

            if self.verbose:
                for t in computed:
                    logger.info(f"  [OK] Term {t} computed for {sat2}")
                for t in missing:
                    logger.warning(f"  [WARNING] Term {t} NOT computed for {sat2}")

        if source_terms:
            merged_quantities = {}
            for quantity in dic_quantities['sat_0'].keys():
                merged_quantities[quantity] = np.concatenate((
                    dic_quantities['sat_0'][quantity],
                    dic_quantities['sat_0'][quantity]
                ), axis=1)

            for term_name in source_terms:
                try:
                    term_obj = TERMS[term_name]
                    if filter_enabled:
                        result['sat_0'][term_name] = term_obj.calc_filter(
                            n_points, n_trajectories, fs, **merged_quantities)
                    else:
                        result['sat_0'][term_name] = term_obj.calc_incr_traj(
                            n_points, n_trajectories, **merged_quantities)
                    if self.verbose:
                        logger.info(f"  [OK] Source term {term_name} computed from sat_0")
                except Exception as e:
                    if self.verbose:
                        logger.error(f"  [ERROR] Failed source term {term_name} for sat_0: {e}")

        return result
