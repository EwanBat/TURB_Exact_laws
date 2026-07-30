import numpy as np
import logging

from trajectories.derivation_satellite import divergence_1satellite, divergence_4satellite

logger = logging.getLogger(__name__)


class LawCoefficientsMixIn:

    def _apply_law_coefficients_1satellite(self, dic_terms_sat: dict, law_obj):
        law_terms, coeffs, div_coeffs, source_coeffs, simple_coeffs = \
            self._partition_coefficients(law_obj)
        result = {}
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
        law_terms, coeffs, div_coeffs, source_coeffs, simple_coeffs = \
            self._partition_coefficients(law_obj)
        result = {}
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

        return result, coeffs

    def _apply_law_coefficients_9satellite(self, dic_terms: dict, law_obj, method: str = None):
        law_terms, coeffs, div_coeffs, source_coeffs, simple_coeffs = \
            self._partition_coefficients(law_obj)
        result = {}
        incomputable_terms = []

        for coeff_key, coeff_value in div_coeffs.items():
            term_name = coeff_key.replace('div_', '')
            if term_name in dic_terms['sat_0']:
                try:
                    if method == "incremental":
                        tuples = self._get_9satellite_tuples_with_sat0()
                        offsets = self.traj_param['satellite_offsets']
                        div_results = []
                        for (i, j, k) in tuples:
                            dic_quant_sub = {
                                'sat_0': dic_terms['sat_0'],
                                'sat_1': dic_terms[f'sat_{i}'],
                                'sat_2': dic_terms[f'sat_{j}'],
                                'sat_3': dic_terms[f'sat_{k}'],
                            }
                            traj_param_sub = dict(self.traj_param)
                            traj_param_sub['dR1'] = offsets[f'sat_{i}']
                            traj_param_sub['dR2'] = offsets[f'sat_{j}']
                            traj_param_sub['dR3'] = offsets[f'sat_{k}']
                            div_results.append(divergence_4satellite(
                                dic_quant_sub, term_name, traj_param_sub
                            ))
                        result[coeff_key] = np.mean(div_results, axis=0)
                        if self.verbose:
                            logger.info(f"  [OK] Divergence {coeff_key} averaged over {len(div_results)} tetrahedrons")
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
