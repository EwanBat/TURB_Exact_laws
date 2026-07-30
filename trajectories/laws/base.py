import logging

from exact_laws.el_calc_mod.laws import LAWS

logger = logging.getLogger(__name__)


class TrajectoryLawsBase:

    def __init__(self, verbose: bool = False, physical_param: dict = None,
                 traj_param: dict = None):
        self.verbose = verbose
        self.physical_param = physical_param or {}
        self.traj_param = traj_param or {}
        self.nbsatellite = self.traj_param.get('nbsatellite', 1)

    def compute_laws_terms(self, dic_terms: dict, laws=None,
                           filename="laws_terms.h5", method: str = None):
        if self.verbose:
            logging.info("\n" + "=" * 70)
            logging.info("COMPUTING LAW TERMS WITH COEFFICIENTS")
            logging.info(f"  Nbsatellite:  {self.nbsatellite}")

        if laws is None:
            laws = []

        dic_law_terms = {'sat_' + str(i): {} for i in range(self.nbsatellite)}
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

    def _partition_coefficients(self, law_obj):
        params_clean = self._prepare_dic_param_for_terms_and_coeffs(self.physical_param)
        law_terms, coeffs = law_obj.terms_and_coeffs(params_clean)

        div_coeffs = {k: v for k, v in coeffs.items() if k.startswith('div_')}
        source_coeffs = {k: v for k, v in coeffs.items() if k.startswith('source_')}
        simple_coeffs = {k: v for k, v in coeffs.items()
                         if not k.startswith(('div_', 'source_'))}

        return law_terms, coeffs, div_coeffs, source_coeffs, simple_coeffs

    def _get_9satellite_tuples_with_sat0(self):
        satellite_offsets = self.traj_param.get('satellite_offsets', {})
        if not satellite_offsets:
            raise ValueError("Missing satellite_offsets in traj_param for nbsatellite=9")

        faces = [
            [1, 2, 5, 6],
            [3, 4, 7, 8],
            [1, 3, 5, 7],
            [2, 4, 6, 8],
            [1, 2, 3, 4],
            [5, 6, 7, 8],
        ]

        valid_tuples = []
        for surface in faces:
            for i in range(4):
                t = tuple(surface[:i] + surface[i+1:])
                valid_tuples.append(t)
        return valid_tuples

    def _prepare_dic_param_for_terms_and_coeffs(self, dic_param: dict):
        params_clean = {}

        for key, value in dic_param.items():
            if isinstance(value, list):
                params_clean[key] = value[0]
            elif isinstance(value, dict):
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
