import h5py
import logging

logger = logging.getLogger(__name__)


class LawsIOMixIn:

    def laws_to_h5(self, dic_law_terms, dic_coefficients, filename: str = "laws_terms.h5"):
        with h5py.File(filename, 'w') as f:
            law_terms_group = f.create_group('law_terms')
            for sat_name, terms_dict in dic_law_terms.items():
                sat_group = law_terms_group.create_group(sat_name)
                for term_key, value in terms_dict.items():
                    sat_group.create_dataset(term_key, data=value, compression="gzip", compression_opts=9)

            coeffs_group = f.create_group('coefficients')
            for coeff_key, coeff_value in dic_coefficients.items():
                coeffs_group.create_dataset(coeff_key, data=coeff_value)

        logging.info(f"  [OK] Saved law terms for {len(dic_law_terms)} satellite(s) to {filename}")
