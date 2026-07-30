import h5py
import logging

logger = logging.getLogger(__name__)


class TermsIOMixIn:

    def terms_to_h5(self, result_terms: dict, filename: str = "terms_trajectory.h5"):
        with h5py.File(filename, 'w') as f:
            for sat_name, terms_dict in result_terms.items():
                sat_group = f.create_group(sat_name)
                for term_name, term_value in terms_dict.items():
                    sat_group.create_dataset(term_name, data=term_value,
                        compression='gzip', compression_opts=4)
