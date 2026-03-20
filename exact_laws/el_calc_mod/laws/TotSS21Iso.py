from typing import List

from .abstract_law import AbstractLaw


class TotSs21Iso(AbstractLaw):
    def __init__(self):
        self.terms = [
            "flux_ss21iso",
            "flux_ss21hybiso",
            "source_ss21iso",
        ]
        pass

    def terms_and_coeffs(self, physical_params):
        coeffs = {}
        coeffs["div_flux_ss21iso"] = - 1 / 4
        coeffs["div_flux_ss21hybiso"] = - 1 / 4
        coeffs["source_ss21iso"] = - 1 / 4
        return self.terms, coeffs

    def variables(self) -> List[str]:
        return self.list_variables(self.terms)


def load():
    return TotSs21Iso()
