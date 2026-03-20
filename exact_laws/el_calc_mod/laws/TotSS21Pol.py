from typing import List

from .abstract_law import AbstractLaw


class TotSs21Pol(AbstractLaw):
    def __init__(self):
        self.terms = [
            "flux_ss21pol",
            "flux_ss21hybpol",
            "source_ss21pol",
        ]
        pass

    def terms_and_coeffs(self, physical_params):
        coeffs = {}
        coeffs["div_flux_ss21pol"] = - 1 / 4
        coeffs["div_flux_ss21hybpol"] = - 1 / 4
        coeffs["source_ss21pol"] = - 1 / 4
        return self.terms, coeffs

    def variables(self) -> List[str]:
        return self.list_variables(self.terms)


def load():
    return TotSs21Pol()
