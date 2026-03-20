from typing import List

from .abstract_law import AbstractLaw


class TotSs22Pol(AbstractLaw):
    def __init__(self):
        self.terms = [
            "flux_ss22pol",
            "source_ss22pol",
        ]
        pass

    def terms_and_coeffs(self, physical_params):
        coeffs = {}
        coeffs["div_flux_ss22pol"] = - 1 / 4
        coeffs["source_ss22pol"] = - 1 / 4
        return self.terms, coeffs

    def variables(self) -> List[str]:
        return self.list_variables(self.terms)


def load():
    return TotSs22Pol()
