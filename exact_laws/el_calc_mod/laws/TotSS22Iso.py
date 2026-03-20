from typing import List

from .abstract_law import AbstractLaw


class TotSs22Iso(AbstractLaw):
    def __init__(self):
        self.terms = [
            "flux_ss22iso",
            "source_ss22iso",
        ]
        pass

    def terms_and_coeffs(self, physical_params):
        coeffs = {}
        coeffs["div_flux_ss22iso"] = - 1 / 4
        coeffs["source_ss22iso"] = - 1 / 4
        return self.terms, coeffs

    def variables(self) -> List[str]:
        return self.list_variables(self.terms)


def load():
    return TotSs22Iso()
