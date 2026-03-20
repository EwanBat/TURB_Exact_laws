from typing import List

from .abstract_law import AbstractLaw

class CorEtot(AbstractLaw):
    def __init__(self):
        self.terms = ["cor_rvv","cor_ru","cor_rbb","source_dpan"]
        pass
    
    def terms_and_coeffs(self,physical_params):
        coeffs = {} 
        coeffs["cor_rvv"] = 1
        coeffs["cor_ru"] = 1
        coeffs["cor_rbb"] = 1
        return self.terms, coeffs

    def variables(self) -> List[str]:
        return self.list_variables(self.terms)

def load():
    return CorEtot()

