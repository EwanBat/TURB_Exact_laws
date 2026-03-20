from typing import Dict, List, Tuple

from ..terms import TERMS

class AbstractLaw:
    def __init__(self):
        pass
    
    def terms_and_coeffs(self, *args, **kwargs) -> Tuple[List[str], Dict[str,float]]:
        raise NotImplementedError("You have to reimplement this method")
    
    def list_variables(self,terms):
        variables = []
        for term in terms:
            variables += TERMS[term].variables()
        return list(set(variables))

    def variables(self) -> List[str]:
        raise NotImplementedError("You have to reimplement this method") 