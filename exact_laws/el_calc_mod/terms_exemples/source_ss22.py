from typing import List
from numba import njit
import sympy as sp

from .abstract_term import AbstractTerm, calc_source_with_numba

class SourceSs22(AbstractTerm):
    def __init__(self):
        self.set_sympy_expr()
        quantities = ("vx'", "vy'", "vz'", "vx", "vy", "vz", 
                      "bx'", "by'", "bz'", "bx", "by", "bz", 
                      "rho'", "rho", "u'", "u", "pm'", "pm", "p'", "p",
                     "divv'", "divb'", "dxrho'", "dyrho'", "dzrho'",
                )
        self.fct = sp.lambdify(
            sp.symbols(quantities),
            self.expr,
            "numpy",
        )
        
    def set_sympy_expr(self):
        vxP, vyP, vzP = sp.symbols(("vx'", "vy'", "vz'"))
        vxNP, vyNP, vzNP = sp.symbols(("vx", "vy", "vz"))
        divvP = sp.symbols(("divv'"))
        bxP, byP, bzP = sp.symbols(("bx'", "by'", "bz'"))
        bxNP, byNP, bzNP = sp.symbols(("bx", "by", "bz"))
        divbP = sp.symbols(("divb'"))
        rhoP, rhoNP = sp.symbols(("rho'", "rho"))
        pmP, pmNP = sp.symbols(("pm'", "pm"))
        pP, pNP = sp.symbols(("p'", "p"))
        uP, uNP = sp.symbols(("u'", "u"))
        dxrhoP, dyrhoP, dzrhoP = sp.symbols(("dxrho'", "dyrho'", "dzrho'"))
        
        self.expr = (divvP * (rhoNP * (vxNP * (vxP - vxNP) + vyNP * (vyP - vyNP) + vzNP * (vzP - vzNP)
                                       + 1/2 * (bxNP * (bxP - bxNP) + byNP * (byP - byNP) + bzNP * (bzP - bzNP))
                                       + 2 * (uP - uNP - pP + pNP))
                              - 1/2 * (bxNP * (rhoP * bxP - rhoNP * bxNP) + byNP * (rhoP * byP - rhoNP * byNP) + bzNP * (rhoP * bzP - rhoNP * bzNP)))
                     + divbP * ( rhoNP * (-2 *(vxNP * (bxP - bxNP) + vyNP * (byP - byNP) + vzNP * (bzP - bzNP))
                                          - (bxNP * (vxP - vxNP) + byNP * (vyP - vyNP) + bzNP * (vzP - vzNP)))
                                + (bxNP * (rhoP * vxP - rhoNP * vxNP) + byNP * (rhoP * vyP - rhoNP * vyNP) + bzNP * (rhoP * vzP - rhoNP * vzNP)))
                     + ((rhoP - rhoNP) * (pNP + pmNP) - rhoNP * (pP + pmP - pNP - pmNP)) * (vxNP * dxrhoP + vyNP * dyrhoP + vzNP * dzrhoP)/ rhoNP)
        
    
    def calc(self, vector:List[int], cube_size:List[int],vx, vy, vz, bx, by, bz, rho, pm, p, u, divv, divb, dxrho, dyrho, dzrho, **kwarg) -> List[float]:
        return calc_source_with_numba(calc_in_point_with_sympy, *vector, *cube_size, vx, vy, vz, bx, by, bz, rho, pm, p, u, divv, divb, dxrho, dyrho, dzrho)

    def variables(self) -> List[str]:
        return ['v', 'b', 'rho', 'pm', 'p', 'u', 'divv', 'divb', 'gradrho']
    
    def print_expr(self):
        sp.init_printing(use_latex=True)
        return self.expr

def load():
    return SourceSs22()

def print_expr():
    return SourceSs22().print_expr()

@njit
def calc_in_point_with_sympy(i, j, k, ip, jp, kp, 
                             vx, vy, vz, bx, by, bz, rho, pm, p, u, divv, divb, dxrho, dyrho, dzrho,
                             f=njit(SourceSs22().fct)):
    
    vxP, vyP, vzP = vx[ip, jp, kp], vy[ip, jp, kp], vz[ip, jp, kp]
    vxNP, vyNP, vzNP = vx[i, j, k], vy[i, j, k], vz[i, j, k]
    bxP, byP, bzP = bx[ip, jp, kp], by[ip, jp, kp], bz[ip, jp, kp]
    bxNP, byNP, bzNP = bx[i, j, k], by[i, j, k], bz[i, j, k]
    rhoP, rhoNP = rho[ip, jp, kp], rho[i, j, k]
    uP, uNP = u[ip, jp, kp], u[i, j, k]
    pmP, pmNP = pm[ip, jp, kp], pm[i, j, k]
    pP, pNP = p[ip, jp, kp], p[i, j, k]
    divvP, divvNP = divv[ip, jp, kp], divv[i, j, k]
    divbP, divbNP = divb[ip, jp, kp], divb[i, j, k]
    dxrhoP, dyrhoP, dzrhoP = dxrho[ip, jp, kp], dyrho[ip, jp, kp], dzrho[ip, jp, kp]
    dxrhoNP, dyrhoNP, dzrhoNP = dxrho[i, j, k], dyrho[i, j, k], dzrho[i, j, k]
    
    out = (f(vxP, vyP, vzP, vxNP, vyNP, vzNP, 
                      bxP, byP, bzP, bxNP, byNP, bzNP, 
                      rhoP, rhoNP, uP, uNP, 
                      pmP, pmNP, pP, pNP, divvP, divbP, dxrhoP, dyrhoP, dzrhoP)
           + f(vxNP, vyNP, vzNP, vxP, vyP, vzP, 
                      bxNP, byNP, bzNP, bxP, byP, bzP, 
                      rhoNP, rhoP, uNP, uP, 
                      pmNP, pmP, pNP, pP, divvNP, divbNP, dxrhoNP, dyrhoNP, dzrhoNP))
    
    return out
