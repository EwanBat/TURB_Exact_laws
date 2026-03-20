from typing import List
from numba import njit
import sympy as sp

from .abstract_term import AbstractTerm, calc_source_with_numba

class SourceSs21(AbstractTerm):
    def __init__(self):
        self.set_sympy_expr()
        quantities = ("vx'", "vy'", "vz'", "vx", "vy", "vz", 
                      "bx'", "by'", "bz'", "bx", "by", "bz", 
                      "rho'", "rho", "u'", "u", "pm'", "p'",
                     "divv'", "divb'", "dxu'", "dyu'", "dzu'",
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
        pmP = sp.symbols(("pm'"))
        pP = sp.symbols(("p'"))
        uP, uNP = sp.symbols(("u'", "u"))
        dxuP, dyuP, dzuP = sp.symbols(("dxu'", "dyu'", "dzu'"))
        
        self.expr = (divvP * (rhoNP * (vxNP * (vxP - vxNP) + vyNP * (vyP - vyNP) + vzNP * (vzP - vzNP)
                                       + bxNP * (bxP - bxNP) + byNP * (byP - byNP) + bzNP * (bzP - bzNP)
                                       + 2 * (uP - uNP - pP))
                              - 1/2 * (rhoP + rhoNP) * (bxP * bxNP + byP * byNP + bzP * bzNP))
                     - divbP * (2 * rhoNP * (vxNP * (bxP - bxNP) + vyNP * (byP - byNP) + vzNP * (bzP - bzNP))
                                - (rhoP - rhoNP) * (vxP * bxNP + vyP * byNP + vzP * bzNP))
                     - pmP / pP * rhoNP * (vxNP * dxuP + vyNP * dyuP + vzNP * dzuP))
        
    
    def calc(self, vector:List[int], cube_size:List[int],vx, vy, vz, bx, by, bz, rho, pm, p, u, divv, divb, dxu, dyu, dzu, **kwarg) -> List[float]:
        return calc_source_with_numba(calc_in_point_with_sympy, *vector, *cube_size, vx, vy, vz, bx, by, bz, rho, pm, p, u, divv, divb, dxu, dyu, dzu)

    def variables(self) -> List[str]:
        return ['v', 'b', 'rho', 'pm', 'p', 'u', 'divv', 'divb', 'gradu']
    
    def print_expr(self):
        sp.init_printing(use_latex=True)
        return self.expr

def load():
    return SourceSs21()

def print_expr():
    return SourceSs21().print_expr()

@njit
def calc_in_point_with_sympy(i, j, k, ip, jp, kp, 
                             vx, vy, vz, bx, by, bz, rho, pm, p, u, divv, divb, dxu, dyu, dzu,
                             f=njit(SourceSs21().fct)):
    
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
    dxuP, dyuP, dzuP = dxu[ip, jp, kp], dyu[ip, jp, kp], dzu[ip, jp, kp]
    dxuNP, dyuNP, dzuNP = dxu[i, j, k], dyu[i, j, k], dzu[i, j, k]
    
    out = (f(vxP, vyP, vzP, vxNP, vyNP, vzNP, 
                      bxP, byP, bzP, bxNP, byNP, bzNP, 
                      rhoP, rhoNP, uP, uNP, 
                      pmP, pP, divvP, divbP, dxuP, dyuP, dzuP)
           + f(vxNP, vyNP, vzNP, vxP, vyP, vzP, 
                      bxNP, byNP, bzNP, bxP, byP, bzP, 
                      rhoNP, rhoP, uNP, uP, 
                      pmNP, pNP, divvNP, divbNP, dxuNP, dyuNP, dzuNP))
    
    return out
