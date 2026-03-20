import numpy as np
import random

from .incgrid import IncGrid
from .grid import Grid
from ...mathematical_tools.derivation import cdiff

def logregular_axis(size, maxi):
    axis = np.unique(np.logspace(0, np.log10(maxi), size, endpoint=True, dtype=int))
    axis = np.array(np.append([0], axis))
    return axis


def build_logregular_cylindrical_incremental_grid(original_grid, Nmax_scale, Nmax_list, kind="cls"):
    axis = ["lz", "lperp", "listperp"]
    grid = {}

    # liste des échelles parallèles (indexes des coupes parallèles)
    grid["lz"] = logregular_axis(Nmax_scale, original_grid.N[2] / 2)
    N_par = np.shape((grid["lz"]))[0]

    # liste des échelles perp (norme polaire)
    grid["lperp"] = logregular_axis(Nmax_scale, np.min(original_grid.N[0:2]) / 2)
    N_perp = np.shape((grid["lperp"]))[0]

    # init listes des points présent sur un même arc de rayon lperp +/- un ecart
    grid["listperp"] = [[]] * N_perp  # list [[(x,y),...],...]
    grid["listecart"] = [[]] * N_perp  # list [[ecart,...],...]
    grid["count"] = []  # list [len(listperp),...]

    # precision : ecart maximal accepté entre un point et l'arc associé
    precision = [
        np.min(((grid["lperp"][r + 1] - grid["lperp"][r]) / 2, (grid["lperp"][r] - grid["lperp"][r - 1]) / 2))
        for r in range(1, N_perp - 1)
    ]
    precision = np.append([(grid["lperp"][1] - grid["lperp"][0]) / 2], precision)  # ajout premier point
    precision = np.append(precision, [(grid["lperp"][-1] - grid["lperp"][-2]) / 2])  # ajout dernier point

    # liste des points cartésiens
    limX = np.min((grid["lperp"][-1], int(original_grid.N[0] / 2)))  # limite d'intéret en x
    limY = np.min((grid["lperp"][-1], int(original_grid.N[1] / 2)))  # limite d'intéret en y
    points = [[np.sqrt(x * x + y * y), (x, y)] for x in range(-limX, limX + 1) for y in range(-limY, limY + 1)]

    # distribution des points cartésiens suivant les arcs + réduction du nombre de points suivant kind
    for r in range(N_perp):
        limm = grid["lperp"][r] - precision[r]
        limp = grid["lperp"][r] + precision[r]
        lpoints = [
            [np.abs(e[0] - grid["lperp"][r]), e[1]]
            for e in list(filter(lambda e: e[0] >= limm and e[0] <= limp, points))
        ]

        # réduction aléatoire (rdm) ou au plus près (cls) du nombre de points près d'un arc lperp à Nmax_list points si supérieur
        if len(lpoints) > Nmax_list:
            if kind == "cls":
                sortlpoints = sorted(lpoints)[:Nmax_list]
                grid["listperp"][r] = [x for _, x in sortlpoints]
                grid["listecart"][r] = [x for x, _ in sortlpoints]

            elif kind == "rdm":
                rdmlpoints = random.sample(lpoints, Nmax_list)
                grid["listperp"][r] = [x for _, x in rdmlpoints]
                grid["listecart"][r] = [x for x, _ in rdmlpoints]
        else:
            grid["listperp"][r] = [x for _, x in lpoints]
            grid["listecart"][r] = [x for x, _ in lpoints]

        grid["count"].append(len(grid["listperp"][r]))

    return IncGrid(
        original_grid=original_grid,
        N=np.array([N_par, N_perp, np.max(grid["count"])], dtype=int),
        axis=axis,
        coords=grid,
        kind= kind,
        coord = 'logcyl',
    )


def load(original_grid, Nmax_scale, Nmax_list, kind, **kargs):
    return build_logregular_cylindrical_incremental_grid(original_grid, Nmax_scale, Nmax_list, kind)


def build_listcoords(incgrid, nb_sec_by_dirr=1, **kargs):
    N = incgrid.spatial_grid.N

    list_prim = []
    for z in incgrid.coords["lz"]:
        for perp in range(len(incgrid.coords["lperp"])):
            for vect_perp in incgrid.coords["listperp"][perp]:
                list_prim.append((vect_perp[0], vect_perp[1], z))
    set_prim = set(list_prim)
    list_prim = list(set_prim)

    list_sec = []
    for vect_prim in list_prim:
        for dirr in range(len(N)):
            for i in range(-nb_sec_by_dirr, nb_sec_by_dirr + 1):
                if i != 0:
                    vect = list(vect_prim)
                    vect[dirr] = (
                        (vect[dirr] + i)
                        - (N[dirr] * ((vect[dirr] + i) >= (N[dirr] / 2)))
                        + (N[dirr] * ((vect[dirr] + i) <= (-N[dirr] / 2)))
                    )
                    vect = (*vect,)
                    if not vect in set_prim:
                        list_sec.append(vect)
    list_sec = list(set(list_sec))
    return list_prim, list_sec, nb_sec_by_dirr

def load_outputgrid(incgrid, nb_sec_by_dirr=1):
    """
    Args:
        incgrid (IncGrid object)
        nb_sec_by_dirr (int) : 0, 1, 2 
    Returns:
        Grid object that contains list of coordinates
    """
    coords = {}
    coords['listprim'], coords['listsec'], coords['nb_sec_by_dirr'] = build_listcoords(incgrid, nb_sec_by_dirr)    
    return Grid(axis=['listprim','listsec'], N=[len(coords['listprim']),len(coords['listsec'])], coords=coords)

def coordinate_sec_in_primsec_grid(vect_prim,list_prim,list_sec,nb_sec_by_dirr,N):
    points_sec = [[], [], []]
    for dirr in range(3):
        for i in range(-nb_sec_by_dirr, nb_sec_by_dirr + 1):
            if not i == 0:
                vect = list(vect_prim)
                vect[dirr] = (
                    (vect[dirr] + i)
                    - (N[dirr] * ((vect[dirr] + i) >= (N[dirr] / 2)))
                    + (N[dirr] * ((vect[dirr] + i) <= (-N[dirr] / 2)))
                )
                vect = tuple(vect)
                loc = -1
                try:
                    index = list_prim.index(vect)
                    loc = 0
                except:
                    index = list_sec.index(vect)
                    loc = 1
                points_sec[dirr].append((loc, index))
    return points_sec

def div(incgrid, dataset_terms):
    list_prim = dataset_terms.grid.coords['listprim']
    list_sec = dataset_terms.grid.coords['listsec']
    nb_sec_by_dirr = dataset_terms.grid.coords['nb_sec_by_dirr']
    N = incgrid.spatial_grid.N
    c = incgrid.spatial_grid.c
    
    output = {}
    list_flux = []
    for t in dataset_terms.quantities:
        if t.startswith("flux"):
            output["div_" + t] = np.zeros((len(list_prim)))
            list_flux.append(t)

    for ind, vect_prim in enumerate(list_prim):
        points_sec = coordinate_sec_in_primsec_grid(vect_prim,list_prim, list_sec, nb_sec_by_dirr,N)
        for t in list_flux:
            div_point = 0
            if len(points_sec[0]) == 0:
                div_point += np.nan
            else:
                for dirr in range(3):
                    values = []
                    for i in range(len(points_sec[dirr])):
                        values.append(dataset_terms.quantities[t][points_sec[dirr][i][0]][points_sec[dirr][i][1]][dirr])
                    div_point += cdiff(
                        tab = values,
                        length_case=c[dirr],
                        precision=nb_sec_by_dirr * 2,
                        period=False,
                        point=True,
                    )
            output["div_" + t][ind] = div_point
    return output

def reorganise_quantities(output_quantities, incgrid, output_grid,  nb_sec_by_dirr=1):
    output = {}
    N = incgrid.spatial_grid.N
    shape_scalar = incgrid.N
    shape_vector = [*shape_scalar, 3]
    shape_termdiv = [*shape_vector, 2 * nb_sec_by_dirr]

    list_flux = [k for k in output_quantities if k.startswith("flux")]
    for t in list_flux:
        output[t] = np.zeros(shape_vector)*np.nan
        output["term_div_" + t] = np.zeros(shape_termdiv)*np.nan

    list_other = [k for k in output_quantities if not k.startswith("flux")]
    for t in list_other:
        output[t] = np.zeros(shape_scalar)*np.nan

    sec_list = list(np.arange(-nb_sec_by_dirr, nb_sec_by_dirr + 1, 1))
    sec_list.remove(0)

    for ind_z, z in enumerate(incgrid.coords["lz"]):
        for ind_perp in range(incgrid.N[1]):
            for ind_vect, vect_perp in enumerate(incgrid.coords["listperp"][ind_perp]):
                vect_prim = (vect_perp[0], vect_perp[1], z)
                prim_index = output_grid.coords["listprim"].index(vect_prim)
                for t in list_flux:
                    output[t][ind_z, ind_perp, ind_vect] = [*output_quantities[t][0][prim_index]]
                for t in list_other:
                    output[t][ind_z, ind_perp, ind_vect] = output_quantities[t][0][prim_index]
                for dirr in range(len(N)):
                    for ind_point, i in enumerate(sec_list):
                        vect = list(vect_prim)
                        vect[dirr] = (
                            (vect[dirr] + i)
                            - (N[dirr] * ((vect[dirr] + i) >= (N[dirr] / 2)))
                            + (N[dirr] * ((vect[dirr] + i) <= (-N[dirr] / 2)))
                        )
                        vect = (*vect,)
                        loc = -1
                        try:
                            index = output_grid.coords["listprim"].index(vect)
                            loc = 0
                        except:
                            index = output_grid.coords["listsec"].index(vect)
                            loc = 1
                        for t in list_flux:
                            output["term_div_" + t][ind_z, ind_perp, ind_vect, dirr, ind_point] = output_quantities[t][
                                loc
                            ][index][dirr]
    return output

def reformat_grid_compatible_to_h5(incgrid):
    output = {}
    output['inc_axis'] = incgrid.axis
    output['inc_N'] = incgrid.N
    output['kind'] = incgrid.kind
    output['axis'] = incgrid.spatial_grid.axis
    output['N'] = incgrid.spatial_grid.N
    output['L'] = incgrid.spatial_grid.L
    output['c'] = incgrid.spatial_grid.c
    output['coords'] = {k:incgrid.coords[k] for k in incgrid.coords if not k in ['listperp','listecart']}
    n = incgrid.N[2]
    output['coords']['listperp'] = [x + [[np.nan,np.nan]]*(n-len(x)) for x in incgrid.coords['listperp']]
    output['coords']['listecart'] = [x + [np.nan]*(n-len(x)) for x in incgrid.coords['listecart']]
    return output
