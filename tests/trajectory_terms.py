# trajectory_terms.py
"""
Module pour calculer les termes le long d'une trajectoire.
Analogue à trajectory_quantities.py mais pour les termes.
Utilise les méthodes calc_fourier() des termes pour les trajectoires.
"""

import numpy as np
from exact_laws.el_calc_mod.laws import LAWS
from exact_laws.el_calc_mod.terms import TERMS
from trajectory_quantities import (
    list_required_quantities,
    compute_all_available_quantities
)


# Mapping des variables abstraites vers leurs composantes concrètes
VARIABLE_COMPONENTS = {
    # === Données brutes ===
    'v': ['vx', 'vy', 'vz'],
    'Iv': ['Ivx', 'Ivy', 'Ivz'],
    'b': ['bx', 'by', 'bz'],
    'Ib': ['Ibx', 'Iby', 'Ibz'],
    'w': ['wx', 'wy', 'wz'],  # Vitesse de vent (compressible)
    'j': ['jx', 'jy', 'jz'],  # Courant
    'Ij': ['Ijx', 'Ijy', 'Ijz'],  # Courant incompressible
    'f': ['fx', 'fy', 'fz'],  # Force
    
    # === Scalaires directs ===
    'rho': ['rho'],
    'Irho': ['Irho'],
    'pm': ['pm'],  # Pression magnétique
    'Ipm': ['Ipm'],  # Pression magnétique incompressible
    'pgyr': ['pgyr'],  # Pression gyrotropique
    'Ipgyr': ['Ipgyr'],  # Pression gyrotropique incompressible
    'piso': ['piso'],  # Pression isotrope
    'Ipiso': ['Ipiso'],  # Pression isotrope incompressible
    'ppol': ['ppol'],  # Pression poloïdale
    'Ippol': ['Ippol'],  # Pression poloïdale incompressible
    'pcgl': ['pcgl'],  # Pression CGL
    'Ipcgl': ['Ipcgl'],  # Pression CGL incompressible
    'ugyr': ['ugyr'],  # Vitesse gyrotropique
    'Iugyr': ['Iugyr'],  # Vitesse gyrotropique incompressible
    'uiso': ['uiso'],  # Vitesse isotrope
    'Iuiso': ['Iuiso'],  # Vitesse isotrope incompressible
    'upol': ['upol'],  # Vitesse poloïdale
    'Iupol': ['Iupol'],  # Vitesse poloïdale incompressible
    'ucgl': ['ucgl'],  # Vitesse CGL
    'Iucgl': ['Iucgl'],  # Vitesse CGL incompressible
    
    # === Divergences ===
    'divv': ['divvx', 'divvy', 'divvz'],  # Divergence de vitesse
    'divb': ['divbx', 'divby', 'divbz'],  # Divergence du champ magnétique
    'divj': ['divjx', 'divjy', 'divjz'],  # Divergence du courant
    
    # === Gradients ===
    'gradrho': ['dxrho', 'dyrho', 'dzrho'],  # Gradient de densité
    'gradv': ['grad_v_x', 'grad_v_y', 'grad_v_z'],  # Gradient de vitesse
    'graduiso': ['grad_uiso_x', 'grad_uiso_y', 'grad_uiso_z'],  # Gradient vitesse isotrope
    'gradupol': ['grad_upol_x', 'grad_upol_y', 'grad_upol_z'],  # Gradient vitesse poloïdale
    
    # === Hyperdissipation ===
    'hdk': ['hdkx', 'hdky', 'hdkz'],  # Hyperdissipation cinétique
    'hdm': ['hdmx', 'hdmy', 'hdmz'],  # Hyperdissipation magnétique
    'hdk2': ['hdk2x', 'hdk2y', 'hdk2z'],  # Hyperdissipation cinétique ordre 2
}


def list_required_terms(laws=None):
    """
    Retourne la liste des termes requis pour calculer les lois données.
    
    Paramètres:
    -----------
    laws : list[str]
        Liste des noms des lois
    
    Retour:
    -------
    set : Ensemble des termes requis
    """
    if laws is None:
        laws = []
    
    terms = set()
    
    # Add terms from laws
    if laws:
        for law_name in laws:
            if law_name in LAWS:
                law_obj = LAWS[law_name]
                # terms_and_coeffs() retourne (terms_list, coeffs_dict)
                # On passe des paramètres vides pour cette étape
                law_terms, _ = law_obj.terms_and_coeffs({"rho_mean": 1.0})
                terms.update(law_terms)
    
    return terms


def get_concrete_variables_from_abstract(abstract_vars, dic_quant):
    """
    Convertit les variables abstraites en composantes concrètes.
    
    Paramètres:
    -----------
    abstract_vars : list[str]
        Liste des variables abstraites (ex: ['v', 'b'])
    dic_quant : dict
        Dictionnaire contenant les données
    
    Retour:
    -------
    list : Liste de np.ndarray correspondant aux variables concrètes
    """
    concrete_data = []
    
    for var in abstract_vars:
        if var in VARIABLE_COMPONENTS:
            components = VARIABLE_COMPONENTS[var]
            for comp in components:
                if comp not in dic_quant:
                    raise ValueError(f"Component '{comp}' (from variable '{var}') not found in data")
                concrete_data.append(dic_quant[comp])
        else:
            # Variable qui n'est pas dans le mapping, la chercher directement
            if var not in dic_quant:
                raise ValueError(f"Variable '{var}' not found in data and not in VARIABLE_COMPONENTS mapping")
            concrete_data.append(dic_quant[var])
    
    return concrete_data


def compute_term_from_TERMS(term_name, dic_quant, dic_param, verbose=False):
    """
    Calcule un terme en utilisant la méthode calc_fourier de TERMS.
    Adapté pour les données 1D de trajectoires.
    
    Paramètres:
    -----------
    term_name : str
        Nom du terme (ex: "flux_dvdvdv", "bg17_vwv")
    dic_quant : dict
        Dictionnaire des données 1D (trajectoire) avec quantités calculées
    dic_param : dict
        Paramètres de simulation
    verbose : bool
    
    Retour:
    -------
    np.ndarray : Valeur du terme calculée (toujours un array)
    """
    
    if verbose:
        print(f"  Computing term {term_name}...", end=" ")
    
    if term_name not in TERMS:
        raise ValueError(f"Term '{term_name}' not found in TERMS")
    
    term_obj = TERMS[term_name]
    
    try:
        # Récupérer les variables abstraites requises par le terme
        abstract_vars = term_obj.variables()
        
        # Convertir en composantes concrètes
        args = get_concrete_variables_from_abstract(abstract_vars, dic_quant)
        
        # Calculer le terme avec traj=True pour indiquer une trajectoire
        result = term_obj.calc_fourier(*args, traj=True)
        
        # Convertir le résultat en array numpy
        if isinstance(result, list):
            result = np.array(result)
        elif not isinstance(result, np.ndarray):
            result = np.array([result])
        
    except Exception as e:
        if verbose:
            print(f"✗ {e}")
        raise
    
    if verbose:
        print("✓")
    
    return result


def compute_all_terms_for_laws(dic_quant, dic_param, laws=None, verbose=False):
    """
    Calcule tous les termes requis pour les lois données.
    Les quantités dérivées sont calculées automatiquement avant les termes.
    
    Paramètres:
    -----------
    dic_quant : dict
        Dictionnaire des données 1D ou 3D
    dic_param : dict
        Paramètres de simulation
    laws : list[str]
        Liste des lois
    verbose : bool
    
    Retour:
    -------
    dict : Dictionnaire des termes calculés
    """
    
    if laws is None:
        laws = []
    
    # Get required terms
    required_terms = list_required_terms(laws)
    
    if verbose:
        print(f"\n=== Computing {len(required_terms)} terms ===")
        print(f"Required terms: {required_terms}\n")
    
    # Get required quantities for these terms and compute them
    required_quantities = list_required_quantities(laws=laws)
    if verbose:
        print(f"Required quantities for terms: {required_quantities}")
    
    # Compute all available quantities (this will calculate v, b, Ib, etc. from raw data)
    dic_quant_complete = compute_all_available_quantities(
        dic_quant, dic_param, required_quantities, verbose=verbose
    )
    
    result = {}
    
    for term_name in required_terms:
        try:
            computed = compute_term_from_TERMS(term_name, dic_quant_complete, dic_param, verbose)
            result[term_name] = computed
        except Exception as e:
            if verbose:
                print(f"  ✗ {term_name}: {str(e)}")
    
    return result


def extract_trajectory_and_compute_terms(dic_quant, y_pos, z_pos, dic_param=None, 
                                         laws=None, verbose=False):
    """
    Extrait une trajectoire 1D et calcule les termes.
    
    Paramètres:
    -----------
    dic_quant : dict
        Dictionnaire des données 3D
    y_pos : int
        Position y de la trajectoire
    z_pos : int
        Position z de la trajectoire
    dic_param : dict
        Paramètres (optionnels)
    laws : list[str]
        Lois dont on veut calculer les termes
    verbose : bool
    
    Retour:
    -------
    dict : Dictionnaire des termes calculés
    """
    
    if dic_param is None:
        dic_param = {}
    
    if laws is None:
        laws = []
    
    # Extract 1D trajectory
    trajectory_data = {}
    
    for key in dic_quant.keys():
        if isinstance(dic_quant[key], np.ndarray) and dic_quant[key].ndim == 3:
            trajectory_data[key] = dic_quant[key][:, y_pos, z_pos]
        else:
            trajectory_data[key] = dic_quant[key]
    
    # Add means if available
    for key in dic_quant.keys():
        if isinstance(dic_quant[key], np.ndarray) and dic_quant[key].ndim == 3:
            if key == "ppar":
                dic_param["meanppar"] = np.mean(dic_quant[key])
            elif key == "pperp":
                dic_param["meanpperp"] = np.mean(dic_quant[key])
            elif key == "rho":
                dic_param["rho_mean"] = np.mean(dic_quant[key])
    
    if verbose:
        print(f"\n=== Trajectory at (y={y_pos}, z={z_pos}) ===")
        array_keys = [k for k in trajectory_data.keys() if isinstance(trajectory_data[k], np.ndarray)]
        if array_keys:
            print(f"Data shape: {trajectory_data[array_keys[0]].shape}")
    
    # Compute all terms for laws
    dic_terms = compute_all_terms_for_laws(trajectory_data, dic_param, laws, verbose)
    
    return dic_terms


def display_results(dic_terms, title="Results along trajectory"):
    """
    Affiche les résultats des termes le long d'une trajectoire.
    
    Paramètres:
    -----------
    dic_terms : dict
        Dictionnaire des termes
    title : str
        Titre de l'affichage
    """
    print(f"\n=== {title} ===")
    
    for key in sorted(dic_terms.keys()):
        value = dic_terms[key]
        if isinstance(value, np.ndarray) and value.ndim == 1:
            print(f"{key:30s}: min={value.min():12.6e}, max={value.max():12.6e}, mean={value.mean():12.6e}")
        elif isinstance(value, np.ndarray):
            print(f"{key:30s}: shape={value.shape}, mean={value.mean():12.6e}")
        elif isinstance(value, (int, float, np.number)):
            print(f"{key:30s}: {value:15.6e}")
        else:
            print(f"{key:30s}: {value}")