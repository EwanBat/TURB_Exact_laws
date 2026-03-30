# trajectory_quantities.py - VERSION SIMPLIFIÉE
"""
Module pour calculer les quantités non-dérivatives le long d'une trajectoire.
Analogue à process_on_oca_files.py mais adapté aux trajectoires et utilisant QUANTITIES.
Les quantités (v ou Iv, etc.) sont déterminées par les variables requises des lois.
"""

import numpy as np
import numexpr as ne
import configparser
import logging
from pathlib import Path
from exact_laws.preprocessing.quantities import QUANTITIES
from exact_laws.el_calc_mod.laws import LAWS
from exact_laws.el_calc_mod.terms import TERMS

logger = logging.getLogger(__name__)

# ========== MAPPING DES QUANTITÉS ET LEURS DÉPENDANCES ==========
# Toutes les quantités (compressible et incompressible) dans un même dictionnaire
# La version correcte (v ou Iv) est déterminée par les lois

QUANTITY_DEPENDENCIES = {
    # === Données brutes ===
    "v": {"requires": ["vx", "vy", "vz"]},
    "Iv": {"requires": ["Ivx", "Ivy", "Ivz"]},
    "rho": {"requires": ["rho"]},
    "Irho": {"requires": ["Irho"]},
    "b": {"requires": ["bx", "by", "bz"]},
    "Ib": {"requires": ["bx", "by", "bz"]},
    
    # === Vitesse ===
    "v2": {"requires": ["vx", "vy", "vz"]},
    "Iv2": {"requires": ["Ivx", "Ivy", "Ivz"]},
    "vnorm": {"requires": ["vx", "vy", "vz"]},
    "Ivnorm": {"requires": ["Ivx", "Ivy", "Ivz"]},
    
    # === Champ magnétique ===
    "bnorm": {"requires": ["bx", "by", "bz"]},
    "Ibnorm": {"requires": ["bx", "by", "bz"]},
    "pm": {"requires": ["bx", "by", "bz"]},
    "Ipm": {"requires": ["bx", "by", "bz"]},
    
    # === Pressions ===
    "pgyr": {"requires": ["pperp", "rho"]},
    "Ipgyr": {"requires": ["pperp"]},
    "piso": {"requires": ["ppar", "pperp"]},
    "Ipiso": {"requires": ["ppar", "pperp"]},
    "ppol": {"requires": ["pperp"]},
    "Ippol": {"requires": ["pperp"]},
    "pcgl": {"requires": ["bx", "by", "bz", "rho", "ppar", "pperp"]},
    "Ipcgl": {"requires": ["bx", "by", "bz", "ppar", "pperp"]},
    
    # === Vitesses dérivées de pression ===
    "ugyr": {"requires": ["pperp", "rho"]},
    "Iugyr": {"requires": ["pperp"]},
    "uiso": {"requires": ["ppar", "pperp", "rho"]},
    "Iuiso": {"requires": ["ppar", "pperp"]},
    "ucgl": {"requires": ["bx", "by", "bz", "rho", "ppar", "pperp"]},
    "Iucgl": {"requires": ["bx", "by", "bz", "ppar", "pperp"]},
    "upol": {"requires": ["ppar", "pperp", "rho"]},
    "Iupol": {"requires": ["pperp"]},
}


def load_config_from_ini(config_file):
    """
    Charge les paramètres depuis un fichier .ini
    
    Retour:
    -------
    tuple : (laws, terms, quantities, physical_params)
    """
    config = configparser.ConfigParser()
    config.read(config_file)
    
    laws = eval(config["OUTPUT_DATA"].get("laws", "[]"))
    terms = eval(config["OUTPUT_DATA"].get("terms", "[]"))
    quantities = eval(config["OUTPUT_DATA"].get("quantities", "[]"))
    
    physical_params = {}
    if "PHYSICAL_PARAMS" in config:
        for key in config["PHYSICAL_PARAMS"].keys():
            try:
                physical_params[key] = float(eval(config["PHYSICAL_PARAMS"][key]))
            except:
                physical_params[key] = config["PHYSICAL_PARAMS"][key]
    
    return laws, terms, quantities, physical_params


def list_required_quantities(laws=None, terms=None, quantities=None):
    """
    Retourne la liste des quantités requises pour calculer les lois et termes donnés.
    Les versions correctes (v ou Iv) viennent des variables des lois.
    
    Paramètres:
    -----------
    laws : list[str]
        Liste des noms des lois
    terms : list[str]
        Liste des noms des termes
    quantities : list[str]
        Liste des quantités à ajouter directement
    
    Retour:
    -------
    set : Ensemble des quantités requises
    """
    if quantities is None:
        quantities = []
    else:
        quantities = list(quantities)
    
    # Add requirements from terms
    if terms:
        for term_name in terms:
            if term_name in TERMS:
                term_variables = TERMS[term_name].variables()
                quantities.extend(term_variables)
    
    # Add requirements from laws
    if laws:
        for law_name in laws:
            if law_name in LAWS:
                law_variables = LAWS[law_name].variables()
                quantities.extend(law_variables)
    
    return set(quantities)


def get_available_quantities_for_requirements(dic_quant, required_quantities):
    """
    Retourne la liste des quantités disponibles à partir des données et des exigences.
    Les quantités (v ou Iv) sont déjà correctement spécifiées dans required_quantities.
    
    Paramètres:
    -----------
    dic_quant : dict
        Dictionnaire des données brutes ou 1D
    required_quantities : set
        Ensemble des quantités requises
    
    Retour:
    -------
    list : Liste des quantités calculables
    """
    available = []
    
    for quantity_name in required_quantities:
        if quantity_name not in QUANTITY_DEPENDENCIES:
            continue
        
        # Check if it's a direct raw quantity
        if quantity_name in dic_quant:
            available.append(quantity_name)
        # Check if it's a derived quantity we can compute
        elif all(req in dic_quant or req in available for req in QUANTITY_DEPENDENCIES[quantity_name]["requires"]):
            available.append(quantity_name)
    
    return available


def compute_quantity_from_QUANTITIES(quantity_name, dic_quant, dic_param, verbose=False):
    """
    Calcule une quantité en utilisant les objets de QUANTITIES.
    Adapté pour les données 1D de trajectoires.
    
    Paramètres:
    -----------
    quantity_name : str
        Nom complet de la quantité (ex: "v2", "Iv2", "bnorm")
    dic_quant : dict
        Dictionnaire des données 1D (trajectoire)
    dic_param : dict
        Paramètres de simulation
    verbose : bool
    """
    
    if verbose:
        logger.info(f"Computing {quantity_name}...")
    
    if quantity_name not in QUANTITIES:
        raise ValueError(f"Quantity '{quantity_name}' not found in QUANTITIES")
    
    quantity_obj = QUANTITIES[quantity_name]
    
    class MockFile:
        def __init__(self):
            self.data = {}
        
        def create_dataset(self, name, data=None, **kwargs):
            self.data[name] = data if data is not None else np.empty(0)
    
    mock_file = MockFile()
    
    try:
        quantity_obj.create_datasets(mock_file, dic_quant, dic_param)
    except Exception as e:
        if verbose:
            logger.error(f"Failed to compute {quantity_name}: {e}")
        raise
    
    if len(mock_file.data) == 1:
        result = list(mock_file.data.values())[0]
    else:
        result = mock_file.data
    
    if verbose:
        logger.info(f"Quantity {quantity_name} computed")
    
    return result


def compute_all_available_quantities(dic_quant, dic_param, required_quantities=None, verbose=False):
    """
    Calcule TOUTES les quantités disponibles en fonction des données et des exigences.
    
    Paramètres:
    -----------
    dic_quant : dict
        Dictionnaire des données 1D
    dic_param : dict
        Paramètres de simulation
    required_quantities : set or list
        Quantités requises (si None, calcule toutes les possibles)
    verbose : bool
    
    Retour:
    -------
    dict : Dictionnaire de toutes les quantités calculées
    """
    
    if required_quantities is None:
        required_quantities = set(QUANTITY_DEPENDENCIES.keys())
    else:
        required_quantities = set(required_quantities)
    
    # Filter to computable quantities
    available_quantities = get_available_quantities_for_requirements(dic_quant, required_quantities)
    
    if verbose:
        logger.info(f"Computing {len(available_quantities)} quantities")
        logger.info(f"Required: {required_quantities}")
        logger.info(f"Available to compute: {available_quantities}")
    
    result = dic_quant.copy()
    
    for quantity_name in available_quantities:
        # Skip if already computed or is raw data
        if quantity_name in result:
            continue
        
        try:
            computed = compute_quantity_from_QUANTITIES(
                quantity_name, result, dic_param, verbose
            )
            
            if isinstance(computed, dict):
                result.update(computed)
            else:
                result[quantity_name] = computed
        except Exception as e:
            if verbose:
                logger.error(f"Failed to compute {quantity_name}: {str(e)}")
    
    return result


def extract_trajectory_and_compute(dic_quant, y_pos, z_pos, dic_param=None, 
                                   laws=None, terms=None, quantities=None,
                                   verbose=False):
    """
    Extrait une trajectoire 1D et calcule les quantités requises pour les lois/termes.
    
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
        Lois à calculer
    terms : list[str]
        Termes à calculer
    quantities : list[str]
        Quantités additionnelles
    verbose : bool
    
    Retour:
    -------
    dict : Quantités 1D calculées le long de la trajectoire
    """
    
    if dic_param is None:
        dic_param = {}
    
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
                trajectory_data["meanppar"] = np.mean(dic_quant[key])
            elif key == "pperp":
                trajectory_data["meanpperp"] = np.mean(dic_quant[key])
    
    if verbose:
        logger.info(f"Trajectory at (y={y_pos}, z={z_pos})")
        array_keys = [k for k in trajectory_data.keys() if isinstance(trajectory_data[k], np.ndarray)]
        if array_keys:
            logger.info(f"Data shape: {trajectory_data[array_keys[0]].shape}")
    
    # Determine required quantities
    required_qty = list_required_quantities(laws, terms, quantities)
    
    if verbose:
        logger.info(f"Required quantities from laws/terms: {required_qty}")
    
    # Compute all available quantities
    return compute_all_available_quantities(
        trajectory_data, dic_param, required_qty, verbose
    )


def display_results(traj_quantities, title="Results along trajectory"):
    """
    Affiche les résultats d'une trajectoire.
    """
    logger.info(f"\n{title}")
    logger.info("-" * 70)
    for key in sorted(traj_quantities.keys()):
        value = traj_quantities[key]
        if isinstance(value, np.ndarray) and value.ndim == 1:
            logger.info(f"  {key:20s}: min={value.min():12.6e} | max={value.max():12.6e} | mean={value.mean():12.6e}")
        elif isinstance(value, (int, float, np.number)):
            logger.info(f"  {key:20s}: {value:15.6e}")
