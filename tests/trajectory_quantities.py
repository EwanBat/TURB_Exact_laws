# trajectory_quantities.py - VERSION AVEC SUPPORT GRADIENT
"""
Module pour calculer les quantités non-dérivatives le long d'une trajectoire.
Analogue à process_on_oca_files.py mais adapté aux trajectoires et utilisant QUANTITIES.
Les quantités (v ou Iv, etc.) sont déterminées par les variables requises des lois.
Support pour gradient et divergence avec formations de 4 satellites.
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

# Quantités de gradients et divergences avec leurs dépendances
GRADIENT_QUANTITIES = {
    'gradv': {'requires': ['vx', 'vy', 'vz']},
    'gradv2': {'requires': ['v2']},  # v2 dépend de vx, vy, vz
    'gradrho': {'requires': ['rho']},
    'graduiso': {'requires': ['uiso']},  # uiso dépend de ppar, pperp, rho
    'gradupol': {'requires': ['upol']},  # upol dépend de ppar, pperp, rho
    'gradugyr': {'requires': ['ugyr']},  # ugyr dépend de pperp, rho
    'gradpcgl': {'requires': ['pcgl']},  # pcgl dépend de bx, by, bz, rho, ppar, pperp
    'divv': {'requires': ['vx', 'vy', 'vz']},
    'divb': {'requires': ['bx', 'by', 'bz']},
    'divj': {'requires': ['jx', 'jy', 'jz']},
    'Igradv': {'requires': ['Ivx', 'Ivy', 'Ivz']},
    'Igradv2': {'requires': ['Iv2']},
    'Igradrho': {'requires': ['Irho']},
    'Igraduiso': {'requires': ['Iuiso']},
    'Igradupol': {'requires': ['Iupol']},
}

def list_computable_quantities(dic_quant, laws=None, terms=None, quantities=None, nbsatellite=1):
    """
    Liste les quantités calculables à partir des données et des exigences.
    
    Combine l'extraction des exigences (lois/termes) avec la vérification 
    de disponibilité basée sur les données et nbsatellite.
    
    Paramètres:
    -----------
    dic_quant : dict
        Dictionnaire des données disponibles
    laws, terms, quantities : list, optional
        Spécifications des exigences
    nbsatellite : int
        Nombre de satellites (filtre les gradients si < 4)
    
    Retour:
    -------
    list : Quantités calculables
    """
    # === ÉTAPE 1 : Extraire les exigences ===
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
    
    required_quantities = set(quantities)
    
    # === ÉTAPE 2 : Filtrer les gradients selon nbsatellite ===
    if nbsatellite < 4:
        required_quantities = {q for q in required_quantities if q not in GRADIENT_QUANTITIES}
    
    # === ÉTAPE 3 : Vérifier la disponibilité ===
    # Combiner les dépendances normales et des gradients
    all_dependencies = {**QUANTITY_DEPENDENCIES, **GRADIENT_QUANTITIES}
    
    available = []
    for quantity_name in required_quantities:
        if quantity_name not in all_dependencies:
            # Quantité non documentée - on la tente quand même si elle existe
            if quantity_name in dic_quant:
                available.append(quantity_name)
            continue
        
        # Check if it's a direct raw quantity
        if quantity_name in dic_quant:
            available.append(quantity_name)
        # Check if it's a derived quantity we can compute
        elif all(req in dic_quant or req in available for req in all_dependencies[quantity_name]["requires"]):
            available.append(quantity_name)
    
    return available


def compute_gradient_stub(quantity_name: str, dic_quant: dict, separation: float = 1.0) -> np.ndarray:
    """
    Calcule une approximation du gradient d'une quantité.
    
    Utilise une fausse implémentation en attente d'une version propre.
    Assume que dic_quant[quantity_name] est un dictionnaire {sat_0, sat_1, sat_2, sat_3}
    
    Paramètres:
    -----------
    quantity_name : str
        Nom de la quantité dont on veut le gradient
    dic_quant : dict
        Dictionnaire contenant les données pour les 4 satellites
    separation : float
        Séparation entre les satellites en unités de grille
    
    Retour:
    -------
    dict : {sat_name: gradient_vector (n_points, 3)} ou (n_points, 3) selon la quantité
    """
    
    # Récupérer les données de base (sans le "grad")
    base_quantity = quantity_name.lstrip('I').replace('grad', '')
    
    if base_quantity not in dic_quant:
        raise ValueError(f"Quantité de base '{base_quantity}' non trouvée pour calculer {quantity_name}")
    
    data_per_sat = dic_quant[base_quantity]
    
    if not isinstance(data_per_sat, dict):
        raise ValueError(f"{base_quantity} ne contient pas de données par satellite")
    
    n_points = len(data_per_sat['sat_0'])
    
    # Initialiser le résultat
    result = np.zeros((n_points, 3))
    
    # Récupérer les données pour chaque satellite
    sat_0 = data_per_sat['sat_0']  # avant, haut
    sat_1 = data_per_sat['sat_1']  # avant, bas
    sat_2 = data_per_sat['sat_2']  # arrière, haut
    sat_3 = data_per_sat['sat_3']  # arrière, bas
    
    # Approximation du gradient (fausse implémentation)
    # À remplacer par une vraie interpolation spatiale
    
    # ∂f/∂x ≈ (f_avant - f_arrière) / (2 * separation)
    # Moyenne entre haut et bas pour avant et arrière
    f_avant = (sat_0 + sat_1) / 2.0
    f_arriere = (sat_2 + sat_3) / 2.0
    result[:, 0] = (f_avant - f_arriere) / (separation if separation > 0 else 1.0)
    
    # ∂f/∂y ≈ (f_haut - f_bas) / (2 * separation)
    # Moyenne entre avant et arrière pour haut et bas
    f_haut = (sat_0 + sat_2) / 2.0
    f_bas = (sat_1 + sat_3) / 2.0
    result[:, 1] = (f_haut - f_bas) / (separation if separation > 0 else 1.0)
    
    # ∂f/∂z ≈ 0 (pas de variation en z avec 4 satellites dans le plan xy)
    result[:, 2] = 0.0
    
    logger.warning(f"STUB: Gradient de {quantity_name} calculé avec approximation simple")
    
    return result


def compute_divergence_stub(base_quantity: str, dic_quant: dict, separation: float = 1.0) -> np.ndarray:
    """
    Calcule une approximation de la divergence d'un vecteur.
    
    Utilise une fausse implémentation en attente d'une version propre.
    
    Paramètres:
    -----------
    base_quantity : str
        Nom du vecteur (ex: "v", "b", "j")
    dic_quant : dict
        Dictionnaire contenant les données pour les 4 satellites
    separation : float
        Séparation entre les satellites
    
    Retour:
    -------
    np.ndarray : (n_points,) - divergence
    """
    
    if base_quantity not in dic_quant:
        raise ValueError(f"Quantité '{base_quantity}' non trouvée pour calculer divergence")
    
    data_per_sat = dic_quant[base_quantity]
    
    if not isinstance(data_per_sat, dict):
        raise ValueError(f"{base_quantity} ne contient pas de données par satellite")
    
    n_points = len(data_per_sat['sat_0'])
    
    # Récupérer les composantes vectorielles
    components = f"{base_quantity}x", f"{base_quantity}y", f"{base_quantity}z"
    
    # Approximation stub : divergence ≈ 0
    # À remplacer par une vraie implémentation
    result = np.zeros(n_points)
    
    logger.warning(f"STUB: Divergence de {base_quantity} = 0 (implémentation stub)")
    
    return result


def compute_quantity_from_QUANTITIES(quantity_name, dic_quant, dic_param, nbsatellite=1, 
                                     separation=1.0, verbose=False):
    """
    Calcule une quantité en utilisant les objets de QUANTITIES.
    Adapté pour les données 1D de trajectoires avec support gradient.
    """
    
    if verbose:
        logger.info(f"Computing {quantity_name} (nbsatellite={nbsatellite})...")
    
    # ===== CAS GRADIENT/DIVERGENCE =====
    if quantity_name in GRADIENT_QUANTITIES:
        if nbsatellite < 4:
            raise ValueError(f"{quantity_name} nécessite nbsatellite >= 4")
        
        base_quantity = quantity_name.replace('div', '').lstrip('I').replace('grad', '')
        
        if 'div' in quantity_name:
            result = compute_divergence_stub(base_quantity, dic_quant, separation)
        else:
            result = compute_gradient_stub(quantity_name, dic_quant, separation)
        
        return result
    
    # ===== CAS QUANTITÉ NORMALE =====
    if quantity_name not in QUANTITIES:
        raise ValueError(f"Quantity '{quantity_name}' not found in QUANTITIES")
    
    class MockFile:
        def __init__(self):
            self.data = {}
        
        def create_dataset(self, name, data=None, **kwargs):
            self.data[name] = data if data is not None else np.empty(0)
    
    # Cas nbsatellite = 1
    if nbsatellite == 1:
        mock_file = MockFile()
        try:
            QUANTITIES[quantity_name].create_datasets(mock_file, dic_quant, dic_param)
        except Exception as e:
            if verbose:
                logger.error(f"Failed to compute {quantity_name}: {e}")
            raise
        
        result = list(mock_file.data.values())[0] if len(mock_file.data) == 1 else mock_file.data
    
    # Cas nbsatellite > 1 : calculer pour chaque satellite
    else:
        result = {}
        for sat_name in ['sat_0', 'sat_1', 'sat_2', 'sat_3']:
            # Extraire les données pour ce satellite
            dic_quant_sat = {}
            dic_param_sat = {}
            for key, value in dic_quant.items():
                dic_quant_sat[key] = value[sat_name] if isinstance(value, dict) and sat_name in value else value
            for key, value in dic_param.items():
                dic_param_sat[key] = value[sat_name] if isinstance(value, dict) and sat_name in value else value

            # Calculer la quantité
            mock_file = MockFile()
            try:
                QUANTITIES[quantity_name].create_datasets(mock_file, dic_quant_sat, dic_param)
            except Exception as e:
                if verbose:
                    logger.error(f"Failed to compute {quantity_name} for {sat_name}: {e}")
                raise
            
            result[sat_name] = list(mock_file.data.values())[0] if len(mock_file.data) == 1 else mock_file.data
    
    if verbose:
        logger.info(f"Quantity {quantity_name} computed")
    
    return result

def compute_all_available_quantities(dic_quant, dic_param, available_quantities=None, 
                                     nbsatellite=1, separation=1.0, verbose=False):
    """
    Calcule TOUTES les quantités disponibles en fonction des données et des exigences.
    Traite séparément les cas nbsatellite=1 et nbsatellite=4.
    """
    
    if available_quantities is None:
        available_quantities = set(QUANTITY_DEPENDENCIES.keys())
    else:
        available_quantities = set(available_quantities)
    
    result = dic_quant.copy()
    
    if nbsatellite == 1:
        # ===== CAS nbsatellite=1 : pas de gradients =====
        for quantity_name in available_quantities:
            if quantity_name in result or quantity_name in GRADIENT_QUANTITIES:
                continue
            
            try:
                computed = compute_quantity_from_QUANTITIES(
                    quantity_name, result, dic_param, nbsatellite=1, verbose=verbose
                )
                if isinstance(computed, dict):
                    result.update(computed)
                else:
                    result[quantity_name] = computed
            except Exception as e:
                if verbose:
                    logger.error(f"Failed to compute {quantity_name}: {e}")
    
    elif nbsatellite == 4:
        # ===== CAS nbsatellite=4 : avec gradients =====
        # Étape 1 : quantités normales
        for quantity_name in available_quantities:
            if quantity_name in result or quantity_name in GRADIENT_QUANTITIES:
                continue
            
            try:
                computed = compute_quantity_from_QUANTITIES(
                    quantity_name, result, dic_param, nbsatellite=4, separation=separation, 
                    verbose=verbose
                )
                if isinstance(computed, dict):
                    result.update(computed)
                else:
                    result[quantity_name] = computed
            except Exception as e:
                if verbose:
                    logger.error(f"Failed to compute {quantity_name}: {e}")
        
        # Étape 2 : gradients et divergences
        for quantity_name in available_quantities:
            if quantity_name in result or quantity_name not in GRADIENT_QUANTITIES:
                continue
            
            try:
                computed = compute_quantity_from_QUANTITIES(
                    quantity_name, result, dic_param, nbsatellite=4, separation=separation, 
                    verbose=verbose
                )
                if isinstance(computed, dict):
                    result.update(computed)
                else:
                    result[quantity_name] = computed
            except Exception as e:
                if verbose:
                    logger.error(f"Failed to compute {quantity_name}: {e}")
    
    else:
        raise ValueError(f"nbsatellite doit être 1 ou 4, reçu: {nbsatellite}")
    
    return result


def extract_trajectory_and_compute(dic_quant, y_pos, z_pos, dic_param=None, 
                                   laws=None, terms=None, quantities=None,
                                   nbsatellite=1, separation=1.0, verbose=False):
    """
    Extrait une trajectoire 1D et calcule les quantités requises pour les lois/termes.
    
    Paramètres:
    -----------
    dic_quant : dict
        Dictionnaire des données 1D (trajectoire simple ou multi-satellites)
    y_pos, z_pos : int
        Pour compatibilité (non utilisés si nbsatellite=4)
    dic_param : dict
        Paramètres
    laws, terms, quantities : list
        Configurations
    nbsatellite : int
        Nombre de satellites
    separation : float
        Séparation entre les satellites pour gradients
    verbose : bool
    """
    
    if dic_param is None:
        dic_param = {}
    
    # Les données sont déjà extraites (1D ou multi-satellites)
    trajectory_data = dic_quant.copy()
    
    if verbose:
        logger.info(f"Nbsatellite: {nbsatellite}")
        logger.info(f"Separation: {separation}")
    
    # Determine required quantities
    available_quantities = list_computable_quantities(
        trajectory_data, laws, terms, quantities, nbsatellite=nbsatellite
)    
    if verbose:
        logger.info(f"Required quantities from laws/terms: {available_quantities}")
    
    # Compute all available quantities
    return compute_all_available_quantities(
        trajectory_data, dic_param, available_quantities, nbsatellite=nbsatellite,
        separation=separation, verbose=verbose
    )


def display_results(traj_quantities, title="Results along trajectory"):
    """
    Affiche les résultats d'une trajectoire.
    """
    logger.info(f"\n{title}")
    logger.info("-" * 70)
    for key in sorted(traj_quantities.keys()):
        value = traj_quantities[key]
        if isinstance(value, np.ndarray):
            if value.ndim == 1:
                logger.info(f"  {key:20s}: shape={value.shape} | min={value.min():12.6e} | max={value.max():12.6e}")
            elif value.ndim == 2:
                logger.info(f"  {key:20s}: shape={value.shape} | min={value.min():12.6e} | max={value.max():12.6e}")
        elif isinstance(value, (int, float, np.number)):
            logger.info(f"  {key:20s}: {value:15.6e}")
