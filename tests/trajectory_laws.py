# trajectory_laws.py
"""
Module pour calculer les termes des lois le long d'une trajectoire.
Utilise les termes calculés par trajectory_terms.py directement.
Pour les divergences, utilise un facteur commun par loi au lieu de calculer réellement.
"""

import numpy as np
import logging
from exact_laws.el_calc_mod.laws import LAWS
from exact_laws.el_calc_mod.terms import TERMS
from trajectory_terms import compute_all_terms_for_laws

logger = logging.getLogger(__name__)


# Facteurs de remplacement pour les divergences
# Un facteur unique par loi qui s'applique à TOUS les termes avec divergence de cette loi
# Format: {'law_name': factor}
# Pour le moment, tous les facteurs sont à 1
DIVERGENCE_REPLACEMENT_FACTORS = {
    'PP98': 1.0,
    'BG17': 1.0,
    'COR_Etot': 1.0,
    'Hallcor': 1.0,
    'IHallcor': 1.0,
    'ISS22Cgl': 1.0,
    'ISS22Gyr': 1.0,
    'ISS22Iso': 1.0,
    'SS21Iso': 1.0,
    'SS21Pol': 1.0,
    'SS22Cgl': 1.0,
    'SS22Gyr_flux': 1.0,
    'SS22Gyr_sources': 1.0,
    'SS22Gyr': 1.0,
    'SS22Iso_flux': 1.0,
    'SS22Iso_sources': 1.0,
    'SS22Iso': 1.0,
    'SS22Pol': 1.0,
    'TotSS21Iso': 1.0,
    'TotSS21Pol': 1.0,
    'TotSS22Iso': 1.0,
    'TotSS22Pol': 1.0,
}


def get_divergence_factor_for_law(law_name):
    """
    Retourne le facteur de remplacement de divergence pour une loi donnée.
    Ce facteur s'applique à TOUS les termes avec divergence de cette loi.
    
    Paramètres:
    -----------
    law_name : str
        Nom de la loi
    
    Retour:
    -------
    float : Facteur de divergence (par défaut 1.0)
    """
    return DIVERGENCE_REPLACEMENT_FACTORS.get(law_name, 1.0)


def apply_law_coefficients_and_factors(dic_terms, law_obj, law_name, dic_param, verbose=False):
    """
    Applique SEULEMENT le facteur de divergence aux termes.
    Les coefficients propres de la loi ne sont pas appliqués ici.
    
    Paramètres:
    -----------
    dic_terms : dict
        Dictionnaire des termes calculés
    law_obj : AbstractLaw
        Objet de la loi
    law_name : str
        Nom de la loi
    dic_param : dict
        Paramètres de simulation
    verbose : bool
    
    Retour:
    -------
    dict : Dictionnaire des termes de loi avec facteur de divergence appliqué
    """
    
    law_terms, coeffs = law_obj.terms_and_coeffs(dic_param)
    result = {}
    
    # Récupérer le facteur de divergence commun pour cette loi
    div_factor = get_divergence_factor_for_law(law_name)
    
    # Récupérer la liste des termes flux de la loi (ceux qui auront une divergence)
    law_flux_terms = set(law_obj.terms) if hasattr(law_obj, 'terms') else set()
    
    # Déterminer la taille des arrays (à partir du premier terme calculé)
    array_size = None
    for term_value in dic_terms.values():
        if isinstance(term_value, np.ndarray):
            array_size = term_value.shape
            break
    
    incomputable_terms = []
    applied_terms = []
    
    for coeff_key, coeff_value in coeffs.items():
        # Déterminer si c'est un terme de divergence
        is_divergence_term = coeff_key.startswith('div_')
        is_source_term = coeff_key.startswith('source_')
        
        # Chercher le terme correspondant
        if is_divergence_term:
            # Pour les divergences: div_flux_X → flux_X
            base_term = coeff_key.replace('div_', '')
            
            # Vérifier si c'est un terme flux (listés dans law_obj.terms)
            is_flux_term = base_term in law_flux_terms
            
            if base_term in dic_terms:
                term_value = dic_terms[base_term]
                # Appliquer le facteur de divergence (pas le coefficient)
                result[coeff_key] = div_factor * np.sum(term_value, axis=0)
                applied_terms.append(coeff_key)
                if verbose and is_flux_term:
                    logger.info(f"{coeff_key} (flux): div_factor={div_factor:.4f} applied")
            else:
                incomputable_terms.append((coeff_key, f"term '{base_term}' not computed"))
        
        elif is_source_term:
            # Termes sources
            # Pour les source: vérifier si c'est un gradient ou une divergence
            has_gradient = any(keyword in coeff_key for keyword in ['drdr', 'dr2', 'dx', 'dy', 'dz'])
            
            if has_gradient:
                # Terme source avec gradient/divergence: remplacer par array facteur
                if array_size is not None:
                    # Créer un array de la même taille rempli avec le facteur
                    result[coeff_key] = np.full(array_size, div_factor)
                else:
                    # Fallback: créer un scalaire
                    result[coeff_key] = div_factor
                applied_terms.append(coeff_key)
                if verbose:
                    logger.info(f"{coeff_key} (source+grad): div_factor={div_factor:.4f} applied")
            else:
                # Source normal sans gradient
                if coeff_key in dic_terms:
                    term_value = dic_terms[coeff_key]
                    # Appliquer SEULEMENT le facteur de divergence
                    result[coeff_key] = div_factor * term_value
                    applied_terms.append(coeff_key)
                    if verbose:
                        logger.info(f"{coeff_key} (source): div_factor={div_factor:.4f} applied")
                else:
                    incomputable_terms.append((coeff_key, "term not computed"))
        
        else:
            # Terme simple (pas de div_ ni source_)
            if coeff_key in dic_terms:
                term_value = dic_terms[coeff_key]
                # Appliquer SEULEMENT le facteur de divergence
                result[coeff_key] = div_factor * term_value
                applied_terms.append(coeff_key)
            else:
                incomputable_terms.append((coeff_key, "term not computed"))
    
    if verbose:
        if applied_terms:
            logger.info(f"Applied terms ({len(applied_terms)}):")
            for term in applied_terms:
                logger.info(f"  {term}")
        if incomputable_terms:
            logger.info(f"Incomputable terms ({len(incomputable_terms)}):")
            for term, reason in incomputable_terms:
                logger.info(f"  {term}: {reason}")
    
    return result, coeffs  # ← Retourner aussi les coefficients originaux


def compute_laws_terms_with_coefficients(dic_terms, dic_param, laws=None, verbose=False):
    """
    Calcule les termes des lois avec coefficients appliqués.
    Retourne un dictionnaire plat des termes de lois (div_flux_*, source_*, etc.).
    
    Note: Puisque les termes sont identiques pour toutes les lois (le facteur de divergence
    est appliqué à tous les termes indépendamment de la loi), les termes ne sont calculés
    qu'une seule fois pour la première loi valide.
    
    Paramètres:
    -----------
    dic_terms : dict
        Dictionnaire des termes de base calculés
    dic_param : dict
        Paramètres de simulation
    laws : list[str]
        Liste des lois
    verbose : bool
    
    Retour:
    -------
    dict : Dictionnaire plat des termes de lois avec coefficients {{term_key: value}}
    """
    
    if laws is None:
        laws = []
    
    # Ajouter rho_mean aux paramètres si absent
    if "rho_mean" not in dic_param:
        dic_param["rho_mean"] = 1.0
    
    result = {}
    dic_coefficients = {}
    
    # Calculer les termes une seule fois (ils sont identiques pour toutes les lois)
    computed = False
    for law_name in laws:
        if law_name not in LAWS:
            if verbose:
                logger.warning(f"Law '{law_name}' not found")
            continue
        
        if verbose:
            logger.info(f"Processing law: {law_name}")
        
        try:
            law_obj = LAWS[law_name]
            
            # Appliquer les coefficients et facteurs de divergence
            law_terms_dict, law_coeffs = apply_law_coefficients_and_factors(
                dic_terms, law_obj, law_name, dic_param, verbose=verbose
            )
            
            # Pour la première loi, stocker les termes (ils seront identiques pour les autres)
            if not computed:
                result = law_terms_dict
                computed = True
            
            # Ajouter les coefficients avec clés formatées
            for term_key, coeff_value in law_coeffs.items():
                dic_coefficients[f"{law_name}_{term_key}"] = coeff_value
            
            if verbose:
                logger.info(f"Added {len(law_terms_dict)} terms for {law_name}")
        
        except Exception as e:
            logger.error(f"Failed to process {law_name}: {e}")
    
    return result, dic_coefficients


def extract_trajectory_and_compute_law_terms(dic_quant, y_pos, z_pos, dic_param=None, 
                                            laws=None, verbose=False):
    """
    Extrait une trajectoire 1D et calcule les termes des lois.
    
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
        Lois à calculer (terme utilisé pour récupérer les coefficients)
    verbose : bool
    
    Retour:
    -------
    tuple : (dic_terms, dic_law_terms, dic_coefficients)
        - dic_terms: dictionnaire des termes de base
        - dic_law_terms: dictionnaire plat des termes de lois avec coefficients {terme: array}
        - dic_coefficients: dictionnaire des coefficients de loi {loi_terme: coeff}
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
        logger.info(f"Trajectory at (y={y_pos}, z={z_pos})")
        array_keys = [k for k in trajectory_data.keys() if isinstance(trajectory_data[k], np.ndarray)]
        if array_keys:
            logger.info(f"Data shape: {trajectory_data[array_keys[0]].shape}")
    
    # Compute all terms for laws
    dic_terms = compute_all_terms_for_laws(trajectory_data, dic_param, laws, verbose)
    
    # Compute law terms with coefficients
    dic_law_terms, dic_coefficients = compute_laws_terms_with_coefficients(dic_terms, dic_param, laws, verbose)
    
    return dic_terms, dic_law_terms, dic_coefficients  # ← 3e retour


def display_law_terms_results(dic_law_terms, title="Law Terms Results"):
    """
    Affiche les résultats des termes de lois le long d'une trajectoire.
    
    Paramètres:
    -----------
    dic_law_terms : dict
        Dictionnaire plat des termes de lois {terme: array}
    title : str
        Titre de l'affichage
    """
    logger.info(f"\n{title}")
    logger.info("-" * 70)
    
    for term_key in sorted(dic_law_terms.keys()):
        value = dic_law_terms[term_key]
        if isinstance(value, np.ndarray) and value.ndim == 1:
            if value.size > 0:
                logger.info(f"  {term_key:40s}: min={value.min():12.6e} | max={value.max():12.6e} | mean={value.mean():12.6e}")
            else:
                logger.info(f"  {term_key:40s}: empty array")
        elif isinstance(value, np.ndarray):
            logger.info(f"  {term_key:40s}: shape={value.shape} | mean={value.mean():12.6e}")
        elif isinstance(value, (int, float, np.number)):
            logger.info(f"  {term_key:40s}: {value:15.6e}")
        else:
            logger.info(f"  {term_key:40s}: {value}")