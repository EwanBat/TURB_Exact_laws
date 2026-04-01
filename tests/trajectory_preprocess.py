"""
Module de prétraitement pour les trajectoires satellites.
Encapsule le chargement des données OCA et la récupération des quantités requises.
Support pour trajectoires personnalisées - indexage simple dans le cube.
"""

import logging
import numpy as np
import h5py
import configparser
from pathlib import Path
from datetime import datetime
from typing import Callable, Tuple

from exact_laws.preprocessing.process_on_oca_files import (
    extract_quantities_from_OCA_file,
    extract_simu_param_from_OCA_file
)

# ========== TRAJECTORY DEFINITIONS ==========

def trajectory_linear_x(t: np.ndarray, y_pos: int, z_pos: int, 
                        N: np.ndarray) -> np.ndarray:
    """
    Trajectoire linéaire le long de l'axe x (indices).
    
    Paramètres:
    -----------
    t : np.ndarray
        Paramètre de trajectoire (0 à N[0]-1)
    y_pos : int
        Position fixe sur y (indice)
    z_pos : int
        Position fixe sur z (indice)
    N : np.ndarray
        Dimensions de la grille (N[0], N[1], N[2])
    
    Retour:
    -------
    np.ndarray
        Points de la trajectoire (n_points, 3) avec [x, y, z] indices
    """
    x = t
    y = np.full_like(t, y_pos, dtype=int)
    z = np.full_like(t, z_pos, dtype=int)
    
    # Clip aux limites de la grille
    x = np.clip(x, 0, N[0]-1).astype(int)
    y = np.clip(y, 0, N[1]-1).astype(int)
    z = np.clip(z, 0, N[2]-1).astype(int)
    
    return np.array([x, y, z]).T


def trajectory_circular_xy(t: np.ndarray, radius: int, center_y: int, center_z: int,
                           N: np.ndarray) -> np.ndarray:
    """
    Trajectoire circulaire dans le plan xy autour d'un centre (indices).
    
    Paramètres:
    -----------
    t : np.ndarray
        Paramètre de trajectoire (indice 0 à N[0]-1 pour une trajectoire complète)
    radius : int
        Rayon du cercle (en indices de grille)
    center_y : int
        Centre en y (indice)
    center_z : int
        Position fixe en z (indice)
    N : np.ndarray
        Dimensions de la grille
    
    Retour:
    -------
    np.ndarray
        Points de la trajectoire (n_points, 3) avec [x, y, z] indices
    """
    # Paramètre d'angle normalisé (0 à 2π)
    theta = 2 * np.pi * t / N[0]
    
    y = center_y + radius * np.cos(theta)
    x = radius * np.sin(theta)
    z = np.full_like(t, center_z, dtype=int)
    
    # Clip aux limites
    x = np.clip(x, 0, N[0]-1).astype(int)
    y = np.clip(y, 0, N[1]-1).astype(int)
    z = np.clip(z, 0, N[2]-1).astype(int)
    
    return np.array([x, y, z]).T


def trajectory_helical(t: np.ndarray, pitch: int, radius: int, center_y: int, center_z: int,
                       N: np.ndarray) -> np.ndarray:
    """
    Trajectoire hélicoïdale spiralant autour d'un centre (indices).
    
    Paramètres:
    -----------
    t : np.ndarray
        Paramètre de trajectoire (indice 0 à N[0]-1)
    pitch : int
        Pas de l'hélice (progression en x par tour complète)
    radius : int
        Rayon de l'hélice (en indices)
    center_y : int
        Centre en y (indice)
    center_z : int
        Centre en z (indice)
    N : np.ndarray
        Dimensions de la grille
    
    Retour:
    -------
    np.ndarray
        Points de la trajectoire (n_points, 3) avec [x, y, z] indices
    """
    # Progression en x avec pitch
    x = (t / N[0]) * pitch
    
    # Paramètre d'angle normalisé
    theta = 2 * np.pi * t / N[0]
    
    y = center_y + radius * np.cos(theta)
    z = center_z + radius * np.sin(theta)
    
    # Clip aux limites
    x = np.clip(x, 0, N[0]-1).astype(int)
    y = np.clip(y, 0, N[1]-1).astype(int)
    z = np.clip(z, 0, N[2]-1).astype(int)
    
    return np.array([x, y, z]).T


def trajectory_diagonal(t: np.ndarray, N: np.ndarray) -> np.ndarray:
    """
    Trajectoire diagonale à travers le cube.
    
    Paramètres:
    -----------
    t : np.ndarray
        Paramètre de trajectoire (0 à N[0]-1)
    N : np.ndarray
        Dimensions de la grille
    
    Retour:
    -------
    np.ndarray
        Points de la trajectoire (n_points, 3) avec [x, y, z] indices
    """
    # Progression en x, y, z proportionnelle
    x = t
    y = (t * N[1] / N[0]).astype(int)
    z = (t * N[2] / N[0]).astype(int)
    
    # Clip aux limites
    x = np.clip(x, 0, N[0]-1).astype(int)
    y = np.clip(y, 0, N[1]-1).astype(int)
    z = np.clip(z, 0, N[2]-1).astype(int)
    
    return np.array([x, y, z]).T


# ========== UTILITY FUNCTIONS ==========

def setup_logging(config_name="trajectory_preprocess"):
    """Crée un logger avec timestamp."""
    log_filename = f"{config_name}_{datetime.now().strftime('%d%m%Y_%H%M%S')}.log"
    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-7s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging


def load_oca_data(input_folder, cycle, sim_type):
    """
    Charge toutes les données OCA requises.
    
    Paramètres:
    -----------
    input_folder : str
        Chemin du dossier contenant les fichiers 3Dfields_*.h5
    cycle : str
        Nom du cycle (ex: "cycle_0")
    sim_type : str
        Type de simulation (ex: "CGL5") pour identifier les paramètres
    
    Retour:
    -------
    tuple : (dic_datas, dic_param) dictionnaires des quantités et paramètres
    """
    logging.info("\n" + "="*70)
    logging.info("LOADING OCA DATA")
    logging.info("="*70)
    
    dic_datas = {}
    dic_param = {}
    
    # Load velocity
    with h5py.File(f"{input_folder}/3Dfields_v.h5", "r") as fv:
        param_key = "3Dgrid" if sim_type.endswith(("CGL3", "CGL5")) else "Simulation_Parameters"
        dic_param = extract_simu_param_from_OCA_file(fv, dic_param, param_key)
        (dic_datas["vx"],
         dic_datas["vy"],
         dic_datas["vz"]) = extract_quantities_from_OCA_file(fv, ["vx", "vy", "vz"], cycle)
    logging.info(f"  [OK] Velocity loaded:         {dic_datas['vx'].shape}")
    
    # Load density
    with h5py.File(f"{input_folder}/3Dfields_rho.h5", "r") as frho:
        dic_datas["rho"] = extract_quantities_from_OCA_file(frho, ["rho"], cycle)[0]
    logging.info(f"  [OK] Density loaded:          {dic_datas['rho'].shape}")
    
    # Load magnetic field
    with h5py.File(f"{input_folder}/3Dfields_b.h5", "r") as fb:
        (dic_datas["bx"],
         dic_datas["by"],
         dic_datas["bz"]) = extract_quantities_from_OCA_file(fb, ["bx", "by", "bz"], cycle)
    logging.info(f"  [OK] Magnetic field loaded:   {dic_datas['bx'].shape}")
    
    # Load pressure components
    with h5py.File(f"{input_folder}/3Dfields_pi.h5", "r") as fp:
        (dic_datas["ppar"],
         dic_datas["pperp"]) = extract_quantities_from_OCA_file(fp, ["pparli", "pperpi"], cycle)
        dic_datas["ppar"] /= 2
        dic_datas["pperp"] /= 2
    logging.info(f"  [OK] Pressure loaded:         {dic_datas['ppar'].shape}")
    
    # Try loading force amplitude (optional)
    try:
        with h5py.File(f"{input_folder}/3Dfields_forcl_ampl.h5", "r") as ff:
            (dic_datas["fp"],
             dic_datas["fm"]) = extract_quantities_from_OCA_file(ff, ["forcl_ampl_plus", "forcl_ampl_mins"], cycle)
        logging.info(f"  [OK] Force amplitude loaded:  {dic_datas['fp'].shape}")
    except:
        logging.warning("  [SKIP] Force amplitude not loaded")
    
    # Print summary
    logging.info("\n" + "-"*70)
    logging.info("DATA LOADING SUMMARY")
    logging.info("-"*70)
    logging.info(f"  Grid dimensions (N):  {dic_param['N']}")
    logging.info(f"  Domain size (L):      {dic_param['L']}")
    logging.info(f"  Cell spacing (c):     {dic_param['c']}")
    logging.info(f"  Data fields:          {len(dic_datas)} fields loaded")
    for field in sorted(dic_datas.keys()):
        logging.info(f"    - {field}")
    
    return dic_datas, dic_param


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

def extract_quantities_along_trajectory(dic_datas: dict, trajectory: np.ndarray, 
                                       nbsatellite: int = 1, separation: int = 2) -> dict:
    """
    Extrait les quantités le long d'une ou plusieurs trajectoires (indices).
    
    Paramètres:
    -----------
    dic_datas : dict
        Dictionnaire des quantités 3D
    trajectory : np.ndarray
        Trajectoire centrale (n_points, 3) en indices
    nbsatellite : int
        Nombre de satellites (1 ou 4)
    separation : int
        Séparation entre les satellites en indices (utilisé si nbsatellite=4)
    
    Retour:
    -------
    dict : Quantités 1D extraites le long de la/des trajectoire(s)
           Si nbsatellite=1: {quantity_name: (n_points,)}
           Si nbsatellite=4: {quantity_name: {sat_0: (n_points,), sat_1: ..., sat_2: ..., sat_3: ...}}
    """
    n_points = len(trajectory)
    
    if nbsatellite == 1:
        # = ==== SINGLE SATELLITE ====
        trajectory_data = {}
        
        for key in dic_datas.keys():
            if isinstance(dic_datas[key], np.ndarray) and dic_datas[key].ndim == 3:
                # Extraire les valeurs le long de la trajectoire
                trajectory_data[key] = np.array([
                    dic_datas[key][int(trajectory[i, 0]), int(trajectory[i, 1]), int(trajectory[i, 2])]
                    for i in range(n_points)
                ])
            else:
                trajectory_data[key] = dic_datas[key]
    
    elif nbsatellite == 4:
        # ========== QUAD SATELLITE =========
        # Définir les 4 satellites en carré centré
        half_sep = separation / 2
        satellites = {
            'sat_0': np.array([half_sep, half_sep, 0]),   # Centre-avant
            'sat_1': np.array([half_sep, -half_sep, 0]),  # Avant-bas
            'sat_2': np.array([-half_sep, half_sep, 0]),  # Arrière-haut
            'sat_3': np.array([-half_sep, -half_sep, 0])  # Arrière-bas
        }
        
        # Générer les 4 trajectoires
        trajectories = {}
        for sat_name, offset in satellites.items():
            traj = trajectory + offset
            trajectories[sat_name] = np.clip(traj, 0, 255).astype(int)
        
        # Extraire les données pour chaque satellite
        trajectory_data = {}
        
        for key in dic_datas.keys():
            if isinstance(dic_datas[key], np.ndarray) and dic_datas[key].ndim == 3:
                trajectory_data[key] = {}
                
                for sat_name, traj in trajectories.items():
                    # Extraire les valeurs le long de cette trajectoire satellite
                    trajectory_data[key][sat_name] = np.array([
                        dic_datas[key][int(traj[i, 0]), int(traj[i, 1]), int(traj[i, 2])]
                        for i in range(n_points)
                    ])
            else:
                # Garder les scalaires comme avant
                trajectory_data[key] = dic_datas[key]
    
    else:
        raise ValueError(f"nbsatellite doit être 1 ou 4, got {nbsatellite}")
    
    return trajectory_data

def extract_coordinates_along_trajectory(trajectory: np.ndarray, dic_param: dict) -> dict:
    """
    Extrait les coordonnées physiques (x, y, z) le long de la trajectoire à partir des indices.
    
    Paramètres:
    -----------
    trajectory : np.ndarray
        Trajectoire en indices (n_points, 3)
    dic_param : dict
        Dictionnaire contenant les paramètres de la simulation, notamment 'lx', 'ly', 'lz'
    
    Retour:
    -------
    dict : Dictionnaire avec les coordonnées physiques le long de la trajectoire
           {'x': (n_points,), 'y': (n_points,), 'z': (n_points,)}
    """
    
    lx = np.arange(0, dic_param['N'][0]) * dic_param['c'][0]
    ly = np.arange(0, dic_param['N'][1]) * dic_param['c'][1]
    lz = np.arange(0, dic_param['N'][2]) * dic_param['c'][2]

    x = lx[trajectory[:, 0]]
    y = ly[trajectory[:, 1]]
    z = lz[trajectory[:, 2]]

    dic_param['lx'] = lx
    dic_param['ly'] = ly
    dic_param['lz'] = lz

def preprocess_trajectory_from_ini(ini_file, 
                                   trajectory_func: Callable = None,
                                   trajectory_kwargs: dict = None,
                                   input_folder: str = "data_oca",
                                   verbose: bool = True):
    """
    Charge la configuration depuis un fichier INI et prétraite le long d'une trajectoire.
    
    Paramètres:
    -----------
    ini_file : str
        Chemin du fichier .ini de configuration (ex: "traj_satellite.ini")
    trajectory_func : Callable, optionnel
        Fonction qui renvoie une trajectoire: f(t, N, **trajectory_kwargs) -> np.ndarray (n_points, 3)
        Si None, utilise trajectory_linear_x avec y_pos=100, z_pos=100
    trajectory_kwargs : dict, optionnel
        Arguments additionnels pour trajectory_func (tous en indices, pas en unités physiques)
    input_folder : str
        Chemin du dossier contenant les données OCA
    verbose : bool
        Afficher les informations détaillées
    
    Retour:
    -------
    dict : Résultats contenant:
        - 'config': Configuration chargée
        - 'required_quantities': Quantités requises
        - 'dic_datas': Données brutes extraites le long de la trajectoire (1D)
        - 'dic_param': Paramètres de simulation
        - 'trajectory': Points de la trajectoire utilisée (indices)
    """
    
    # Setup logging
    setup_logging(Path(ini_file).stem)
    
    if verbose:
        logging.info("\n" + "="*70)
        logging.info(f"PREPROCESSING TRAJECTORY FROM {ini_file}")
        logging.info("="*70)
    
    # Vérifier que le fichier existe
    ini_path = Path(ini_file)
    if not ini_path.exists():
        raise FileNotFoundError(f"Fichier de configuration non trouvé: {ini_file}")
    
    if verbose:
        logging.info(f"Fichier INI trouvé: {ini_path.absolute()}")
    
    # Load configuration from INI file
    if verbose:
        logging.info("\nLoading configuration from INI file...")
    
    try:
        laws, terms, quantities, physical_params = load_config_from_ini(ini_file)
    except KeyError as e:
        logging.error(f"Clé manquante dans le fichier INI: {e}")
        raise
    except Exception as e:
        logging.error(f"Erreur lors de la lecture du fichier INI: {e}")
        raise
    
    # Load additional parameters from INI
    config = configparser.ConfigParser()
    
    try:
        config.read(ini_file)
    except Exception as e:
        logging.error(f"Erreur lors du parsing du fichier INI: {e}")
        raise
    
    # Vérifier les sections requises
    required_sections = ['RUN_PARAMS', 'INPUT_DATA']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Section requise manquante dans le fichier INI: [{section}]")
    
    try:
        nbsatellite = config["RUN_PARAMS"].getint("nbsatellite", 1)
        dist_satellite = config["RUN_PARAMS"].getfloat("dist_satellite", 1)
        input_folder = config["INPUT_DATA"].get("path", input_folder)
        cycle = config["INPUT_DATA"].get("cycle", "cycle_0")
        sim_type = config["INPUT_DATA"].get("sim_type", "OCA_CGL5").split("_")[-1]
    except Exception as e:
        logging.error(f"Erreur lors de la lecture des paramètres: {e}")
        raise
    
    if verbose:
        logging.info(f"  Laws:            {laws}")
        logging.info(f"  Terms:           {terms}")
        logging.info(f"  Quantities:      {quantities}")
        logging.info(f"  Physical params: {physical_params}")
        logging.info(f"  Nbsatellite:     {nbsatellite}")
        logging.info(f"  Dist satellite:  {dist_satellite}")
        logging.info(f"  Input folder:    {input_folder}")
        logging.info(f"  Cycle:           {cycle}")
        logging.info(f"  Sim type:        {sim_type}")
    
    # Load OCA data (3D)
    try:
        dic_datas_3d, dic_param = load_oca_data(input_folder, cycle, sim_type)
    except Exception as e:
        logging.error(f"Erreur lors du chargement des données OCA: {e}")
        raise
    
    # Determine required quantities
    if verbose:
        logging.info("\n" + "-"*70)
        logging.info("DETERMINING REQUIRED QUANTITIES")
        logging.info("-"*70)
        logging.info(f"  Nbsatellite = {nbsatellite} -> gradients {'enabled' if nbsatellite > 1 else 'disabled'}")
    
    # Default trajectory if none provided
    if trajectory_func is None:
        trajectory_func = trajectory_linear_x
        trajectory_kwargs = {'y_pos': 100, 'z_pos': 100}
    
    if trajectory_kwargs is None:
        trajectory_kwargs = {}
    
    # Generate trajectory
    if verbose:
        logging.info("\n" + "-"*70)
        logging.info("GENERATING TRAJECTORY")
        logging.info("-"*70)
        logging.info(f"  Trajectory function: {trajectory_func.__name__}")
        logging.info(f"  Trajectory kwargs:   {trajectory_kwargs}")
        logging.info(f"  Number of satellites: {nbsatellite}")
    
    try:
        t = np.arange(dic_param['N'][0])
        trajectory = trajectory_func(t, N=dic_param['N'], **trajectory_kwargs)
    except Exception as e:
        logging.error(f"Erreur lors de la génération de la trajectoire: {e}")
        raise
    
    if verbose:
        logging.info(f"  Trajectory points: {len(trajectory)}")
        if nbsatellite == 1:
            logging.info(f"  First point (indices): {trajectory[0]}")
            logging.info(f"  Last point (indices):  {trajectory[-1]}")
        else:
            logging.info(f"  Central trajectory:")
            logging.info(f"    First point (indices): {trajectory[0]}")
            logging.info(f"    Last point (indices):  {trajectory[-1]}")
            logging.info(f"  Separation between satellites: {dist_satellite}")
    
    # Extract quantities along trajectory
    if verbose:
        logging.info("\n" + "-"*70)
        logging.info("EXTRACTING DATA ALONG TRAJECTORY")
        logging.info("-"*70)
    
    dic_datas = extract_quantities_along_trajectory(
        dic_datas_3d, 
        trajectory, 
        nbsatellite=nbsatellite, 
        separation=int(dist_satellite)
    )

    extract_coordinates_along_trajectory(trajectory, dic_param) # Physical coordinates to dic_param
    
    # Add means if available
    if 'ppar' in dic_datas:
        dic_param["meanppar"] = np.mean(dic_datas['ppar'],axis=0) if nbsatellite == 1 else {sat: np.mean(dic_datas['ppar'][sat]) for sat in dic_datas['ppar']}
    if 'pperp' in dic_datas:
        dic_param["meanpperp"] = np.mean(dic_datas['pperp'],axis=0) if nbsatellite == 1 else {sat: np.mean(dic_datas['pperp'][sat]) for sat in dic_datas['pperp']}
    if 'rho' in dic_datas:
        dic_param["rho_mean"] = np.mean(dic_datas['rho'],axis=0) if nbsatellite == 1 else {sat: np.mean(dic_datas['rho'][sat]) for sat in dic_datas['rho']}

    if verbose:
        logging.info(f"  Extracted {len(dic_datas)} quantities")
        if nbsatellite == 1:
            logging.info(f"  Trajectory length: {len(trajectory)} points")
            for key in sorted(dic_datas.keys()):
                if isinstance(dic_datas[key], np.ndarray):
                    logging.info(f"    - {key}: shape={dic_datas[key].shape} | min={dic_datas[key].min():.3e} | max={dic_datas[key].max():.3e}")
        else:
            logging.info(f"  Number of satellites: {nbsatellite}")
            logging.info(f"  Trajectory length: {len(trajectory)} points")
            for key in sorted(dic_datas.keys()):
                if isinstance(dic_datas[key], dict):
                    logging.info(f"    - {key}:")
                    for sat_name in sorted(dic_datas[key].keys()):
                        data = dic_datas[key][sat_name]
                        logging.info(f"        {sat_name}: shape={data.shape} | min={data.min():.3e} | max={data.max():.3e}")
                elif isinstance(dic_datas[key], np.ndarray):
                    logging.info(f"    - {key}: shape={dic_datas[key].shape} | min={dic_datas[key].min():.3e} | max={dic_datas[key].max():.3e}")
        logging.info("="*70 + "\n")
    
    return {
        'config': {
            'laws': laws,
            'terms': terms,
            'quantities': quantities,
            'physical_params': physical_params,
            'nbsatellite': nbsatellite,
            'dist_satellite': dist_satellite
        },
        'dic_datas': dic_datas,  # Données 1D extraites le long de la/des trajectoire(s)
        'dic_param': dic_param,
        'trajectory': trajectory
    }
