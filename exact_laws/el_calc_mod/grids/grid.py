import logging

class Grid:
    """Classe contenant les informations nécessaires sur une grille de données
    Mère de la classe contenant les informations nécessaires sur la grille finale (grille d'échelles)
    """
    def __init__(self, N=[], L=[], c=[], axis = [], coords = {}):
        self.axis = axis
        self.N = N  # Echantillonnage (Nombre de points)
        self.L = L  # Largeur totale réelle
        self.c = c  # real case length
        self.coords = coords
    
    def describ(self):
        message = f"\n\t - axis: {self.axis}"
        message += f"\n\t - N: {self.N}"
        message += f"\n\t - L: {self.L}"
        message += f"\n\t - c: {self.c}"
        return message

    def check(self, name=''):
        """Display the principal parameter of the grid object"""
        message = f"Check Grid object {name}:"
        message = self.describ()
        logging.info(message)