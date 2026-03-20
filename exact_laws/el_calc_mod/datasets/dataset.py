import logging
from ..grids import Grid


class Dataset:
    """Classe de données ayant pour attributs:
        - quantities[dict{name:np.array}]: contient les tableaux de valeurs des quantités
        - grid[Grid object]: contient les informations sur la grille
        - params[dict]: contient d'autres informations"""

    def __init__(self, params={}, quantities={}, grid=Grid()):
        """Initialisation of the class data"""
        self.params = params
        self.quantities = quantities  # dictionary 
        self.grid = grid  # grid

    def describ(self) -> str:
        message = f"\n\t - Params:"
        for k, v in self.params.items():
            message += f"\n\t\t - {k} = {v}"
        message += f"\n\t - Quantities:\n\t\t"
        message += "\n\t\t".join(self.quantities.keys())

        tmp = self.grid.describ().replace('\t', '\t\t')
        message += f"\n\t - Grid: {tmp}"
        return message

    def check(self, name: str) -> None:
        message = message = f"Check Dataset object {name}:"
        message += self.describ()
        logging.info(message)
