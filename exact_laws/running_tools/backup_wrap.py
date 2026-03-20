import os
import pickle as pkl
import logging

class Backup():
    def __init__(self):
        pass
    
    def configure(self,config,time,rank):
        if config is None:
            if rank == 0:
                os.mkdir(f"./backup_{time.strftime('%d%m%Y_%H%M%S')}")  # creation of a recording folder
            self.folder = f"./backup_{time.strftime('%d%m%Y_%H%M%S')}/"
            self.already = False
        else: 
            self.folder = config
            self.already = True
        
    def save(self,object,name,rank='',state=''):
        logging.info(f"Save {name} {state} in folder {self.folder} INIT")
        filename = f"{self.folder}/{name}_rk{rank}.pkl"
        with open(filename, "wb") as f:
            pkl.dump(object, f)
        logging.info(f"Save {name} END")
    
    
    def download(self,name,rank=''):
        logging.info(f"Download {name} from folder {self.folder} INIT")
        filename = f"{self.folder}/{name}_rk{rank}.pkl"
        with open(filename, "rb") as f:
            output = pkl.load(f)
        logging.info(f"Download {name} END")
        return output
       
