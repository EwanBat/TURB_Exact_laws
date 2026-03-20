from exact_laws.preprocessing.process_on_oca_files import reformat_oca_files
import argparse
import logging
from datetime import datetime

version = "09/07/2022"

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--config-file", help="config file", default='example_input_process.ini')
parser.add_argument("-e", "--list-exactlaws", help="List available exact laws", action="store_true")
parser.add_argument("-t", "--list-terms", help="List available terms", action="store_true")
parser.add_argument("-q", "--list-quantities", help="List available quantities", action="store_true")
args = parser.parse_args()

logging.basicConfig(filename=f"reformat_oca_files_{datetime.now().strftime('%d%m%Y_%H%M%S')}.log", 
                    level=logging.INFO, 
                    format='%(asctime)s %(module)-12s %(levelname)-8s %(message)s',
                    )

if __name__ == "__main__":
    
    if args.list_quantities:
        from exact_laws.preprocessing.quantities import QUANTITIES
        print(list(QUANTITIES.keys()))
        exit(0)
    
    if args.list_terms:
        from exact_laws.el_calc_mod.terms import TERMS
        print(list(TERMS.keys()))
        exit(0)
    
    if args.list_exactlaws:
        from exact_laws.el_calc_mod.laws import LAWS
        print(list(LAWS.keys()))
        exit(0)
        
    logging.info(f"Run of {__file__} version {version}\n")
    reformat_oca_files(config_file=args.config_file)
    logging.info(f"Exit")
