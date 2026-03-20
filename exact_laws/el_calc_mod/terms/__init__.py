import importlib
import os

here = os.path.dirname(os.path.realpath(__file__))

def load_term(name):
    mod = importlib.import_module(f"exact_laws.el_calc_mod.terms.{name}", "*")
    return mod.load()

def load_all():
    terms = [f[:-3] for f in os.listdir(here) if f[-3:] == '.py' and f != '__init__.py' and f != 'abstract_term.py']
    return {term: load_term(term) for term in terms}


TERMS = load_all()
