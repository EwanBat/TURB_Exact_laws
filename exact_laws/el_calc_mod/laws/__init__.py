import importlib
import os


here = os.path.dirname(os.path.realpath(__file__))

def load_law(name):
    mod = importlib.import_module(f"exact_laws.el_calc_mod.laws.{name}", "*")
    return mod.load()


def load_all():
    laws = [f[:-3] for f in os.listdir(here) if f[-3:] == '.py' and f != '__init__.py' and f != 'abstract_law.py']
    return {law: load_law(law) for law in laws}


LAWS = load_all()
