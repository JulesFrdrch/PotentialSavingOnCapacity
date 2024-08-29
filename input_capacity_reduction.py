import numpy as np
from pyomo.core import ConcreteModel
from scipy.stats import norm
from pyomo.environ import *
import numpy as np
from utils import *
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
from openpyxl.workbook import Workbook
import matplotlib.pyplot as plt
# from _import_geographic_data import *
from matplotlib.offsetbox import AnchoredText
# from _import_optimization_files import *
import time
from pyomo.core.util import quicksum
from datetime import datetime
from pyomo.environ import ConcreteModel, Objective, quicksum, minimize, maximize




def reduce_cell_capacities(model: ConcreteModel, reduction_factor: float):
    """
    Reduziert die Kapazitäten der Zellen um einen gegebenen Faktor.

    Parameters:
    model (ConcreteModel): Das Pyomo-Modell.
    reduction_factor (float): Der Faktor, um den die Kapazitäten reduziert werden sollen (z.B. 0.8 für eine Reduktion um 20%).
    """
    # Überprüfen, ob der Reduktionsfaktor zwischen 0 und 1 liegt
    if not 0 <= reduction_factor <= 1:
        raise ValueError("Reduktionsfaktor muss zwischen 0 und 1 liegen")

    # Reduzieren der Kapazitäten
    for c in model.nb_cell:
        original_capacity = model.cell_charging_cap[c]
        model.cell_charging_cap[c] = original_capacity * reduction_factor

    print(f"Kapazitäten der Zellen wurden um {reduction_factor * 100}% reduziert.")


def reduce_selected_cell_capacities(model: ConcreteModel, cell_ids: list, reduction_factor: float):
    """
    Reduziert die Kapazitäten bestimmter Zellen um einen gegebenen Faktor.

    Parameters:
    model (ConcreteModel): Das Pyomo-Modell.
    cell_ids (list): Liste der IDs der Zellen, deren Kapazität reduziert werden soll.
    reduction_factor (float): Der Faktor, um den die Kapazitäten reduziert werden sollen (z.B. 0.8 für eine Reduktion um 20%).
    """
    # Überprüfen, ob der Reduktionsfaktor zwischen 0 und 1 liegt
    if not 0 <= reduction_factor <= 1:
        raise ValueError("Reduktionsfaktor muss zwischen 0 und 1 liegen")

    # Reduzieren der Kapazitäten für ausgewählte Zellen
    for c in cell_ids:
        if c in model.nb_cell:
            original_capacity = model.cell_charging_cap[c]
            model.cell_charging_cap[c] = original_capacity * reduction_factor
            print(f"Kapazität der Zelle {c} wurde auf {model.cell_charging_cap[c]} reduziert.")
        else:
            print(f"Zelle {c} existiert nicht im Modell.")


def set_selected_cells_to_zero_capacity(model: ConcreteModel, zero_cell_ids: list):
    """
    Setzt die Kapazität ausgewählter Zellen auf 0 und ändert has_cs von True auf False.

    Parameters:
    model (ConcreteModel): Das Pyomo-Modell.
    cell_ids (list): Liste der IDs der Zellen, deren Kapazität auf 0 gesetzt werden soll.
    """
    # Setzen der Kapazitäten auf 0 und Ändern von has_cs
    for c in zero_cell_ids:
        if c in model.nb_cell:
            model.cell_charging_cap[c] = 0
            if hasattr(model, 'has_cs'):
                model.has_cs[c] = False
            print(f"Kapazität der Zelle {c} wurde auf 0 gesetzt und has_cs auf False geändert.")
        else:
            print(f"Zelle {c} existiert nicht im Modell.")


