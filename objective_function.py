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
from pyomo.environ import *


def maximize_unused_capacities(model: ConcreteModel):
    cost_per_kw = 200                         #based on literature: cost per kw of installation for a DC charger
    model.objective_function = Objective(
        expr=(
            quicksum(model.Unused_capacity_new[c] for c in model.cs_cells)*cost_per_kw
        ),
        sense=maximize,

    )


"""Objective Function old"""
def minimize_waiting_and_maximize_unused_capacities(model: ConcreteModel):
    model.objective_function = Objective(
        expr=(
            quicksum(
                    model.n_wait[el] + model.n_wait_charge_next[el]
                    for el in model.charging_cells_key_set
            )*3
            - quicksum(model.Unused_capacity_new[c] for c in model.cs_cells)
        ),
        sense=minimize,

    )



def minimize_waiting(model: ConcreteModel):
    model.objective_function = Objective(
        expr=(
            quicksum(
                    model.n_wait[el] + model.n_wait_charge_next[el]
                    for el in model.charging_cells_key_set
            )
        ),
        sense=minimize,

    )


def multi_objective_function(model: ConcreteModel, alpha=2, beta=80):
    cost_per_kw = 200  # Kosten pro kW ungenutzter Kapazität


    # Maximierung der ungenutzten Kapazitäten (Zielfunktion 1)
    unused_capacity_term = quicksum(model.Unused_capacity_new[c] for c in model.cs_cells) * cost_per_kw

    # Minimierung der Warteschlangen (Zielfunktion 2)
    waiting_term = quicksum(
                    model.n_wait[el] + model.n_wait_charge_next[el]
                    for el in model.charging_cells_key_set
    )

    # Multi-Objective Funktion: Gewichtete Summe der beiden Ziele
    model.objective_function = Objective(
        expr=(
                alpha * unused_capacity_term - beta * waiting_term
        ),
        sense=maximize,  # Da wir das erste Ziel maximieren und das zweite minimieren wollen
    )
