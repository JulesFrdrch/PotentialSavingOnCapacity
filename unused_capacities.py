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


"""unused nur mit c"""


def charging_at_restarea_mit_unused_capacities(model: ConcreteModel, t, c):
    return (
        quicksum(
            (model.E_charge1[t, c, f] + model.E_charge2[t, c, f] + model.E_charge3[t, c, f])
            for f in model.nb_fleet
            if (t, c, f) in model.charging_cells_key_set
        )
        <= (model.cell_charging_cap[c] - model.Unused_capacity_new[c]) * model.time_resolution * model.ladewirkungsgrad
    )


"""Cell 0: Unused Capacity is not set.
Cell 1: Unused Capacity is not set.
Cell 2: Unused Capacity = 0.0
Cell 3: Unused Capacity = 0.0
Cell 4: Unused Capacity = 0.0
Cell 5: Unused Capacity = 0.0
Cell 6: Unused Capacity is not set.
Cell 7: Unused Capacity = 0.0
Cell 8: Unused Capacity is not set.
Cell 9: Unused Capacity = 0.0
Cell 10: Unused Capacity = 0.0
Cell 11: Unused Capacity is not set.
Cell 12: Unused Capacity = 0.0
Cell 13: Unused Capacity = 0.0
Cell 14: Unused Capacity is not set.
Cell 15: Unused Capacity is not set.
Cell 16: Unused Capacity = 0.0
Cell 17: Unused Capacity is not set.
Cell 18: Unused Capacity = 0.0
Cell 19: Unused Capacity = 0.0
Cell 20: Unused Capacity is not set.
Cell 21: Unused Capacity = 0.0
Cell 22: Unused Capacity is not set.
Cell 23: Unused Capacity = 0.0
Cell 24: Unused Capacity is not set.
Cell 25: Unused Capacity = 0.0
Cell 26: Unused Capacity is not set.
Cell 27: Unused Capacity is not set.
Cell 28: Unused Capacity is not set.
Cell 29: Unused Capacity = 0.0
Cell 30: Unused Capacity = 0.0
Cell 31: Unused Capacity is not set.
Cell 32: Unused Capacity is not set.
Cell 33: Unused Capacity is not set.
Cell 34: Unused Capacity = 0.0
Cell 35: Unused Capacity is not set.
Cell 36: Unused Capacity is not set.
Cell 37: Unused Capacity = 0.0
Cell 38: Unused Capacity is not set.
Cell 39: Unused Capacity is not set.
Cell 40: Unused Capacity is not set.
Cell 41: Unused Capacity is not set.
Cell 42: Unused Capacity = 0.0
Cell 43: Unused Capacity is not set.
Cell 44: Unused Capacity is not set.
Cell 45: Unused Capacity is not set.
Cell 46: Unused Capacity = 0.0
Cell 47: Unused Capacity = 0.0
Cell 48: Unused Capacity = 0.0
Cell 49: Unused Capacity is not set.
Cell 50: Unused Capacity is not set.
Cell 51: Unused Capacity is not set.
Cell 52: Unused Capacity is not set.
Cell 53: Unused Capacity = 0.0
Cell 54: Unused Capacity = 0.0
Cell 55: Unused Capacity = 0.0
Cell 56: Unused Capacity is not set.
Cell 57: Unused Capacity is not set.
Cell 58: Unused Capacity = 0.0
Cell 59: Unused Capacity is not set.
Cell 60: Unused Capacity is not set.
Cell 61: Unused Capacity = 0.0
Cell 62: Unused Capacity is not set.
Cell 63: Unused Capacity = 0.0
Cell 64: Unused Capacity is not set.
Cell 65: Unused Capacity is not set.
Cell 66: Unused Capacity is not set.
Cell 67: Unused Capacity is not set.
Cell 68: Unused Capacity = 0.0
Cell 69: Unused Capacity is not set.
Cell 70: Unused Capacity = 0.0
Cell 71: Unused Capacity is not set.
Cell 72: Unused Capacity = 0.0
Cell 73: Unused Capacity is not set.
Cell 74: Unused Capacity = 0.0
Cell 75: Unused Capacity is not set.
Cell 76: Unused Capacity is not set.
Cell 77: Unused Capacity is not set.
Cell 78: Unused Capacity is not set.
Cell 79: Unused Capacity is not set.
Cell 80: Unused Capacity is not set.
Cell 81: Unused Capacity is not set.
Cell 82: Unused Capacity is not set.
Cell 83: Unused Capacity = 0.0
Cell 84: Unused Capacity is not set.
Cell 85: Unused Capacity = 0.0
Cell 86: Unused Capacity is not set.
Cell 87: Unused Capacity is not set.
Cell 88: Unused Capacity is not set.
Cell 89: Unused Capacity = 0.0
Cell 90: Unused Capacity is not set.
Cell 91: Unused Capacity is not set."""



"""Unused_capacity mit c und t"""

"""Diese Funktion funktioniert, jedoch gibt es mir die unused_capacity in Abhängigkeit von c und t aus"""
def unused_capacity_constraint_rule(model, t, c):
    total_charge = sum(
        model.E_charge1[t, c, f] + model.E_charge2[t, c, f] + model.E_charge3[t, c, f] for f in model.nb_fleet if
        (t, c, f) in model.charging_cells_key_set)
    return model.unused_capacity[t, c] == model.cell_charging_cap[c] - total_charge


"""Gleiche Fuktion gibt es in Funktionen"""
def charging_constraint(model: ConcreteModel, t, c):
    return (
        quicksum(
            (model.E_charge1[t, c, f] + model.E_charge2[t, c, f] + model.E_charge3[t, c, f]) * model.ladewirkungsgrad
            for f in model.nb_fleet
            if (t, c, f) in model.charging_cells_key_set
        )
        <= (model.cell_charging_cap[c] - model.Unused_capacity_new[c]) * model.time_resolution
    )

