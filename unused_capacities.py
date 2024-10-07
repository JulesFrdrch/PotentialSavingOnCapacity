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

def charging_at_restarea(model: ConcreteModel, t, c):
    return (
        quicksum(
            (model.E_charge1[t, c, f] + model.E_charge2[t, c, f] + model.E_charge3[t, c, f])
            for f in model.nb_fleet
            if (t, c, f) in model.charging_cells_key_set
        )
        <= model.cell_charging_cap[c] * model.time_resolution * model.ladewirkungsgrad
    )

def charging_constraint(model: ConcreteModel, t, c):
    return (
            quicksum(
                (model.E_charge1[t, c, f] + model.E_charge2[t, c, f] + model.E_charge3[
                    t, c, f])
                for f in model.nb_fleet
                if (t, c, f) in model.charging_cells_key_set
            )
            <= (model.cell_charging_cap[c] - model.Unused_capacity_new[c]) * model.time_resolution * model.ladewirkungsgrad
    )

def charging_constraint2(model: ConcreteModel, t, c):
    return (
        quicksum(
            (model.E_charge1[t, c, f] + model.E_charge2[t, c, f] + model.E_charge3[t, c, f])
            for f in model.nb_fleet
            if (t, c, f) in model.charging_cells_key_set
        )
        <= (model.cell_charging_cap[c] - model.Unused_capacity_new[c]) * model.time_resolution * model.ladewirkungsgrad
    )



"""Unused_capacity mit c und t"""

"""Diese Funktion funktioniert, jedoch gibt es mir die unused_capacity in Abhängigkeit von c und t aus"""
def unused_capacity_constraint_rule(model, t, c):
    total_charge = sum(
        model.E_charge1[t, c, f] + model.E_charge2[t, c, f] + model.E_charge3[t, c, f] for f in model.nb_fleet if
        (t, c, f) in model.charging_cells_key_set)
    return model.unused_capacity[t, c] == model.cell_charging_cap[c] - total_charge

