import numpy as np
from pyomo.core import ConcreteModel
from scipy.stats import norm
from pyomo.environ import *
import numpy as np
from utils import *
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import matplotlib.pyplot as plt
# from _import_geographic_data import *
from matplotlib.offsetbox import AnchoredText
# from _import_optimization_files import *
import time
from pyomo.core.util import quicksum
from datetime import datetime


def balance_n_cell37_alt(model: ConcreteModel, t, c, f):
    if f == 37 and c == 12:  # Prüfe, ob die Flotte die ID 37 hat und die Zelle die ID 12
        return (
            model.n_in[t, c, f] + model.n_incoming_vehicles[t, c, f]
            == model.n_in_wait_charge[t, c, f]
        )
    elif f == 37:  # Für alle anderen Zellen dieser Flotte
        return (
            model.n_in[t, c, f] + model.n_incoming_vehicles[t, c, f]
            == model.n_pass[t, c, f]
        )
    else:
        # Wenn es nicht die Flotte 37 ist, keine Bedingung anwenden
        return Constraint.Skip

def balance_q_cell37_alt(model: ConcreteModel, t, c, f):
    if f == 37 and c == 12:  # Prüfe, ob die Flotte die ID 37 hat und die Zelle die ID 12
        return (
            model.Q_in[t, c, f] + model.Q_incoming_vehicles[t, c, f]
            == model.Q_in_wait_charge[t, c, f]
        )
    elif f == 37:  # Für alle anderen Zellen dieser Flotte
        return (
            model.Q_in[t, c, f] + model.Q_incoming_vehicles[t, c, f]
            == model.Q_pass[t, c, f]
        )
    else:
        # Wenn es nicht die Flotte 37 ist, keine Bedingung anwenden
        return Constraint.Skip

"""Fehlermeldung: TypeError: balance_n_cell37() missing 3 required positional arguments: 't', 'c', and 'f'"""

def balance_n_cell37(model, t, c, f):
    if f == 37 and c == 12:  # Nur für Flotte 37 und Zelle 12
        return (
            model.n_in[t, c, f] + model.n_incoming_vehicles[t, c, f]
            == model.n_in_wait_charge[t, c, f]
        )
    elif f == 37:  # Nur für Flotte 37 und andere Zellen
        return (
            model.n_in[t, c, f] + model.n_incoming_vehicles[t, c, f]
            == model.n_pass[t, c, f]
        )
    else:
        # Keine Bedingung für andere Flotten
        return Constraint.Skip

def balance_q_cell37(model, t, c, f):
    if f == 37 and c == 12:  # Nur für Flotte 37 und Zelle 12
        return (
            model.Q_in[t, c, f] + model.Q_incoming_vehicles[t, c, f]
            == model.Q_in_wait_charge[t, c, f]
        )
    elif f == 37:  # Nur für Flotte 37 und andere Zellen
        return (
            model.Q_in[t, c, f] + model.Q_incoming_vehicles[t, c, f]
            == model.Q_pass[t, c, f]
        )
    else:
        # Keine Bedingung für andere Flotten
        return Constraint.Skip

#"""Traceback (most recent call last):
#  File "C:\Users\Shadow\PycharmProjects\PotentialSavingOnCapacity\main.py", line 229, in <module>
#    export_energy_comparison_to_excel(charging_model, time_of_optimization)
#  File "C:\Users\Shadow\PycharmProjects\PotentialSavingOnCapacity\Ausgabe.py", line 512, in export_energy_comparison_to_excel
#    charging_model.E_charge1[t, c, f].value +
#TypeError: unsupported operand type(s) for +: 'NoneType' and 'NoneType'"""

#"""Traceback (most recent call last):
#  File "C:\Users\Shadow\PycharmProjects\PotentialSavingOnCapacity\main.py", line 2, in <module>
#    from Funktionen import *
#  File "C:\Users\Shadow\PycharmProjects\PotentialSavingOnCapacity\Funktionen.py", line 19, in <module>
#    from uncontrolled_cars import *
#  File "C:\Users\Shadow\PycharmProjects\PotentialSavingOnCapacity\uncontrolled_cars.py", line 86
#    TypeError: unsupported operand type(s) for +: 'NoneType' and 'NoneType'

#SyntaxError: (unicode error) 'unicodeescape' codec can't decode bytes in position 45-46: truncated \UXXXXXXXX escape

#Process finished with exit code 1"""