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
#  File "C:\Users\Shadow\PycharmProjects\PotentialSavingOnCapacity\functions.py", line 19, in <module>
#    from uncontrolled_cars import *
#  File "C:\Users\Shadow\PycharmProjects\PotentialSavingOnCapacity\uncontrolled_cars.py", line 86
#    TypeError: unsupported operand type(s) for +: 'NoneType' and 'NoneType'

#SyntaxError: (unicode error) 'unicodeescape' codec can't decode bytes in position 45-46: truncated \UXXXXXXXX escape

#Process finished with exit code 1"""

def constraint_n_fleet_37_for_cell_12(model, t, c, f):
    if f == 37 and c == 12:  # Nur für Flotte 37 und Zelle 12
        return (
            model.n_in[t, c, f] + model.n_incoming_vehicles[t, c, f]
            == model.n_in_wait_charge[t, c, f]
        )

    else:
        # Keine Bedingung für andere Flotten
        return Constraint.Skip

def constraint_Q_fleet_37_for_cell_12(model, t, c, f):
    if f == 37 and c == 12:  # Nur für Flotte 37 und Zelle 12
        return (
            model.Q_in[t, c, f] + model.Q_incoming_vehicles[t, c, f]
            == model.Q_in_wait_charge[t, c, f]
        )

    else:
        # Keine Bedingung für andere Flotten
        return Constraint.Skip


def charging_decison_random_fleets_SOC35(model: ConcreteModel, t, c, f):
    if model.Q_in[t, c, f] <= model.n_in[t, c, f] * model.fleet_batt_cap[f] * 0.35:
        return (
            model.n_in[t, c, f] == model.n_in_wait_charge[t, c, f]
        )

    else:
        # Keine Bedingung für andere Flotten
        return Constraint.Skip

def fleet_control_rule(model, t, c, f):
    # Bedingung: Q_in <= n_in * batt_cap * 0.35
    if model.Q_in[t, c, f] <= model.n_in[t, c, f] * model.fleet_batt_cap[f] * 0.35:
        # Setze n_in = n_in_wait_charge
        return model.n_in[t, c, f] == model.n_in_wait_charge[t, c, f]
    # Wenn die Bedingung nicht zutrifft, keine Veränderung:
    return Constraint.Skip

M=1e6
def fleet_control_rule_1(model, t, c, f):
    """
    Erzwingt die Bedingung Q_in[t, c, f] <= n_in[t, c, f] * fleet_batt_cap[f] * 0.35, wenn binary_condition == 1.
    """
    return model.Q_in[t, c, f] <= model.n_in[t, c, f] * model.fleet_batt_cap[f] * 0.35 + M * (
                1 - model.binary_condition[t, c, f])

def fleet_control_rule_2(model, t, c, f):
    """
    Setzt n_in[t, c, f] gleich n_in_wait_charge[t, c, f], wenn binary_condition == 1.
    """
    return model.n_in[t, c, f] == model.n_in_wait_charge[t, c, f] * model.binary_condition[t, c, f]

def fleet_control_rule_3(model, t, c, f):
    """
    Erzwingt die Bedingung Q_in[t, c, f] > n_in[t, c, f] * fleet_batt_cap[f] * 0.35, wenn binary_condition == 0.
    """
    return model.Q_in[t, c, f] >= model.n_in[t, c, f] * model.fleet_batt_cap[f] * 0.35 - M * model.binary_condition[
        t, c, f]



from pyomo.environ import ConcreteModel, Var, Constraint, Binary

def control_random_fleet_with_big_m(model: ConcreteModel, M=1e6, charge_threshold=0.35):
    """
    Steuert Fahrzeuge der random_fleet, basierend auf der Bedingung:
    Wenn Q_in[t, c, f] <= n_in[t, c, f] * fleet_batt_cap[f] * charge_threshold,
    dann sollen die Fahrzeuge in die Ladewarteschlange (n_in_wait_charge) geschickt werden.

    Nutzt die Big-M-Methode zur Modellierung der Bedingung, da Pyomo keine Variablen in einem if-Statement erlaubt.

    :param model: Das ConcreteModel des Optimierungsproblems.
    :param M: Ein sehr großer Wert (Big-M) zur Modellierung der Bedingung.
    :param charge_threshold: Schwellenwert für den Ladezustand (als Anteil der Batterie).
    """

    # Definiere eine binäre Variable für jeden Zeitschritt, Zelle und Flotten-ID
    model.binary_condition = Var(model.random_fleet_cs, within=Binary)

    # Regel 1: Wenn Q_in <= n_in * fleet_batt_cap * charge_threshold, dann binary_condition = 1
    def fleet_control_rule_1(model, t, c, f):
        return model.Q_in[t, c, f] <= model.n_in[t, c, f] * model.fleet_batt_cap[f] * charge_threshold + M * (1 - model.binary_condition[t, c, f])

    # Regel 2: Wenn binary_condition = 1, dann setze n_in = n_in_wait_charge
    def fleet_control_rule_2(model, t, c, f):
        return model.n_in[t, c, f] == model.n_in_wait_charge[t, c, f] * model.binary_condition[t, c, f]

    # Regel 3: Wenn binary_condition = 0, dann muss die Bedingung umgekehrt sein (Q_in > Schwellenwert)
    def fleet_control_rule_3(model, t, c, f):
        return model.Q_in[t, c, f] >= model.n_in[t, c, f] * model.fleet_batt_cap[f] * charge_threshold - M * model.binary_condition[t, c, f]

    # Füge die Regeln als Constraints in das Modell ein
    model.control_rule_1 = Constraint(model.random_fleet_cs, rule=fleet_control_rule_1)
    model.control_rule_2 = Constraint(model.random_fleet_cs, rule=fleet_control_rule_2)
    model.control_rule_3 = Constraint(model.random_fleet_cs, rule=fleet_control_rule_3)

