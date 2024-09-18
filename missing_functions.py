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


def missing_functions(model: ConcreteModel):
    """Fehlende Relations bezüglich Q_min und Q_max"""
    model.setting_relation_n_Q_pass_min = Constraint(
        model.key_set, rule=setting_relation_n_Q_pass_min
    )
    model.setting_relation_n_Q_pass_max = Constraint(
        model.key_set, rule=setting_relation_n_Q_pass_max
    )
    model.setting_relation_n_Q_out_min = Constraint(
        model.key_set, rule=setting_relation_n_Q_out_min
    )
    model.setting_relation_n_Q_out_max = Constraint(
        model.key_set, rule=setting_relation_n_Q_out_max
    )
    model.setting_relation_n_Q_incoming_vehicles_min = Constraint(
        model.key_set, rule=setting_relation_n_Q_incoming_vehicles_min
    )
    model.setting_relation_n_Q_incoming_vehicles_max = Constraint(
        model.key_set, rule=setting_relation_n_Q_incoming_vehicles_max
    )
    """Fehlende Time Jumps"""
    model.limiting_n_arrived_vehicles = Constraint(
        model.key_set_with_only_last_ts, rule=limiting_n_arrived_vehicles
    )
    model.limiting_n_in_wait = Constraint(
        model.key_set_with_only_last_five_ts_CS, rule=limiting_n_in_wait
    )


"""Fehlende Relations bezüglich Q_min und Q_max"""
def setting_relation_n_Q_pass_min(model: ConcreteModel, t, c, f):
    return (
        model.Q_pass[t, c, f]
        >= model.n_pass[t, c, f] * model.fleet_batt_cap[f] * model.SOC_min
    )

def setting_relation_n_Q_pass_max(model: ConcreteModel, t, c, f):
    return (
        model.Q_pass[t, c, f]
        <= model.n_pass[t, c, f] * model.fleet_batt_cap[f] * model.SOC_max
    )

def setting_relation_n_Q_out_min(model: ConcreteModel, t, c, f):
    return (
        model.Q_out[t, c, f]
        >= model.n_out[t, c, f] * model.fleet_batt_cap[f] * model.SOC_min
    )

def setting_relation_n_Q_out_max(model: ConcreteModel, t, c, f):
    return (
        model.Q_out[t, c, f]
        <= model.n_out[t, c, f] * model.fleet_batt_cap[f] * model.SOC_max
    )

def setting_relation_n_Q_incoming_vehicles_min(model: ConcreteModel, t, c, f):
    return (
        model.Q_incoming_vehicles[t, c, f]
        >= model.n_incoming_vehicles[t, c, f] * model.fleet_batt_cap[f] * model.SOC_min
    )

def setting_relation_n_Q_incoming_vehicles_max(model: ConcreteModel, t, c, f):
    return (
        model.Q_incoming_vehicles[t, c, f]
        <= model.n_incoming_vehicles[t, c, f] * model.fleet_batt_cap[f] * model.SOC_max
    )

"""Fehlende Time Jumps"""
def limiting_n_arrived_vehicles(model: ConcreteModel, t, c, f):
    return model.n_arrived_vehicles[t, c, f] == 0

def limiting_n_in_wait(model: ConcreteModel, t, c, f):
    return model.n_in_wait[t, c, f] == 0



