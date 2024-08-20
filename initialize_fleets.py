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

def departing_fleets_n(model: ConcreteModel, t, c, f):
    if c in model.fleet_departing_times[f] and t in model.fleet_departing_times[f][c]:
        return model.n_incoming_vehicles[t, c, f] == model.fleet_departing_times[f][c][t]
    else:
        return model.n_incoming_vehicles[t, c, f] == 0

def departing_fleets_Q(model: ConcreteModel, t, c, f):
    if c in model.fleet_departing_times[f] and t in model.fleet_departing_times[f][c]:
        return (
            model.Q_incoming_vehicles[t, c, f]
            == model.fleet_soc_inits[f] * model.fleet_batt_cap[f] * model.fleet_departing_times[f][c][t]
        )
    else:
        return model.Q_incoming_vehicles[t, c, f] == 0

def routing_n(model: ConcreteModel, t, kl, f):
    return (
        model.n_out[t, model.fleet_routes[f][kl], f]
        == model.n_in[t, model.fleet_routes[f][kl + 1], f]
    )

def routing_Q(model: ConcreteModel, t, kl, f):
    return (
        model.Q_out[t, model.fleet_routes[f][kl], f]
        == model.Q_in[t, model.fleet_routes[f][kl + 1], f]
    )

def init_n_in_fleet(model: ConcreteModel, t, c, f):
    return model.n_in[t, model.fleet_routes[f][0], f] == 0 #Sagt, dass für jede Flotte f n_in für die erste Zelle der Route gleich 0 ist

def init_Q_in_fleet(model: ConcreteModel, t, c, f):
    return model.Q_in[t, model.fleet_routes[f][0], f] == 0

def restrict_arrivals_n(model: ConcreteModel, c, f):
    if c in model.fleet_arriving[f].keys():
        return quicksum(model.n_arrived_vehicles[t0, c, f] for t0 in model.nb_timestep if (t0, c, f) in model.key_set) == model.fleet_arriving[f][c]

    else:
        return quicksum(model.n_arrived_vehicles[t0, c, f] for t0 in model.nb_timestep if (t0, c, f) in model.key_set) == 0

def restrict_arrivals_Q(model: ConcreteModel, c, f):
    if c in model.fleet_arriving[f].keys():
        return quicksum(
            model.Q_arrived_vehicles[t0, c, f] for t0 in model.nb_timestep if (t0, c, f) in model.key_set) >= 0

    else:
        return quicksum(
            model.Q_arrived_vehicles[t0, c, f] for t0 in model.nb_timestep if (t0, c, f) in model.key_set) == 0

def init_n_out_fleet(model: ConcreteModel, t, c, f):
    return model.n_out[t, model.fleet_routes[f][-1], f] == 0 #model.fleet_routes[f][-1] beschreibt die letzte Zelle einer Route

def init_Q_out_fleet(model: ConcreteModel, t, c, f):
    return model.Q_out[t, model.fleet_routes[f][-1], f] == 0

def restrict_time_frame_exit(model: ConcreteModel, t, c, f):
    return model.n_exit[t, c, f] == 0

def restrict_time_frame_arrive(model: ConcreteModel, t, c, f):
    return model.n_arrived_vehicles[t, c, f] == 0

def restrict_time_frame_in(model: ConcreteModel, t, c, f):
    return model.n_in[t, c, f] == 0

