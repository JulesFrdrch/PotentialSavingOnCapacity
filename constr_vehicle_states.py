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


"""Initialization"""
def constraint_rule_in(model: ConcreteModel, t, c, f):
    return (
        model.Q_in[t, c, f]                             # Die Kapazität der Flotte muss mindestens SOC min entsprechen
        >= model.n_in[t, c, f] * model.fleet_batt_cap[f] * model.SOC_min #kWh = Anzahl * kWh * Faktor
    )

def constraint_balance_constraint(model: ConcreteModel, t, c, f):
    return (            #Gesamt Abfahrende Autos - Ankommende Autos = Fahrzeuge die die Zelle verlassen und noch auf der Route sind
        model.n_exit[t, c, f] - model.n_arrived_vehicles[t, c, f]
        == model.n_out[t, c, f]
    )

def init_n_finished_charge1(model: ConcreteModel, c, f):
    return (
        model.n_finished_charge1[
            model.fleet_depart_times[f], c, f
        ]
        == 0
    )

def init_n_finished_charge2(model: ConcreteModel, c, f):
    return (
        model.n_finished_charge2[
            model.fleet_depart_times[f], c, f
        ]
        == 0
    )

def init_n_finished_charge2_t1(model: ConcreteModel, c, f):
    return (
        model.n_finished_charge2[
            model.fleet_depart_times[f] + 1, c, f
        ]
        == 0
    )

def init_n_finished_charge3(model: ConcreteModel, c, f):
    return (
        model.n_finished_charge3[
            model.fleet_depart_times[f], c, f
        ]
        == 0
    )

def init_n_finished_charge3_t1(model: ConcreteModel, c, f):
    return (
        model.n_finished_charge3[
            model.fleet_depart_times[f] + 1, c, f
        ]
        == 0
    )

def init_n_finished_charge3_t2(model: ConcreteModel, c, f):
    return (
        model.n_finished_charge3[
            model.fleet_depart_times[f] + 2, c, f
        ]
        == 0
    )

def init_Q_finished_charging(model: ConcreteModel, c, f):
    return (
        model.Q_finished_charging[
            model.fleet_depart_times[f], c, f
        ]
        == 0
    )

def init_Q_input_charge1(model: ConcreteModel, c, f):
    return (
        model.Q_input_charge1[model.fleet_depart_times[f], c, f]
        == model.Q_in_charge[model.fleet_depart_times[f], c, f]
    )

def init_Q_output_charge2(model: ConcreteModel, c, f):
    return (
        model.Q_output_charge2[model.fleet_depart_times[f], c, f]
        == 0
    )

def init_Q_output_charge2_t1(model: ConcreteModel, c, f):
    return (
        model.Q_output_charge2[
            model.fleet_depart_times[f] + 1, c, f
        ]
        == 0
    )

def init_Q_output_charge3(model: ConcreteModel, c, f):
    return (
        model.n_charge3[model.fleet_depart_times[f] + 1, c, f]
        == 0
    )

def init_Q_output_charge3_t1(model: ConcreteModel, c, f):
    return (
        model.n_charge3[model.fleet_depart_times[f] + 2, c, f]
        == 0
    )

def init_n_charge2(model: ConcreteModel, c, f):
    return (
        model.n_charge2[model.fleet_depart_times[f], c, f]
        == 0
    )

def init_n_charge2_t1(model: ConcreteModel, c, f):
    return (
        model.n_charge2[model.fleet_depart_times[f] + 1, c, f]
        == 0
    )

def init_n_charge3(model: ConcreteModel, c, f):
    return (
        model.n_charge3[model.fleet_depart_times[f], c, f]
        == 0
    )

def init_n_charge3_t1(model: ConcreteModel, c, f):
    return (
        model.n_charge3[model.fleet_depart_times[f] + 1, c, f]
        == 0
    )

def init_balance_n_to_charge_n_in_charge(model: ConcreteModel, c, f):
    return (
        model.n_charge1[model.fleet_depart_times[f], c, f]
        == model.n_in_charge[model.fleet_depart_times[f], c, f]
    )

def init_balance_n_wait(model: ConcreteModel, c, f):
    return (
        model.n_wait[model.fleet_depart_times[f], c, f]
        + model.n_wait_charge_next[
            model.fleet_depart_times[f], c, f
        ]
        == model.n_in_wait[model.fleet_depart_times[f], c, f]
    )

def init_balance_Q_wait(model: ConcreteModel, c, f):
    return (
        model.Q_wait[model.fleet_depart_times[f], c, f]
        + model.Q_wait_charge_next[
            model.fleet_depart_times[f], c, f
        ]
        == model.Q_in_wait[model.fleet_depart_times[f], c, f]
    )

"""Balances"""
def balance_n_incoming(model: ConcreteModel, t, c, f):
    if model.cell_charging_cap[c] > 0:
        return (
            model.n_in[t, c, f] + model.n_incoming_vehicles[t, c, f]
            == model.n_pass[t, c, f] + model.n_in_wait_charge[t, c, f]
        )
    else:
        return (
                model.n_in[t, c, f] + model.n_incoming_vehicles[t, c, f]
                == model.n_pass[t, c, f]
        )

def balance_Q_incoming(model: ConcreteModel, t, c, f):
    if model.cell_charging_cap[c] > 0:
        return (
            model.Q_in[t, c, f] + model.Q_incoming_vehicles[t, c, f]
            == model.Q_pass[t, c, f] + model.Q_in_charge_wait[t, c, f]
        )
    else:
        return (
                model.Q_in[t, c, f] + model.Q_incoming_vehicles[t, c, f]
                == model.Q_pass[t, c, f]
        )

def balance_waiting_and_charging(model: ConcreteModel, t, c, f):
    return (
        model.n_in_wait_charge[t, c, f]
        == model.n_in_charge[t, c, f] + model.n_in_wait[t, c, f]
    )

def balance_n_finishing(model: ConcreteModel, t, c, f):
    return (
        model.n_finished_charging[t, c, f]
        == model.n_finished_charge1[t, c, f]
        + model.n_finished_charge2[t, c, f]
        + model.n_finished_charge3[t, c, f]
    )

def balance_Q_finishing(model: ConcreteModel, t, c, f):
    return (
        model.Q_finished_charging[t, c, f]
        == model.Q_finished_charge1[t, c, f]
        + model.Q_finished_charge2[t, c, f]
        + model.Q_finished_charge3[t, c, f]
    )

def balance_Q_out(model: ConcreteModel, t, c, f):
    return (
        model.Q_exit[t, c, f] - model.Q_arrived_vehicles[t, c, f]
        == model.Q_out[t, c, f]
    )

def balance_n_charge_transfer_1(model: ConcreteModel, t, c, f):
    return (
        model.n_output_charged1[t, c, f]
        == model.n_finished_charge1[t, c, f] + model.n_charge2[t, c, f]
    )

def balance_n_charge_transfer_2(model: ConcreteModel, t, c, f):
    return (
        model.n_output_charged2[t, c, f]
        == model.n_finished_charge2[t, c, f] + model.n_charge3[t, c, f]
    )

def balance_n_charge_transfer_3(model: ConcreteModel, t, c, f):
    return model.n_output_charged3[t, c, f] == model.n_finished_charge3[t, c, f]

"""Energy Cons"""
def calc_energy_consumption_while_passing(model: ConcreteModel, t, c, f):
    return (
        model.E_consumed_pass[t, c, f]
        == model.n_pass[t, c, f] * model.cell_width[c] * model.fleet_d_spec[f]
    )

def calc_energy_consumption_before_charging(model: ConcreteModel, t, c, f):
    return (
        model.E_consumed_charge_wait[t, c, f]
        == model.n_in_wait_charge[t, c, f] * (1/2) * model.cell_width[c] * model.fleet_d_spec[f]
    )

def calc_energy_consumption_after_charging(model: ConcreteModel, t, c, f):
    return(
        model.E_consumed_exit_charge[t, c, f]
        == model.n_finished_charging[t, c, f] * (1/2) * model.cell_width[c] * model.fleet_d_spec[f]
    )
def energy_consumption_before_charging(model: ConcreteModel, t, c, f):
    return (
        model.Q_in_charge_wait[t, c, f] - model.n_in_wait_charge[t, c, f] * (1/2) * model.cell_width[c] * model.fleet_d_spec[f]
        == model.Q_in_charge[t, c, f] + model.Q_in_wait[t, c, f]
    )

def charging_1(model: ConcreteModel, t, c, f):
    return (
        model.E_charge1[t, c, f]
        <= model.n_charge1[t, c, f] * model.fleet_charge_cap[f] * model.time_resolution * model.ladewirkungsgrad            #* (model.fleet_charge_cap[f]/ 350)
    )

def charging_2(model: ConcreteModel, t, c, f):
    return (
        model.E_charge2[t, c, f]
        <= model.n_charge2[t, c, f] * model.fleet_charge_cap[f] * model.time_resolution * model.ladewirkungsgrad            #* (model.fleet_charge_cap[f]/ 350)
    )

def charging_3(model: ConcreteModel, t, c, f):
    return (
        model.E_charge3[t, c, f]
        <= model.n_charge3[t, c, f] * model.fleet_charge_cap[f] * model.time_resolution * model.ladewirkungsgrad            #* (model.fleet_charge_cap[f]/ 350)
    )

def min_charging_1(model: ConcreteModel, t, c, f):
    return (
        model.E_charge1[t, c, f]
        >= model.n_charge1[t, c, f] * model.fleet_charge_cap[f] * model.t_min * model.ladewirkungsgrad                      #* (model.fleet_charge_cap[f]/ 350)
    )

def min_charging_2(model: ConcreteModel, t, c, f):
    return (
        model.E_charge2[t, c, f]
        >= model.n_charge2[t, c, f] * model.fleet_charge_cap[f] * model.t_min * model.ladewirkungsgrad                      #* (model.fleet_charge_cap[f]/ 350)
    )

def min_charging_3(model: ConcreteModel, t, c, f):
    return (
        model.E_charge3[t, c, f]
        >= model.n_charge3[t, c, f] * model.fleet_charge_cap[f] * model.t_min * model.ladewirkungsgrad                      #* (model.fleet_charge_cap[f]/ 350)
    )

def balance_Q_charging_transfer(model: ConcreteModel, t, c, f):
    return (
        model.Q_output_charge1[t, c, f]
        == model.Q_input_charge2[t, c, f] + model.Q_finished_charge1[t, c, f]
    )

def balance_Q_charging_transfer_1(model: ConcreteModel, t, c, f):
    return (
        model.Q_output_charge2[t, c, f]
        == model.Q_input_charge3[t, c, f] + model.Q_finished_charge2[t, c, f]
    )

def balance_Q_charging_transfer_2(model: ConcreteModel, t, c, f):
    return model.Q_output_charge3[t, c, f] == model.Q_finished_charge3[t, c, f]

"""Relations"""
def setting_relation_n_Q_in_min(model: ConcreteModel, t, c, f):
    return (
        model.Q_in[t, c, f]
        >= model.n_in[t, c, f] * model.fleet_batt_cap[f] * model.SOC_min
    )

def setting_relation_n_Q_in_max(model: ConcreteModel, t, c, f):
    return (
        model.Q_in[t, c, f]
        <= model.n_in[t, c, f] * model.fleet_batt_cap[f] * model.SOC_max
    )

def setting_relation_n_Q_in_wait_charge_min(model: ConcreteModel, t, c, f):
    return (
        model.Q_in_charge_wait[t, c, f]
        >= model.n_in_wait_charge[t, c, f] * model.fleet_batt_cap[f] * model.SOC_min
    )

def setting_relation_n_Q_in_wait_charge_max(model: ConcreteModel, t, c, f):
    return (
        model.Q_in_charge_wait[t, c, f]
        <= model.n_in_wait_charge[t, c, f] * model.fleet_batt_cap[f] * model.SOC_max
    )

def setting_relation_n_Q_in_wait_min(model: ConcreteModel, t, c, f):
    return (
        model.Q_in_wait[t, c, f]
        >= model.n_in_wait[t, c, f] * model.fleet_batt_cap[f] * model.SOC_min
    )

def setting_relation_n_Q_in_wait_max(model: ConcreteModel, t, c, f):
    return (
        model.Q_in_wait[t, c, f]
        <= model.n_in_wait[t, c, f] * model.fleet_batt_cap[f] * model.SOC_max
    )

def setting_relation_n_Q_wait_min(model: ConcreteModel, t, c, f):
    return (
        model.Q_wait[t, c, f]
        >= model.n_wait[t, c, f] * model.fleet_batt_cap[f] * model.SOC_min
    )

def setting_relation_n_Q_wait_max(model: ConcreteModel, t, c, f):
    return (
        model.Q_wait[t, c, f]
        <= model.n_wait[t, c, f] * model.fleet_batt_cap[f] * model.SOC_max
    )

def setting_relation_n_Q_in_charge_min(model: ConcreteModel, t, c, f):
    return (
        model.Q_in_charge[t, c, f]
        >= model.n_in_charge[t, c, f] * model.fleet_batt_cap[f] * model.SOC_min
    )

def setting_relation_n_Q_in_charge_max(model: ConcreteModel, t, c, f):
    return (
        model.Q_in_charge[t, c, f]
        <= model.n_in_charge[t, c, f] * model.fleet_batt_cap[f] * model.SOC_max
    )

def setting_relation_n_Q_wait_charge_next_min(model: ConcreteModel, t, c, f):
    return (
        model.Q_wait_charge_next[t, c, f]
        >= model.n_wait_charge_next[t, c, f] * model.fleet_batt_cap[f] * model.SOC_min
    )

def setting_relation_n_Q_wait_charge_next_max(model: ConcreteModel, t, c, f):
    return (
        model.Q_wait_charge_next[t, c, f]
        <= model.n_wait_charge_next[t, c, f] * model.fleet_batt_cap[f] * model.SOC_max
    )

def setting_relation_n_Q_charge_1_min(model: ConcreteModel, t, c, f):
    return (
        model.Q_input_charge1[t, c, f]
        >= model.n_charge1[t, c, f] * model.fleet_batt_cap[f] * model.SOC_min
    )

def setting_relation_n_Q_charge_1_max(model: ConcreteModel, t, c, f):
    return (
        model.Q_input_charge1[t, c, f]
        <= model.n_charge1[t, c, f] * model.fleet_batt_cap[f] * model.SOC_max
    )

def setting_relation_n_Q_charge_2_min(model: ConcreteModel, t, c, f):
    return (
        model.Q_input_charge2[t, c, f]
        >= model.n_charge2[t, c, f] * model.fleet_batt_cap[f] * model.SOC_min
    )

def setting_relation_n_Q_charge_2_max(model: ConcreteModel, t, c, f):
    return (
        model.Q_input_charge2[t, c, f]
        <= model.n_charge2[t, c, f] * model.fleet_batt_cap[f] * model.SOC_max
    )

def setting_relation_n_Q_charge_3_min(model: ConcreteModel, t, c, f):
    return (
        model.Q_input_charge3[t, c, f]
        >= model.n_charge3[t, c, f] * model.fleet_batt_cap[f] * model.SOC_min
    )

def setting_relation_n_Q_charge_3_max(model: ConcreteModel, t, c, f):
    return (
        model.Q_input_charge3[t, c, f]
        <= model.n_charge3[t, c, f] * model.fleet_batt_cap[f] * model.SOC_max
    )

def setting_relation_n_Q_output_charge_1_min(model: ConcreteModel, t, c, f):
    return (
        model.Q_output_charge1[t, c, f]
        >= model.n_output_charged1[t, c, f] * model.fleet_batt_cap[f] * model.SOC_min
    )

def setting_relation_n_Q_output_charge_1_max(model: ConcreteModel, t, c, f):
    return (
        model.Q_output_charge1[t, c, f]
        <= model.n_output_charged1[t, c, f] * model.fleet_batt_cap[f] * model.SOC_max
    )

def setting_relation_n_Q_output_charge_2_min(model: ConcreteModel, t, c, f):
    return (
        model.Q_output_charge2[t, c, f]
        >= model.n_output_charged2[t, c, f] * model.fleet_batt_cap[f] * model.SOC_min
    )

def setting_relation_n_Q_output_charge_2_max(model: ConcreteModel, t, c, f):
    return (
        model.Q_output_charge2[t, c, f]
        <= model.n_output_charged2[t, c, f] * model.fleet_batt_cap[f] * model.SOC_max
    )

def setting_relation_n_Q_output_charge_3_min(model: ConcreteModel, t, c, f):
    return (
        model.Q_output_charge3[t, c, f]
        >= model.n_output_charged3[t, c, f] * model.fleet_batt_cap[f] * model.SOC_min
    )

def setting_relation_n_Q_output_charge_3_max(model: ConcreteModel, t, c, f):
    return (
        model.Q_output_charge3[t, c, f]
        <= model.n_output_charged3[t, c, f] * model.fleet_batt_cap[f] * model.SOC_max
    )

def setting_relation_n_Q_finished_min(model: ConcreteModel, t, c, f):
    return (
        model.Q_finished_charging[t, c, f]
        >= model.n_finished_charging[t, c, f] * model.fleet_batt_cap[f] * model.SOC_min
    )

def setting_relation_n_Q_finished_max(model: ConcreteModel, t, c, f):
    return (
        model.Q_finished_charging[t, c, f]
        <= model.n_finished_charging[t, c, f] * model.fleet_batt_cap[f] * model.SOC_max
    )

def setting_relation_n_Q_finished_1_min(model: ConcreteModel, t, c, f):
    return (
        model.Q_finished_charge1[t, c, f]
        >= model.n_finished_charge1[t, c, f] * model.fleet_batt_cap[f] * model.SOC_min
    )

def setting_relation_n_Q_finished_1_max(model: ConcreteModel, t, c, f):
    return (
        model.Q_finished_charge1[t, c, f]
        <= model.n_finished_charge1[t, c, f] * model.fleet_batt_cap[f] * model.SOC_max
    )

def setting_relation_n_Q_finished_2_min(model: ConcreteModel, t, c, f):
    return (
        model.Q_finished_charge2[t, c, f]
        >= model.n_finished_charge2[t, c, f] * model.fleet_batt_cap[f] * model.SOC_min
    )

def setting_relation_n_Q_finished_2_max(model: ConcreteModel, t, c, f):
    return (
        model.Q_finished_charge2[t, c, f]
        <= model.n_finished_charge2[t, c, f] * model.fleet_batt_cap[f] * model.SOC_max
    )

def setting_relation_n_Q_finished_3_min(model: ConcreteModel, t, c, f):
    return (
        model.Q_finished_charge3[t, c, f]
        >= model.n_finished_charge3[t, c, f] * model.fleet_batt_cap[f] * model.SOC_min
    )

def setting_relation_n_Q_finished_3_max(model: ConcreteModel, t, c, f):
    return (
        model.Q_finished_charge3[t, c, f]
        <= model.n_finished_charge3[t, c, f] * model.fleet_batt_cap[f] * model.SOC_max
    )

def setting_relation_n_Q_exit_min(model: ConcreteModel, t, c, f):
    return (
        model.Q_exit[t, c, f]
        >= model.n_exit[t, c, f] * model.fleet_batt_cap[f] * model.SOC_min
    )

def setting_relation_n_Q_exit_max(model: ConcreteModel, t, c, f):
    return (
        model.Q_exit[t, c, f]
        <= model.n_exit[t, c, f] * model.fleet_batt_cap[f] * model.SOC_max
    )

def setting_relation_n_Q_arriving_min(model: ConcreteModel, t, c, f):
    return (
        model.Q_arrived_vehicles[t, c, f]
        >= model.n_arrived_vehicles[t, c, f] * model.fleet_batt_cap[f] * model.SOC_min
    )

def setting_relation_n_Q_arriving_max(model: ConcreteModel, t, c, f):
    return (
        model.Q_arrived_vehicles[t, c, f]
        <= model.n_arrived_vehicles[t, c, f] * model.fleet_batt_cap[f] * model.SOC_max
    )

"""Time Jumps"""

def limiting_n_pass(model: ConcreteModel, t, c, f):
    return (model.n_pass[t, c, f] == 0)

def limiting_n_finished_charging(model: ConcreteModel, t, c, f):
    return model.n_finished_charging[t, c, f] == 0

def limiting_n_incoming_vehicles(model: ConcreteModel, t, c, f):
    return model.n_incoming_vehicles[t, c, f] == 0

def limiting_n_in(model: ConcreteModel, t, c, f):
    return model.n_in[t, c, f] == 0

def limiting_n_exit(model: ConcreteModel, t, c, f):
    return model.n_exit[t, c, f] == 0

def limiting_n_out(model: ConcreteModel, t, c, f):
    return model.n_out[t, c, f] == 0

def limiting_n_in_wait_charge(model: ConcreteModel, t, c, f):
    return model.n_in_wait_charge[t, c, f] == 0

def limiting_n_wait(model: ConcreteModel, t, c, f):
    return model.n_wait[t, c, f] == 0

def limiting_n_wait_charge_next(model: ConcreteModel, t, c, f):
    return model.n_wait_charge_next[t, c, f] == 0

def limiting_n_in_charge(model: ConcreteModel, t, c, f):
    return model.n_in_charge[t, c, f] == 0

def limiting_n_charge1(model: ConcreteModel, t, c, f):
    return model.n_charge1[t, c, f] == 0

def limiting_n_charge2(model: ConcreteModel, t, c, f):
    return model.n_charge2[t, c, f] == 0

def limiting_n_charge3(model: ConcreteModel, t, c, f):
    return model.n_charge3[t, c, f] == 0

def limiting_n_output_charged1(model: ConcreteModel, t, c, f):
    return model.n_output_charged1[t, c, f] == 0

def limiting_n_output_charged2(model: ConcreteModel, t, c, f):
    return model.n_output_charged2[t, c, f] == 0

def limiting_n_output_charged3(model: ConcreteModel, t, c, f):
    return model.n_output_charged3[t, c, f] == 0

def limiting_n_finished_charge1(model: ConcreteModel, t, c, f):
    return model.n_finished_charge1[t, c, f] == 0

def limiting_n_finished_charge2(model: ConcreteModel, t, c, f):
    return model.n_finished_charge2[t, c, f] == 0

def limiting_n_finished_charge3(model: ConcreteModel, t, c, f):
    return model.n_finished_charge3[t, c, f] == 0

def limiting_n_exit_charge(model: ConcreteModel, t, c, f):
    return model.n_exit_charge[t, c, f] == 0

def inserting_time_jump_n_pass_exit(model: ConcreteModel, t, c, f):
    if model.cell_charging_cap[c] > 0:
        return (
            model.n_exit[t + 1, c, f]
            == model.n_pass[t, c, f] + model.n_finished_charging[t, c, f]
        )
    else:
        return model.n_exit[t + 1, c, f] == model.n_pass[t, c, f]

def inserting_time_jump_Q_charging_1(model: ConcreteModel, t, c, f):
    return (
        model.Q_output_charge1[t + 1, c, f]
        == model.Q_input_charge1[t, c, f] + model.E_charge1[t, c, f]
    )

def inserting_time_jump_Q_charging_2(model: ConcreteModel, t, c, f):
    return (
        model.Q_output_charge2[t + 1, c, f]
        == model.Q_input_charge2[t, c, f] + model.E_charge2[t, c, f]
    )

def inserting_time_jump_Q_charging_3(model: ConcreteModel, t, c, f):
    return (
        model.Q_output_charge3[t + 1, c, f]
        == model.Q_input_charge3[t, c, f] + model.E_charge3[t, c, f]
    )

def inserting_time_jump_n_charging_1(model: ConcreteModel, t, c, f):
    return model.n_output_charged1[t + 1, c, f] == model.n_charge1[t, c, f]

def inserting_time_jump_n_charging_2(model: ConcreteModel, t, c, f):
    return model.n_output_charged2[t + 1, c, f] == model.n_charge2[t, c, f]

def inserting_time_jump_n_charging_3(model: ConcreteModel, t, c, f):
    return model.n_output_charged3[t + 1, c, f] == model.n_charge3[t, c, f]

def inserting_time_jump_n_charged_exit(model: ConcreteModel, t, c, f):
    return model.n_exit_charge[t + 1, c, f] == model.n_finished_charging[t, c, f]

def inserting_time_jump_Q_passed_exit(model: ConcreteModel, t, c, f):
    if model.cell_charging_cap[c] > 0:
        return (
            model.Q_exit[t + 1, c, f]
            == model.Q_pass[t, c, f] - model.n_pass[t, c, f] * model.cell_width[c] * model.fleet_d_spec[f] + model.Q_finished_charging[t, c, f]
            - model.n_finished_charging[t, c, f] * (1 / 2) * model.cell_width[c] * model.fleet_d_spec[f]
        )
    else:
        return model.Q_exit[t + 1, c, f] == model.Q_pass[t, c, f] - model.n_pass[t, c, f] * model.cell_width[c] \
               * model.fleet_d_spec[f]

"""Queue and Entering Charging"""

def queue_n(model: ConcreteModel, t, c, f):
    return (
        model.n_wait[t, c, f]
        == model.n_wait[t - 1, c, f]
        + model.n_in_wait[t, c, f]
        - model.n_wait_charge_next[t, c, f]
    )

def queue_Q(model: ConcreteModel, t, c, f):
    return (
        model.Q_wait[t, c, f]
        == model.Q_wait[t - 1, c, f]
        + model.Q_in_wait[t, c, f]
        - model.Q_wait_charge_next[t, c, f]
    )

def entering_charging_station_n(model: ConcreteModel, t, c, f):
    return (
        model.n_charge1[t, c, f]
        == model.n_in_charge[t, c, f] + model.n_wait_charge_next[t - 1, c, f]
    )

def entering_charging_station_Q(model: ConcreteModel, t, c, f):
    return (
        model.Q_input_charge1[t, c, f]
        == model.Q_in_charge[t, c, f] + model.Q_wait_charge_next[t - 1, c, f]
    )

