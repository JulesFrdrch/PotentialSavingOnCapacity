import numpy as np
from pyomo.core import ConcreteModel
from scipy.stats import norm
from pyomo.environ import *
from initialize_fleets import *
from constr_vehicle_states import *
import numpy as np
from utils import *
import warnings
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
from uncontrolled_cars import *
from unused_capacities import *
warnings.simplefilter(action='ignore', category=FutureWarning)

"""Read Fleets"""
def read_fleets(fleet_df):
    """

    :param fleet_df:
    :param touple_column:
    :return:
    """
    for col in ["route", "incoming", "arriving", "depart_time"]:
        route_column = fleet_df[col].to_list()
        fleet_df[col] = [eval(item) for item in route_column]
    return fleet_df


"""Add Decision Variables"""
def add_decision_variables_and_create_key_sets(
    model: ConcreteModel,
    time_resolution: float,
    nb_fleets: int,
    nb_cells: int,
    nb_timesteps: int,
    SOC_min: float,
    SOC_max: float,
    fleet_df: pd.DataFrame, #DataFrame
    cell_df: pd.DataFrame,
    t_min: float,
    SOC_upper_threshold: float,
    SOC_lower_threshold: float,
    SOC_finished_charging_random: float,
    SOC_loading_controlled: float,
    t_min_random: float,
    t_min_controlled: float,
):
    """

    :param model:
    :param time_resolution:
    :param nb_fleets:
    :param nb_cells:
    :param nb_timesteps:
    :param SOC_min:
    :param SOC_max:
    :return:
    """
    model.nb_fleets = nb_fleets
    model.nb_cells = nb_cells
    model.nb_timesteps = nb_timesteps
    model.nb_fleet = range(0, nb_fleets)
    model.nb_cell = range(0, nb_cells)
    model.nb_timestep = range(0, nb_timesteps)
    model.time_resolution = time_resolution
    model.SOC_min = SOC_min
    model.SOC_max = SOC_max
    model.SOC_upper_threshold = SOC_upper_threshold
    model.SOC_lower_threshold = SOC_lower_threshold
    model.t_min = t_min
    model.t_min_random = t_min_random
    model.t_min_controlled = t_min_controlled
    model.SOC_finished_charging_random = SOC_finished_charging_random
    model.SOC_loading_controlled = SOC_loading_controlled
    #print("Model.tim_min ist:",t_min)
    model.fleet_df = fleet_df #DataFrame
    #print("Model.fleet_df ist:", model.fleet_df)
    model.fleet_routes = fleet_df["route"].to_list() #DataFrame
    #print("Model.fleet_routes ist:",model.fleet_routes)
    model.fleet_depart_times = fleet_df["start_timestep"].to_list() #DataFrame
    #print("Model.fleet_depart_times ist:", model.fleet_depart_times)
    model.cell_width = cell_df["length"].array
    #print("Model.cell_width ist:", model.cell_width)
    model.cell_charging_cap = cell_df["capacity"].array
    #print("Model.cell_charging_cap ist:", model.cell_charging_cap)
    model.random_fleet = fleet_df["random_fleet"].to_list()
    #print("Model.random_fleet ist:", model.random_fleet)

    """Erstellung der wichtigsten Key Sets"""
    t0 = time.time()
    model.key_set = Set(initialize=create_set_init)                                                                         #Alle t, c, f
    model.charging_cells_key_set = set([key for key in model.key_set if model.cell_charging_cap[
        key[1]] > 0])                                                                                                       #t, c (mit CS), f
    model.no_charging_cells_key_set = [key for key in model.key_set if model.cell_charging_cap[
        key[1]] == 0]                                                                                                       #t, c (ohe CS), f
    model.random_fleet_key_set = set(key for key in model.key_set if model.random_fleet[key[2]] == 1)                       #t, c, f (random)
    model.random_fleet_cs = set([key for key in model.random_fleet_key_set if model.cell_charging_cap[
        key[1]] > 0])                                                                                                       #t, c (mit CS), f (random)
    model.controlled_fleet_key_set = set(key for key in model.key_set if model.random_fleet[key[2]] == 0)                   #t, c, f (controlled)
    model.controlled_fleet_cs = set([key for key in model.controlled_fleet_key_set if model.cell_charging_cap[
        key[1]] > 0])                                                                                                       #t, c (mit CS), f (controlled)


    print("Nb. of keys (key_set)             :", len(model.key_set), " dies entspricht", len(model.key_set) / 120,
          "Einträgen")  # Anzahl an erstellten keys (siehe overleaf)
    print("Nb. of keys (charging_cells)      :", len(model.charging_cells_key_set), " dies entspricht",
          len(model.charging_cells_key_set) / 120, "Einträgen")
    print("Nb. of keys (no_charging_cells)   :", len(model.no_charging_cells_key_set), " dies entspricht",
          len(model.no_charging_cells_key_set) / 120, "Einträgen")
    print("Nb. of keys (random_fleet)        :", len(model.random_fleet_key_set), " dies entspricht",
          len(model.random_fleet_key_set) / 120, "Einträgen")
    print("Nb. of keys (random_fleet_CS)     :", len(model.random_fleet_cs), " dies entspricht",
          len(model.random_fleet_cs) / 120, "Einträgen")
    print("Nb. of keys (controlled_fleet)    :", len(model.controlled_fleet_key_set), " dies entspricht",
          len(model.controlled_fleet_key_set) / 120, "Einträgen")
    print("Nb. of keys (controlled_fleet_CS) :", len(model.controlled_fleet_cs), " dies entspricht",
          len(model.controlled_fleet_cs) / 120, "Einträgen")

    """Erstellung restlichen Key Sets mit 3 Einträgen"""
    model.routing_set = Set(initialize=create_set_routing)                                                                  #t, c (r-1), f

    """Erstellung restlichen Key Sets mit 2 Einträgen"""
    model.t_cs = set([(el[0], el[1]) for el in model.charging_cells_key_set])                                               #t, c (mit CS)
    model.keys_c_f = set([(el[1], el[2]) for el in model.key_set])                                                          #c, f
    model.cell_and_fleets_CS = set([(el[1], el[2]) for el in model.charging_cells_key_set])                                 #c (mit CS), f

    # Definiere ein Key_Set für Zellen, die eine Ladestation haben
    model.cs_cells = set([(el[1]) for el in model.charging_cells_key_set])
    print("Overview of the cells with a charging station:")
    print(model.cs_cells)

    print("\nDie Erstellung aller Key Sets dauerte... ", time.time() - t0, "sec")

    """Restliche Key Sets die gelöscht werden können"""
    #model.key_routing_set = Set(initialize=create_set_routing)                                                             #kann gelöscht werden
    #model.cell_and_fleets = set([(el[1], el[2]) for el in model.key_set])                                                  #Kann denke ich rausgenommen werden da nicht verwendet!


    """Erzeugung der Entscheidungsvariablen"""
    #mit Var werden Entscheidungsvariablen definiert in Klammern wird angegeben welchen Wertebereich (Domain) die Variable annehmen kann

    model.ladewirkungsgrad = Param(initialize=0.71428)  # Beispiel: 0.8 = 80% Ladewirkungsgrad

    """n Varialen mit NonNegativeReals"""
    #model.n_incoming_vehicles = Var(model.key_set, within=NonNegativeReals)
    #model.n_in = Var(model.key_set, within=NonNegativeReals)
    #model.n_pass = Var(model.key_set, within=NonNegativeReals)
    #model.n_exit = Var(model.key_set, within=NonNegativeReals)
    #model.n_out = Var(model.key_set, within=NonNegativeReals)
    #model.n_arrived_vehicles = Var(model.key_set, within=NonNegativeReals)
    #model.n_in_wait_charge = Var(model.charging_cells_key_set, within=NonNegativeReals)
    #model.n_in_wait = Var(model.charging_cells_key_set, within=NonNegativeReals)
    #model.n_wait = Var(model.charging_cells_key_set, within=NonNegativeReals)
    #model.n_wait_charge_next = Var(model.charging_cells_key_set, within=NonNegativeReals)
    #model.n_in_charge = Var(model.charging_cells_key_set, within=NonNegativeReals)
    #model.n_charge1 = Var(model.charging_cells_key_set, within=NonNegativeReals)
    #model.n_charge2 = Var(model.charging_cells_key_set, within=NonNegativeReals)
    #model.n_charge3 = Var(model.charging_cells_key_set, within=NonNegativeReals)
    #model.n_output_charged1 = Var(model.charging_cells_key_set, within=NonNegativeReals)
    #model.n_output_charged2 = Var(model.charging_cells_key_set, within=NonNegativeReals)
    #model.n_output_charged3 = Var(model.charging_cells_key_set, within=NonNegativeReals)
    #model.n_finished_charge1 = Var(model.charging_cells_key_set, within=NonNegativeReals)
    #model.n_finished_charge2 = Var(model.charging_cells_key_set, within=NonNegativeReals)
    #model.n_finished_charge3 = Var(model.charging_cells_key_set, within=NonNegativeReals)
    #model.n_finished_charging = Var(model.charging_cells_key_set, within=NonNegativeReals)
    #model.n_exit_charge = Var(model.charging_cells_key_set, within=NonNegativeReals)

    """n Varialen mit NonNegativeIntegers"""
    model.n_incoming_vehicles = Var(model.key_set, within=NonNegativeIntegers)
    model.n_in = Var(model.key_set, within=NonNegativeIntegers)
    model.n_pass = Var(model.key_set, within=NonNegativeIntegers)
    model.n_exit = Var(model.key_set, within=NonNegativeIntegers)
    model.n_out = Var(model.key_set, within=NonNegativeIntegers)
    model.n_arrived_vehicles = Var(model.key_set, within=NonNegativeIntegers)
    model.n_in_wait_charge = Var(model.charging_cells_key_set, within=NonNegativeIntegers)
    model.n_in_wait = Var(model.charging_cells_key_set, within=NonNegativeIntegers)
    model.n_wait = Var(model.charging_cells_key_set, within=NonNegativeIntegers)
    model.n_wait_charge_next = Var(model.charging_cells_key_set, within=NonNegativeIntegers)
    model.n_in_charge = Var(model.charging_cells_key_set, within=NonNegativeIntegers)
    model.n_charge1 = Var(model.charging_cells_key_set, within=NonNegativeIntegers)
    model.n_charge2 = Var(model.charging_cells_key_set, within=NonNegativeIntegers)
    model.n_charge3 = Var(model.charging_cells_key_set, within=NonNegativeIntegers)
    model.n_output_charged1 = Var(model.charging_cells_key_set, within=NonNegativeIntegers)
    model.n_output_charged2 = Var(model.charging_cells_key_set, within=NonNegativeIntegers)
    model.n_output_charged3 = Var(model.charging_cells_key_set, within=NonNegativeIntegers)
    model.n_finished_charge1 = Var(model.charging_cells_key_set, within=NonNegativeIntegers)
    model.n_finished_charge2 = Var(model.charging_cells_key_set, within=NonNegativeIntegers)
    model.n_finished_charge3 = Var(model.charging_cells_key_set, within=NonNegativeIntegers)
    model.n_finished_charging = Var(model.charging_cells_key_set, within=NonNegativeIntegers)
    model.n_exit_charge = Var(model.charging_cells_key_set, within=NonNegativeIntegers)

    """Q und E Variablen!"""

    model.Q_incoming_vehicles = Var(model.key_set, within=NonNegativeReals)
    model.Q_in = Var(model.key_set, within=NonNegativeReals)
    model.Q_pass = Var(model.key_set, within=NonNegativeReals)
    model.Q_exit = Var(model.key_set, within=NonNegativeReals)
    model.Q_out = Var(model.key_set, within=NonNegativeReals)
    model.Q_arrived_vehicles = Var(model.key_set, within=NonNegativeReals)
    model.Q_in_charge_wait = Var(model.charging_cells_key_set, within=NonNegativeReals)
    model.Q_in_wait = Var(model.charging_cells_key_set, within=NonNegativeReals)
    model.Q_in_charge = Var(model.charging_cells_key_set, within=NonNegativeReals)
    model.Q_wait = Var(model.charging_cells_key_set, within=NonNegativeReals)
    model.Q_wait_charge_next = Var(model.charging_cells_key_set, within=NonNegativeReals)
    model.Q_input_charge1 = Var(model.charging_cells_key_set, within=NonNegativeReals)
    model.Q_output_charge1 = Var(model.charging_cells_key_set, within=NonNegativeReals)
    model.Q_input_charge2 = Var(model.charging_cells_key_set, within=NonNegativeReals)
    model.Q_output_charge2 = Var(model.charging_cells_key_set, within=NonNegativeReals)
    model.Q_input_charge3 = Var(model.charging_cells_key_set, within=NonNegativeReals)
    model.Q_output_charge3 = Var(model.charging_cells_key_set, within=NonNegativeReals)
    model.Q_finished_charge1 = Var(model.charging_cells_key_set, within=NonNegativeReals)
    model.Q_finished_charge2 = Var(model.charging_cells_key_set, within=NonNegativeReals)
    model.Q_finished_charge3 = Var(model.charging_cells_key_set, within=NonNegativeReals)
    model.Q_finished_charging = Var(model.charging_cells_key_set, within=NonNegativeReals)

    model.E_consumed_pass = Var(model.key_set, within=NonNegativeReals)
    model.E_consumed_charge_wait = Var(model.key_set, within=NonNegativeReals)
    model.E_consumed_exit_charge = Var(model.key_set, within=NonNegativeReals)
    model.E_charge1 = Var(model.charging_cells_key_set, within=NonNegativeReals)
    model.E_charge2 = Var(model.charging_cells_key_set, within=NonNegativeReals)
    model.E_charge3 = Var(model.charging_cells_key_set, within=NonNegativeReals)

    """Neue Variablen / Julius"""
    # Neue Variable für nicht genutzte Kapazität
    model.unused_capacity = Var(model.t_cs, within=NonNegativeReals)
    model.Unused_capacity_new = Var(model.cs_cells, within=NonNegativeReals)
    #model.Unused_capacity_new = Var(model.nb_cell, within=NonNegativeReals)


def create_set_init(model):
    """
    time range is already defined
    :param model:
    :param fleet_df:
    :param cell_df:
    :return:
    """
    # model.nb_timesteps
    # model.nb_timestep
    # TODO: create here set of touples with each three entries
    # for each fleet: from start time until end + cell only including the cells along the route
    # fleet_df = read_fleets(pd.read_csv("data/fleets.csv"))
    fleet_df = model.fleet_df #Hier wird das Dataframe betrachtet welches in der Main an Add_decision_variables übergeben wird
    #print("Das DataFrame ist:", fleet_df)
    start_time_steps = model.fleet_depart_times
    #print("start_time_steps ist:", start_time_steps)
    routes = model.fleet_routes
    #print("routes ist:", routes)

    for ij in range(0, len(fleet_df)): #ij ist die Flotten_id
        tau = start_time_steps[ij] #Tau ist der Start_time_step
        #print("tau ist:", tau)
        r = routes[ij] #r ist die zu fahrende Route
        #print("r ist:",r)
        # (time, highway section, fleet)
        for t in range(tau, int(model.nb_timesteps)):
            for c in r: #für jede Zelle in der Route
                yield (t, c, ij)
                #print("t (timestep) ist:", t, "c (zelle) ist:", c, "ij (flotten_id) ist:", ij)

def create_set_routing(model):
    """
    time range is already defined
    :param model:
    :param fleet_df:
    :param cell_df:
    :return:
    """
    # model.nb_timesteps
    # model.nb_timestep
    # TODO: create here set of touples with each three entries
    # for each fleet: from start time until end + cell only including the cells along the route
    # fleet_df = read_fleets(pd.read_csv("data/fleets.csv"))
    start_time_steps = model.fleet_depart_times
    routes = model.fleet_routes
    fleet_df = model.fleet_df
    for ij in range(0, len(fleet_df)):  # ij ist die Flotten_id
        tau = start_time_steps[ij]  # Tau ist der Start_time_step
        r = routes[ij]  # r ist die zu fahrende Route
        #print("r ist:",r)
        # (time, highway section, fleet)
        for t in range(tau, int(model.nb_timesteps)):
            for kl in range(0, len(r) - 1): #kl ist die Route ohne die letzte Zelle. Eine 6 Zellen lange Route hat hier 5 Einträge (0,1,2,3,4)
                #print("kl ist:", kl)
                yield (t, kl, ij)
                #print("jetzt yield: t ist", t, "kl ist", kl, "ij ist", ij)

"""initialize Fleets"""

def initialize_fleets(model, fleet_df):
    model.c_departing = ConstraintList()
    model.c_fleet_sizes = ConstraintList()
    model.c_arrivals_fleet = ConstraintList()
    model.c_route = ConstraintList()
    model.c_timeframe = ConstraintList()

    model.fleet_sizes = fleet_df["fleet_size"].array #Flottengröße wird gespeichert\item
    model.fleet_incoming = fleet_df["incoming"].to_list()
    model.fleet_arriving = fleet_df["arriving"].to_list()
    #print("Liste Arriving:  ", model.fleet_arriving)
    model.fleet_charge_cap = fleet_df["charge_cap"].array
    #print("Array Charge Cap:", model.fleet_charge_cap)
    model.fleet_batt_cap = fleet_df["batt_cap"].array
    model.fleet_d_spec = fleet_df["d_spec"].array
    model.fleet_mu = [cap / 350 for cap in fleet_df["charge_cap"].array] #Ladewirkungsgrad (0.7142857142857143)
    model.fleet_routes = fleet_df["route"].to_list()                #bereits oben ausgeführt
    model.fleet_depart_times = fleet_df["start_timestep"].to_list() #bereits oben ausgeführt
    model.fleet_departing_times = fleet_df["depart_time"].to_list()
    model.fleet_soc_inits = fleet_df["SOC_init"].to_list()
    model.departing_fleets_n = Constraint(model.key_set, rule=departing_fleets_n)
    model.departing_fleets_Q = Constraint(model.key_set, rule=departing_fleets_Q)

    # print(model.routing_set.pprint())
    model.routing_n = Constraint(model.routing_set, rule=routing_n)
    model.routing_Q = Constraint(model.routing_set, rule=routing_Q)
    model.init_n_in_fleet = Constraint(model.key_set, rule=init_n_in_fleet)
    model.init_Q_in_fleet = Constraint(model.key_set, rule=init_Q_in_fleet)
    # TODO:  restrict arrivals in such a way that the sum over time can be at maximum a certain number

    model.restrict_arrivals_n = Constraint(                                                                                 #fehlt in Overleaf
        model.keys_c_f,
        rule=restrict_arrivals_n,
    )
    model.restrict_arrivals_Q = Constraint(
        model.keys_c_f,
        rule=restrict_arrivals_Q,
    )
    model.init_n_out_fleet = Constraint(model.key_set, rule=init_n_out_fleet)
    model.init_Q_out_fleet = Constraint(model.key_set, rule=init_Q_out_fleet)
    model.restrict_time_frame_exit = Constraint(
        [
            el                                                                                                              #Nur T und C
            for el in model.key_set                                                                                         #c muss teil der Route sein
            if el[1] in model.fleet_routes[el[2]]                                                                           #Flotte darf noch nicht losgefahren sein
            and el[0] <= model.fleet_depart_times[el[2]]                                                                    #t liegt vor der Abfahrtszeit
        ],
        rule=restrict_time_frame_exit,
    )
    model.restrict_time_frame_arrive = Constraint(
        [
            el
            for el in model.key_set
            if el[1] in model.fleet_routes[el[2]]
            and el[0] <= model.fleet_depart_times[el[2]]
        ],
        rule=restrict_time_frame_arrive,
    )
    model.restrict_time_frame_in = Constraint(
        [
            el
            for el in model.key_set
            if el[1] in model.fleet_routes[el[2]]
            and el[0] <= model.fleet_depart_times[el[2]]
        ],
        rule=restrict_time_frame_in,
    )

"""Zustände definieren"""

def constr_vehicle_states(model: ConcreteModel):
    t0 = time.time()
    model.c_rule_in = Constraint(model.key_set, rule=constraint_rule_in)
    model.c_rule_balance = Constraint(model.key_set, rule=constraint_balance_constraint)

    print("the list comprehension took", time.time() - t0, "sec")

    model.init_n_finished_charge1 = Constraint(
        model.cell_and_fleets_CS, rule=init_n_finished_charge1
    )
    model.init_Q_finished_charging = Constraint(
        model.cell_and_fleets_CS, rule=init_Q_finished_charging
    )
    model.init_n_finished_charge2 = Constraint(
        model.cell_and_fleets_CS, rule=init_n_finished_charge2
    )
    model.init_n_finished_charge2_t1 = Constraint(
        model.cell_and_fleets_CS, rule=init_n_finished_charge2_t1
    )
    model.init_Q_output_charge2 = Constraint(model.cell_and_fleets_CS, rule=init_Q_output_charge2)

    model.init_Q_input_charge1 = Constraint(model.cell_and_fleets_CS, rule=init_Q_input_charge1)
    model.init_n_charge2 = Constraint(model.cell_and_fleets_CS, rule=init_n_charge2)
    model.init_n_charge2_t1 = Constraint(model.cell_and_fleets_CS, rule=init_n_charge2_t1)
    model.init_Q_output_charge2_t1 = Constraint(
        model.cell_and_fleets_CS, rule=init_Q_output_charge2_t1
    )
    model.init_n_finished_charge3 = Constraint(
        model.cell_and_fleets_CS, rule=init_n_finished_charge3
    )
    model.init_n_finished_charge3_t1 = Constraint(
        model.cell_and_fleets_CS, rule=init_n_finished_charge3_t1
    )
    model.init_n_finished_charge3_t2 = Constraint(
        model.cell_and_fleets_CS, rule=init_n_finished_charge3_t2
    )
    model.init_n_charge3 = Constraint(model.cell_and_fleets_CS, rule=init_n_charge3)
    model.init_n_charge3_t1 = Constraint(model.cell_and_fleets_CS, rule=init_n_charge3_t1)
    model.init_Q_output_charge3 = Constraint(model.cell_and_fleets_CS, rule=init_Q_output_charge3)
    model.init_Q_output_charge3_t1 = Constraint(
        model.cell_and_fleets_CS, rule=init_Q_output_charge3_t1
    )
    model.init_balance_n_to_charge_n_in_charge = Constraint(
        model.cell_and_fleets_CS, rule=init_balance_n_to_charge_n_in_charge
    )
    model.init_balance_n_wait = Constraint(model.cell_and_fleets_CS, rule=init_balance_n_wait)
    model.init_balance_Q_wait = Constraint(model.cell_and_fleets_CS, rule=init_balance_Q_wait)


    # constraining_n_pass_to_zero_at_end = Constraint(rule=constraining_n_pass_to_zero_at_end)
    print("Initializations finished")
    # balance constraints for vehicles
    model.balance_n_incoming = Constraint(model.key_set, rule=balance_n_incoming)
    # model.balance_n_incoming_NO_CS = Constraint(model.no_charging_cells_key_set, rule=balance_n_incoming_NO_CS)
    model.balance_Q_incoming = Constraint(model.key_set, rule=balance_Q_incoming)
    #model.balance_Q_incoming_NO_CS = Constraint(model.no_charging_cells_key_set, rule=balance_Q_incoming_NO_CS)
    # model.balance_n_passing = Constraint(model.key_set, rule=balance_n_passing)
    # model.balance_Q_passing = Constraint(model.key_set, rule=balance_Q_passing)
    model.balance_waiting_and_charging = Constraint(
        model.charging_cells_key_set, rule=balance_waiting_and_charging
    )
    # model.balance_n_to_charge = Constraint(model.charging_cells_key_set, rule=balance_n_to_charge)
    model.balance_n_finishing = Constraint(model.charging_cells_key_set, rule=balance_n_finishing)
    model.balance_Q_finishing = Constraint(model.charging_cells_key_set, rule=balance_Q_finishing)
    # model.balance_n_exiting = Constraint(model.charging_cells_key_set, rule=balance_n_exiting)
    # model.balance_n_exiting_NO_CS = Constraint(model.no_charging_cells_key_set, rule=balance_n_exiting_NO_CS)
    # model.balance_Q_exiting = Constraint(model.charging_cells_key_set, rule=balance_Q_exiting)
    # model.balance_Q_exiting_NO_CS = Constraint(model.no_charging_cells_key_set, rule=balance_Q_exiting_NO_CS)
    model.balance_Q_out = Constraint(model.key_set, rule=balance_Q_out)
    model.balance_n_charge_transfer_1 = Constraint(
        model.charging_cells_key_set, rule=balance_n_charge_transfer_1
    )
    model.balance_n_charge_transfer_2 = Constraint(
        model.charging_cells_key_set, rule=balance_n_charge_transfer_2
    )
    model.balance_n_charge_transfer_3 = Constraint(
        model.charging_cells_key_set, rule=balance_n_charge_transfer_3
    )

    print("Balances finished")
    # energy consumption while driving
    model.energy_consumption_while_passing = Constraint(
        model.key_set, rule=calc_energy_consumption_while_passing
    )
    model.calc_energy_consumption_before_charging1 = Constraint(
        model.charging_cells_key_set, rule=calc_energy_consumption_before_charging
    )
    model.energy_consumption_after_charging = Constraint(
        model.charging_cells_key_set, rule=calc_energy_consumption_after_charging
    )
    model.energy_consumption_before_charging = Constraint(
        model.charging_cells_key_set, rule=energy_consumption_before_charging
    )
    # model.calc_energy_consumption_before_charging = Constraint(
    #     model.charging_cells_key_set, rule=calc_energy_consumption_before_charging
    # )

    # model.calc_energy_consumption_after_charging = Constraint(
    #     model.charging_cells_key_set, rule=calc_energy_consumption_after_charging
    # )


    # charging activity
    model.charging_1 = Constraint(model.charging_cells_key_set, rule=charging_1)
    model.charging_2 = Constraint(model.charging_cells_key_set, rule=charging_2)
    model.charging_3 = Constraint(model.charging_cells_key_set, rule=charging_3)
    model.min_charging_1 = Constraint(model.charging_cells_key_set, rule=min_charging_1)
    model.min_charging_2 = Constraint(model.charging_cells_key_set, rule=min_charging_2)
    model.min_charging_3 = Constraint(model.charging_cells_key_set, rule=min_charging_3)

    model.balance_Q_charging_transfer = Constraint(
        model.charging_cells_key_set, rule=balance_Q_charging_transfer
    )
    model.balance_Q_charging_transfer_1 = Constraint(
        model.charging_cells_key_set, rule=balance_Q_charging_transfer_1
    )
    model.balance_Q_charging_transfer_2 = Constraint(
        model.charging_cells_key_set, rule=balance_Q_charging_transfer_2
    )
    print("Energy cons finished")
    # relation between n and Q (n ... nb of vehicels, Q ... cummulated state of charge)
    model.setting_relation_n_Q_in_min = Constraint(
        model.key_set, rule=setting_relation_n_Q_in_min
    )
    model.setting_relation_n_Q_in_max = Constraint(
        model.key_set, rule=setting_relation_n_Q_in_max
    )
    # model.setting_relation_n_Q_in_pass_min = Constraint(
    #     model.key_set, rule=setting_relation_n_Q_in_pass_min
    # )
    # model.setting_relation_n_Q_in_pass_max = Constraint(
    #     model.key_set, rule=setting_relation_n_Q_in_pass_max
    # )
    model.setting_relation_n_Q_in_wait_charge_min = Constraint(
        model.charging_cells_key_set, rule=setting_relation_n_Q_in_wait_charge_min
    )
    model.setting_relation_n_Q_in_wait_charge_max = Constraint(
        model.charging_cells_key_set, rule=setting_relation_n_Q_in_wait_charge_max
    )
    model.setting_relation_n_Q_in_wait_min = Constraint(
        model.charging_cells_key_set, rule=setting_relation_n_Q_in_wait_min
    )
    model.setting_relation_n_Q_in_wait_max = Constraint(
        model.charging_cells_key_set, rule=setting_relation_n_Q_in_wait_max
    )
    model.setting_relation_n_Q_wait_min = Constraint(
        model.charging_cells_key_set, rule=setting_relation_n_Q_wait_min
    )
    model.setting_relation_n_Q_wait_max = Constraint(
        model.charging_cells_key_set, rule=setting_relation_n_Q_wait_max
    )
    model.setting_relation_n_Q_in_charge_min = Constraint(
        model.charging_cells_key_set, rule=setting_relation_n_Q_in_charge_min
    )
    model.setting_relation_n_Q_in_charge_max = Constraint(
        model.charging_cells_key_set, rule=setting_relation_n_Q_in_charge_max
    )
    model.setting_relation_n_Q_wait_charge_next_min = Constraint(
        model.charging_cells_key_set, rule=setting_relation_n_Q_wait_charge_next_min
    )
    model.setting_relation_n_Q_wait_charge_next_max = Constraint(
        model.charging_cells_key_set, rule=setting_relation_n_Q_wait_charge_next_max
    )
    model.setting_relation_n_Q_charge_1_min = Constraint(
        model.charging_cells_key_set, rule=setting_relation_n_Q_charge_1_min
    )
    model.setting_relation_n_Q_charge_1_max = Constraint(
        model.charging_cells_key_set, rule=setting_relation_n_Q_charge_1_max
    )
    model.setting_relation_n_Q_charge_2_min = Constraint(
        model.charging_cells_key_set, rule=setting_relation_n_Q_charge_2_min
    )
    model.setting_relation_n_Q_charge_2_max = Constraint(
        model.charging_cells_key_set, rule=setting_relation_n_Q_charge_2_max
    )
    model.setting_relation_n_Q_charge_3_min = Constraint(
        model.charging_cells_key_set, rule=setting_relation_n_Q_charge_3_min
    )
    model.setting_relation_n_Q_charge_3_max = Constraint(
        model.charging_cells_key_set, rule=setting_relation_n_Q_charge_3_max
    )
    model.setting_relation_n_Q_output_charge_1_min = Constraint(
        model.charging_cells_key_set, rule=setting_relation_n_Q_output_charge_1_min
    )
    model.setting_relation_n_Q_output_charge_1_max = Constraint(
        model.charging_cells_key_set, rule=setting_relation_n_Q_output_charge_1_max
    )
    model.setting_relation_n_Q_output_charge_2_min = Constraint(
        model.charging_cells_key_set, rule=setting_relation_n_Q_output_charge_2_min
    )
    model.setting_relation_n_Q_output_charge_2_max = Constraint(
        model.charging_cells_key_set, rule=setting_relation_n_Q_output_charge_2_max
    )
    model.setting_relation_n_Q_output_charge_3_min = Constraint(
        model.charging_cells_key_set, rule=setting_relation_n_Q_output_charge_3_min
    )
    model.setting_relation_n_Q_output_charge_3_max = Constraint(
        model.charging_cells_key_set, rule=setting_relation_n_Q_output_charge_3_max
    )
    model.setting_relation_n_Q_finished_min = Constraint(
        model.charging_cells_key_set, rule=setting_relation_n_Q_finished_min
    )
    model.setting_relation_n_Q_finished_max = Constraint(
        model.charging_cells_key_set, rule=setting_relation_n_Q_finished_max
    )
    model.setting_relation_n_Q_finished_1_min = Constraint(
        model.charging_cells_key_set, rule=setting_relation_n_Q_finished_1_min
    )
    model.setting_relation_n_Q_finished_1_max = Constraint(
        model.charging_cells_key_set, rule=setting_relation_n_Q_finished_1_max
    )
    model.setting_relation_n_Q_finished_2_min = Constraint(
        model.charging_cells_key_set, rule=setting_relation_n_Q_finished_2_min
    )
    model.setting_relation_n_Q_finished_2_max = Constraint(
        model.charging_cells_key_set, rule=setting_relation_n_Q_finished_2_max
    )
    model.setting_relation_n_Q_finished_3_min = Constraint(
        model.charging_cells_key_set, rule=setting_relation_n_Q_finished_3_min
    )
    model.setting_relation_n_Q_finished_3_max = Constraint(
        model.charging_cells_key_set, rule=setting_relation_n_Q_finished_3_max
    )
    model.setting_relation_n_Q_exit_min = Constraint(
        model.key_set, rule=setting_relation_n_Q_exit_min
    )
    model.setting_relation_n_Q_exit_max = Constraint(
        model.key_set, rule=setting_relation_n_Q_exit_max
    )
    model.setting_relation_n_Q_arriving_min = Constraint(
        model.key_set, rule=setting_relation_n_Q_arriving_min
    )
    model.setting_relation_n_Q_arriving_max = Constraint(
        model.key_set, rule=setting_relation_n_Q_arriving_max
    )
    print("Relations finished")
    # constraints relating to time step
    model.key_set_without_last_t = [                                                                                            #t (0-117), c (CS), f
        el for el in model.key_set if el[0] < model.nb_timesteps - 2
    ]

    model.key_set_without_last_t_CS = [                                                                                         #t (0-117), c (CS), f
        el for el in model.charging_cells_key_set if el[0] < model.nb_timesteps - 2
    ]

    model.key_set_without_last_t_NO_CS = [                                                                                      #Löschen, wird nicht verwendet
        el for el in model.charging_cells_key_set if el[0] < model.nb_timesteps - 2
    ]

    model.key_set_with_only_last_ts = [                                                                                     #t (119), c, f
        el for el in model.key_set if el[0] == model.nb_timesteps-1
    ]
    model.key_set_with_only_last_ts_CS = [                                                                                  #t (119), c (CS), f
        el for el in model.charging_cells_key_set if el[0] == model.nb_timesteps-1
    ]

    model.key_set_with_only_last_two_ts = [                                                                                     #t (118-119), c, f
        el for el in model.key_set if el[0] >= model.nb_timesteps - 2
    ]

    model.key_set_with_only_last_two_ts_CS = [                                                                               #t (118-119), c (CS), f
        el for el in model.charging_cells_key_set if el[0] >= model.nb_timesteps - 2
    ]

    model.key_set_with_only_last_three_ts = [                                                                               #t (117-119), c, f
        el for el in model.key_set if el[0] >= model.nb_timesteps - 3
    ]

    model.key_set_with_only_last_three_ts_CS = [                                                                            #t (117-119), c (CS), f
        el for el in model.charging_cells_key_set if el[0] >= model.nb_timesteps - 3

    ]

    model.key_set_with_only_last_four_ts = [                                                                                #t (116-119), c, f
        el for el in model.key_set if el[0] >= model.nb_timesteps - 4
    ]

    model.key_set_with_only_last_four_ts_CS = [                                                                             #t (116-119), c (CS), f
        el for el in model.charging_cells_key_set if el[0] >= model.nb_timesteps - 4
    ]

    model.key_set_with_only_last_five_ts = [                                                                                #t (115-119), c, f
        el for el in model.key_set if el[0] >= model.nb_timesteps - 5
    ]

    model.key_set_with_only_last_five_ts_CS = [                                                                          #t (115-119), c (CS), f
        el for el in model.charging_cells_key_set if el[0] >= model.nb_timesteps - 5
    ]
    model.limiting_n_pass = Constraint(model.key_set_with_only_last_two_ts,
                                                   rule=limiting_n_pass)

    model.limiting_n_finished_charging = Constraint(model.key_set_with_only_last_ts_CS,
                                                   rule=limiting_n_finished_charging)

    model.limiting_n_incoming_vehicles = Constraint(model.key_set_with_only_last_two_ts,
                                                    rule=limiting_n_incoming_vehicles)

    model.limiting_n_in= Constraint(model.key_set_with_only_last_two_ts,
                                                    rule=limiting_n_in)

    model.limiting_n_exit = Constraint(model.key_set_with_only_last_ts,
                                     rule=limiting_n_exit)
    model.limiting_n_out = Constraint(model.key_set_with_only_last_ts,
                                     rule=limiting_n_out)
    model.limiting_n_in_wait_charge = Constraint(model.key_set_with_only_last_five_ts_CS,
                                     rule=limiting_n_in_wait_charge)
    model.limiting_n_wait = Constraint(model.key_set_with_only_last_five_ts_CS,
                                     rule=limiting_n_wait)
    model.limiting_n_wait_charge_next = Constraint(model.key_set_with_only_last_five_ts_CS,
                                     rule=limiting_n_wait_charge_next)

    model.limiting_n_in_charge = Constraint(model.key_set_with_only_last_five_ts_CS,
                                     rule=limiting_n_in_charge)

    model.limiting_n_charge1 = Constraint(model.key_set_with_only_last_five_ts_CS,
                                     rule=limiting_n_charge1)
    model.limiting_n_charge2 = Constraint(model.key_set_with_only_last_four_ts_CS,
                                     rule=limiting_n_charge2)
    model.limiting_n_charge3 = Constraint(model.key_set_with_only_last_three_ts_CS,
                                     rule=limiting_n_charge3)
    model.limiting_n_output_charged1 = Constraint(model.key_set_with_only_last_four_ts_CS,
                                     rule=limiting_n_output_charged1)
    model.limiting_n_output_charged2 = Constraint(model.key_set_with_only_last_three_ts_CS,
                                     rule=limiting_n_output_charged2)
    model.limiting_n_output_charged3 = Constraint(model.key_set_with_only_last_two_ts_CS,
                                     rule=limiting_n_output_charged3)
    model.limiting_n_finished_charge1 = Constraint(model.key_set_with_only_last_four_ts_CS,
                                     rule=limiting_n_finished_charge1)
    model.limiting_n_finished_charge2= Constraint(model.key_set_with_only_last_three_ts_CS,
                                     rule=limiting_n_finished_charge2)
    model.limiting_n_finished_charge3= Constraint(model.key_set_with_only_last_two_ts_CS,
                                     rule=limiting_n_finished_charge3)
    model.limiting_n_exit_charge= Constraint(model.key_set_with_only_last_ts_CS,
                                     rule=limiting_n_exit_charge)


    model.inserting_time_jump_n_pass_exit = Constraint(
        model.key_set_without_last_t, rule=inserting_time_jump_n_pass_exit
    )
    model.inserting_time_jump_Q_charging_1 = Constraint(
        model.key_set_without_last_t_CS, rule=inserting_time_jump_Q_charging_1
    )
    model.inserting_time_jump_Q_charging_2 = Constraint(
        model.key_set_without_last_t_CS, rule=inserting_time_jump_Q_charging_2
    )
    model.inserting_time_jump_Q_charging_3 = Constraint(
        model.key_set_without_last_t_CS, rule=inserting_time_jump_Q_charging_3
    )

    model.inserting_time_jump_n_charging_1 = Constraint(
        model.key_set_without_last_t_CS, rule=inserting_time_jump_n_charging_1
    )
    model.inserting_time_jump_n_charging_2 = Constraint(
        model.key_set_without_last_t_CS, rule=inserting_time_jump_n_charging_2
    )
    model.inserting_time_jump_n_charging_3 = Constraint(
        model.key_set_without_last_t_CS, rule=inserting_time_jump_n_charging_3
    )

    model.inserting_time_jump_n_charged_exit = Constraint(
        model.key_set_without_last_t_CS, rule=inserting_time_jump_n_charged_exit
    )
    # model.inserting_time_jump_Q_charged_exit = Constraint(
    #     model.key_set_without_last_t_CS, rule=inserting_time_jump_Q_charged_exit
    # )
    model.inserting_time_jump_Q_passed_exit = Constraint(
        model.key_set_without_last_t, rule=inserting_time_jump_Q_passed_exit
    )

    del model.key_set_without_last_t_CS
    del model.key_set_without_last_t
    del model.key_set_without_last_t_NO_CS

    print("Time jumps finished")
    # ISSUE here
    # backwards time jump (t-1)

    # TODO: this can be also more efficient: !!

    model.key_set_without_first_t_CS = [                                                                                    #ALle C's mit CS ohne Timestep 0
        el
        for el in model.charging_cells_key_set
        if el[0] > model.fleet_depart_times[el[2]]
    ]
    #print("key_set_without_first_t_CS ist:", model.key_set_without_first_t_CS)
    #print(len(model.key_set_without_first_t_CS))

    model.queue_n = Constraint(model.key_set_without_first_t_CS, rule=queue_n)
    model.queue_Q = Constraint(model.key_set_without_first_t_CS, rule=queue_Q)

    model.entering_charging_station_n = Constraint(
        model.key_set_without_first_t_CS, rule=entering_charging_station_n
    )
    model.entering_charging_station_Q = Constraint(
        model.key_set_without_first_t_CS, rule=entering_charging_station_Q
    )
    del model.key_set_without_first_t_CS



"""Charging"""
def restraint_charging_capacity(model: ConcreteModel):

    """Mit unused_capacities_NEW"""
    model.c_cell_capacity = Constraint(model.t_cs, rule=charging_constraint)
    #model.c_cell_capacity = Constraint(model.t_cs, rule=charging_constraint2)   #Wirkungsgrad unten statt oben
    """Ohne unused_capacities_NEW"""
    #model.c_cell_capacity = Constraint(model.t_cs, rule=charging_at_restarea)
    #del model.t_cs

"""Uncontrolled / random Cars"""
def uncontrolled_cars_decision(model: ConcreteModel):
    """
    Steuerung der random / uncontrolled cars
    """
    model.charging_decision = Constraint(model.random_fleet_cs, rule=charging_decison_random_fleets_SOC35)


"""Unused_capacity"""
def unused_capacities(model: ConcreteModel):
    """
    Ermittelt die ungenutzten Kapazitäten anhand der Summe der geladenen Energie und der Zellen Kapazität
    """

    """unused_capacity mit c und t"""
    model.unused_capacity_constraint = Constraint(model.t_cs,rule=unused_capacity_constraint_rule)

    """Unused_capacity_new nur mit c"""
    #model.ChargingConstraint = Constraint(model.t_cs, rule=charging_constraint)


"""Setting the waiting queue to zero"""
def set_n_wait_and_n_wait_charge_next_to_zero(model: ConcreteModel):
    """
    Setzt n_wait und n_wait_charge_next für alle Zeitschritte und Zellen auf 0.
    """

    # Definiere die Nebenbedingung für alle Zeitschritte und Zellen
    model.c_n_wait_zero = Constraint(model.charging_cells_key_set, rule=n_wait_to_zero)
    model.c_n_wait_charge_next_zero = Constraint(model.charging_cells_key_set, rule=n_wait_charge_next_to_zero)

"""Zustände definieren"""

def constr_vehicle_states_with_uncontrolled_and_controlled_cars(model: ConcreteModel):
    t0 = time.time()
    model.c_rule_in_c = Constraint(model.controlled_fleet_key_set, rule=constraint_rule_in_c)
    model.c_rule_in_r = Constraint(model.random_fleet_key_set, rule=constraint_rule_in_r)
    model.c_rule_balance = Constraint(model.key_set, rule=constraint_balance_constraint)

    print("the list comprehension took", time.time() - t0, "sec")

    model.init_n_finished_charge1 = Constraint(
        model.cell_and_fleets_CS, rule=init_n_finished_charge1
    )
    model.init_Q_finished_charging = Constraint(
        model.cell_and_fleets_CS, rule=init_Q_finished_charging
    )
    model.init_n_finished_charge2 = Constraint(
        model.cell_and_fleets_CS, rule=init_n_finished_charge2
    )
    model.init_n_finished_charge2_t1 = Constraint(
        model.cell_and_fleets_CS, rule=init_n_finished_charge2_t1
    )
    model.init_Q_output_charge2 = Constraint(model.cell_and_fleets_CS, rule=init_Q_output_charge2)

    model.init_Q_input_charge1 = Constraint(model.cell_and_fleets_CS, rule=init_Q_input_charge1)
    model.init_n_charge2 = Constraint(model.cell_and_fleets_CS, rule=init_n_charge2)
    model.init_n_charge2_t1 = Constraint(model.cell_and_fleets_CS, rule=init_n_charge2_t1)
    model.init_Q_output_charge2_t1 = Constraint(
        model.cell_and_fleets_CS, rule=init_Q_output_charge2_t1
    )
    model.init_n_finished_charge3 = Constraint(
        model.cell_and_fleets_CS, rule=init_n_finished_charge3
    )
    model.init_n_finished_charge3_t1 = Constraint(
        model.cell_and_fleets_CS, rule=init_n_finished_charge3_t1
    )
    model.init_n_finished_charge3_t2 = Constraint(
        model.cell_and_fleets_CS, rule=init_n_finished_charge3_t2
    )
    model.init_n_charge3 = Constraint(model.cell_and_fleets_CS, rule=init_n_charge3)
    model.init_n_charge3_t1 = Constraint(model.cell_and_fleets_CS, rule=init_n_charge3_t1)                          #missing init_n_charge3_t2?
    model.init_Q_output_charge3 = Constraint(model.cell_and_fleets_CS, rule=init_Q_output_charge3)
    model.init_Q_output_charge3_t1 = Constraint(
        model.cell_and_fleets_CS, rule=init_Q_output_charge3_t1
    )
    model.init_balance_n_to_charge_n_in_charge = Constraint(
        model.cell_and_fleets_CS, rule=init_balance_n_to_charge_n_in_charge
    )
    model.init_balance_n_wait = Constraint(model.cell_and_fleets_CS, rule=init_balance_n_wait)
    model.init_balance_Q_wait = Constraint(model.cell_and_fleets_CS, rule=init_balance_Q_wait)


    # constraining_n_pass_to_zero_at_end = Constraint(rule=constraining_n_pass_to_zero_at_end)
    print("Initializations finished")
    # balance constraints for vehicles
    """Controlled"""
    model.balance_n_incoming_c = Constraint(model.controlled_fleet_key_set, rule=balance_n_incoming_c)
    model.balance_Q_incoming_c = Constraint(model.controlled_fleet_key_set, rule=balance_Q_incoming_c)
    model.balance_waiting_and_charging_c = Constraint(
        model.controlled_fleet_cs, rule=balance_waiting_and_charging_c)
    """Random"""
    model.balance_n_incoming_r = Constraint(model.random_fleet_key_set, rule=balance_n_incoming_r)
    model.balance_Q_incoming_r = Constraint(model.random_fleet_key_set, rule=balance_Q_incoming_r)
    model.balance_waiting_and_charging_r = Constraint(
        model.random_fleet_cs, rule=balance_waiting_and_charging_r)


    model.balance_n_finishing = Constraint(model.charging_cells_key_set, rule=balance_n_finishing)                  #x
    model.balance_Q_finishing = Constraint(model.charging_cells_key_set, rule=balance_Q_finishing)                  #x

    model.balance_Q_out_c = Constraint(model.controlled_fleet_key_set, rule=balance_Q_out_c)
    model.balance_Q_out_r = Constraint(model.random_fleet_key_set, rule=balance_Q_out_r)

    model.balance_n_charge_transfer_1 = Constraint(
        model.charging_cells_key_set, rule=balance_n_charge_transfer_1)                                              #x
    model.balance_n_charge_transfer_2 = Constraint(
        model.charging_cells_key_set, rule=balance_n_charge_transfer_2)                                              #x
    model.balance_n_charge_transfer_3 = Constraint(
        model.charging_cells_key_set, rule=balance_n_charge_transfer_3)                                              #x

    print("Balances finished")
    # energy consumption while driving
    model.energy_consumption_while_passing = Constraint(
        model.key_set, rule=calc_energy_consumption_while_passing)
    model.calc_energy_consumption_before_charging1 = Constraint(
        model.charging_cells_key_set, rule=calc_energy_consumption_before_charging)
    model.energy_consumption_after_charging = Constraint(
        model.charging_cells_key_set, rule=calc_energy_consumption_after_charging)
    model.energy_consumption_before_charging = Constraint(
        model.charging_cells_key_set, rule=energy_consumption_before_charging)

    # charging activity
    model.charging_1 = Constraint(model.charging_cells_key_set, rule=charging_1)
    model.charging_2 = Constraint(model.charging_cells_key_set, rule=charging_2)
    model.charging_3 = Constraint(model.charging_cells_key_set, rule=charging_3)

    """controlled fleet minimum charging"""
    model.min_charging_1_c = Constraint(model.controlled_fleet_cs, rule=min_charging_1_c)
    model.min_charging_2_c = Constraint(model.controlled_fleet_cs, rule=min_charging_2_c)
    model.min_charging_3_c = Constraint(model.controlled_fleet_cs, rule=min_charging_3_c)

    """random fleet minimum charging"""
    model.min_charging_1_r = Constraint(model.random_fleet_cs, rule=min_charging_1_r)
    model.min_charging_2_r = Constraint(model.random_fleet_cs, rule=min_charging_2_r)
    model.min_charging_3_r = Constraint(model.random_fleet_cs, rule=min_charging_3_r)

    model.balance_Q_charging_transfer = Constraint(
        model.charging_cells_key_set, rule=balance_Q_charging_transfer)
    model.balance_Q_charging_transfer_1 = Constraint(
        model.charging_cells_key_set, rule=balance_Q_charging_transfer_1)
    model.balance_Q_charging_transfer_2 = Constraint(
        model.charging_cells_key_set, rule=balance_Q_charging_transfer_2)
    print("Energy cons finished")
    # relation between n and Q (n ... nb of vehicels, Q ... cummulated state of charge)
    """Controlled relation settings with SOC_min and SOC_max"""
    model.setting_relation_n_Q_in_wait_charge_min_c = Constraint(
        model.controlled_fleet_cs, rule=setting_relation_n_Q_in_wait_charge_min_c)
    model.setting_relation_n_Q_in_wait_charge_max_c = Constraint(
        model.controlled_fleet_cs, rule=setting_relation_n_Q_in_wait_charge_max_c)
    model.setting_relation_n_Q_in_wait_min_c = Constraint(
        model.controlled_fleet_cs, rule=setting_relation_n_Q_in_wait_min_c)
    model.setting_relation_n_Q_in_wait_max_c = Constraint(
        model.controlled_fleet_cs, rule=setting_relation_n_Q_in_wait_max_c)
    model.setting_relation_n_Q_wait_min_c = Constraint(
        model.controlled_fleet_cs, rule=setting_relation_n_Q_wait_min_c)
    model.setting_relation_n_Q_wait_max_c = Constraint(
        model.controlled_fleet_cs, rule=setting_relation_n_Q_wait_max_c)
    model.setting_relation_n_Q_in_charge_min_c = Constraint(
        model.controlled_fleet_cs, rule=setting_relation_n_Q_in_charge_min_c)
    model.setting_relation_n_Q_in_charge_max_c = Constraint(
        model.controlled_fleet_cs, rule=setting_relation_n_Q_in_charge_max_c)
    model.setting_relation_n_Q_wait_charge_next_min_c = Constraint(
        model.controlled_fleet_cs, rule=setting_relation_n_Q_wait_charge_next_min_c)
    model.setting_relation_n_Q_wait_charge_next_max_c = Constraint(
        model.controlled_fleet_cs, rule=setting_relation_n_Q_wait_charge_next_max_c)

    """Random relation settings with SOC_lower_threshold and SOC_upper_threshold"""
    model.setting_relation_n_Q_in_wait_charge_min_r = Constraint(
        model.random_fleet_cs, rule=setting_relation_n_Q_in_wait_charge_min_r)
    model.setting_relation_n_Q_in_wait_charge_max_r = Constraint(
        model.random_fleet_cs, rule=setting_relation_n_Q_in_wait_charge_max_r)
    model.setting_relation_n_Q_in_wait_min_r = Constraint(
        model.random_fleet_cs, rule=setting_relation_n_Q_in_wait_min_r)
    model.setting_relation_n_Q_in_wait_max_r = Constraint(
        model.random_fleet_cs, rule=setting_relation_n_Q_in_wait_max_r)
    model.setting_relation_n_Q_wait_min_r = Constraint(
        model.random_fleet_cs, rule=setting_relation_n_Q_wait_min_r)
    model.setting_relation_n_Q_wait_max_r = Constraint(
        model.random_fleet_cs, rule=setting_relation_n_Q_wait_max_r)
    model.setting_relation_n_Q_in_charge_min_r = Constraint(
        model.random_fleet_cs, rule=setting_relation_n_Q_in_charge_min_r)
    model.setting_relation_n_Q_in_charge_max_r = Constraint(
        model.random_fleet_cs, rule=setting_relation_n_Q_in_charge_max_r)
    model.setting_relation_n_Q_wait_charge_next_min_r = Constraint(
        model.random_fleet_cs, rule=setting_relation_n_Q_wait_charge_next_min_r)
    model.setting_relation_n_Q_wait_charge_next_max_r = Constraint(
        model.random_fleet_cs, rule=setting_relation_n_Q_wait_charge_next_max_r)


    """Q_in"""
    model.setting_relation_n_Q_in_min = Constraint(model.key_set, rule=setting_relation_n_Q_in_min)
    model.setting_relation_n_Q_in_max = Constraint(model.key_set, rule=setting_relation_n_Q_in_max)

    #model.setting_relation_n_Q_in_min = Constraint(model.controlled_fleet_key_set, rule=setting_relation_n_Q_in_min_c)
    #model.setting_relation_n_Q_in_max = Constraint(model.controlled_fleet_key_set, rule=setting_relation_n_Q_in_max_c)
    #model.setting_relation_n_Q_in_min = Constraint(model.random_fleet_key_set, rule=setting_relation_n_Q_in_min_r)
    #model.setting_relation_n_Q_in_max = Constraint(model.random_fleet_key_set, rule=setting_relation_n_Q_in_max_r)

    """restliche Q Variablen"""
    model.setting_relation_n_Q_charge_1_min = Constraint(
        model.charging_cells_key_set, rule=setting_relation_n_Q_charge_1_min)
    model.setting_relation_n_Q_charge_1_max = Constraint(
        model.charging_cells_key_set, rule=setting_relation_n_Q_charge_1_max)
    model.setting_relation_n_Q_charge_2_min = Constraint(
        model.charging_cells_key_set, rule=setting_relation_n_Q_charge_2_min)
    model.setting_relation_n_Q_charge_2_max = Constraint(
        model.charging_cells_key_set, rule=setting_relation_n_Q_charge_2_max)
    model.setting_relation_n_Q_charge_3_min = Constraint(
        model.charging_cells_key_set, rule=setting_relation_n_Q_charge_3_min)
    model.setting_relation_n_Q_charge_3_max = Constraint(
        model.charging_cells_key_set, rule=setting_relation_n_Q_charge_3_max)

    model.setting_relation_n_Q_output_charge_1_min = Constraint(
        model.charging_cells_key_set, rule=setting_relation_n_Q_output_charge_1_min)
    model.setting_relation_n_Q_output_charge_1_max = Constraint(
        model.charging_cells_key_set, rule=setting_relation_n_Q_output_charge_1_max)
    model.setting_relation_n_Q_output_charge_2_min = Constraint(
        model.charging_cells_key_set, rule=setting_relation_n_Q_output_charge_2_min)
    model.setting_relation_n_Q_output_charge_2_max = Constraint(
        model.charging_cells_key_set, rule=setting_relation_n_Q_output_charge_2_max)
    model.setting_relation_n_Q_output_charge_3_min = Constraint(
        model.charging_cells_key_set, rule=setting_relation_n_Q_output_charge_3_min)
    model.setting_relation_n_Q_output_charge_3_max = Constraint(
        model.charging_cells_key_set, rule=setting_relation_n_Q_output_charge_3_max)

    """controlled finished charging"""
    model.setting_relation_n_Q_finished_min_c = Constraint(
        model.controlled_fleet_cs, rule=setting_relation_n_Q_finished_min_c)
    model.setting_relation_n_Q_finished_max_c = Constraint(
        model.controlled_fleet_cs, rule=setting_relation_n_Q_finished_max_c)

    """random finished charging"""
    model.setting_relation_n_Q_finished_min_r = Constraint(
        model.random_fleet_cs, rule=setting_relation_n_Q_finished_min_r)
    model.setting_relation_n_Q_finished_max_r = Constraint(
        model.random_fleet_cs, rule=setting_relation_n_Q_finished_max_r)

    model.setting_relation_n_Q_finished_1_min = Constraint(
        model.charging_cells_key_set, rule=setting_relation_n_Q_finished_1_min)
    model.setting_relation_n_Q_finished_1_max = Constraint(
        model.charging_cells_key_set, rule=setting_relation_n_Q_finished_1_max)
    model.setting_relation_n_Q_finished_2_min = Constraint(
        model.charging_cells_key_set, rule=setting_relation_n_Q_finished_2_min)
    model.setting_relation_n_Q_finished_2_max = Constraint(
        model.charging_cells_key_set, rule=setting_relation_n_Q_finished_2_max)
    model.setting_relation_n_Q_finished_3_min = Constraint(
        model.charging_cells_key_set, rule=setting_relation_n_Q_finished_3_min)
    model.setting_relation_n_Q_finished_3_max = Constraint(
        model.charging_cells_key_set, rule=setting_relation_n_Q_finished_3_max)
    model.setting_relation_n_Q_exit_min = Constraint(
        model.key_set, rule=setting_relation_n_Q_exit_min)
    model.setting_relation_n_Q_exit_max = Constraint(
        model.key_set, rule=setting_relation_n_Q_exit_max)
    model.setting_relation_n_Q_arriving_min = Constraint(
        model.key_set, rule=setting_relation_n_Q_arriving_min)
    model.setting_relation_n_Q_arriving_max = Constraint(
        model.key_set, rule=setting_relation_n_Q_arriving_max)
    print("Relations finished")
    # constraints relating to time step
    model.key_set_without_last_t = [                                                                                            #t (0-117), c (CS), f
        el for el in model.key_set if el[0] < model.nb_timesteps - 2
    ]

    model.key_set_without_last_t_CS = [                                                                                         #t (0-117), c (CS), f
        el for el in model.charging_cells_key_set if el[0] < model.nb_timesteps - 2
    ]

    model.key_set_without_last_t_NO_CS = [                                                                                      #Löschen, wird nicht verwendet
        el for el in model.charging_cells_key_set if el[0] < model.nb_timesteps - 2
    ]

    model.key_set_with_only_last_ts = [                                                                                     #t (119), c, f
        el for el in model.key_set if el[0] == model.nb_timesteps-1
    ]
    model.key_set_with_only_last_ts_CS = [                                                                                  #t (119), c (CS), f
        el for el in model.charging_cells_key_set if el[0] == model.nb_timesteps-1
    ]

    model.key_set_with_only_last_two_ts = [                                                                                     #t (118-119), c, f
        el for el in model.key_set if el[0] >= model.nb_timesteps - 2
    ]

    model.key_set_with_only_last_two_ts_CS = [                                                                               #t (118-119), c (CS), f
        el for el in model.charging_cells_key_set if el[0] >= model.nb_timesteps - 2
    ]

    model.key_set_with_only_last_three_ts = [                                                                               #t (117-119), c, f
        el for el in model.key_set if el[0] >= model.nb_timesteps - 3
    ]

    model.key_set_with_only_last_three_ts_CS = [                                                                            #t (117-119), c (CS), f
        el for el in model.charging_cells_key_set if el[0] >= model.nb_timesteps - 3

    ]

    model.key_set_with_only_last_four_ts = [                                                                                #t (116-119), c, f
        el for el in model.key_set if el[0] >= model.nb_timesteps - 4
    ]

    model.key_set_with_only_last_four_ts_CS = [                                                                             #t (116-119), c (CS), f
        el for el in model.charging_cells_key_set if el[0] >= model.nb_timesteps - 4
    ]

    model.key_set_with_only_last_five_ts = [                                                                                #t (115-119), c, f
        el for el in model.key_set if el[0] >= model.nb_timesteps - 5
    ]

    model.key_set_with_only_last_five_ts_CS = [                                                                          #t (115-119), c (CS), f
        el for el in model.charging_cells_key_set if el[0] >= model.nb_timesteps - 5
    ]
    model.limiting_n_pass = Constraint(model.key_set_with_only_last_two_ts,
                                                   rule=limiting_n_pass)

    model.limiting_n_finished_charging = Constraint(model.key_set_with_only_last_ts_CS,
                                                   rule=limiting_n_finished_charging)

    model.limiting_n_incoming_vehicles = Constraint(model.key_set_with_only_last_two_ts,
                                                    rule=limiting_n_incoming_vehicles)

    model.limiting_n_in= Constraint(model.key_set_with_only_last_two_ts,
                                                    rule=limiting_n_in)

    model.limiting_n_exit = Constraint(model.key_set_with_only_last_ts,
                                     rule=limiting_n_exit)
    model.limiting_n_out = Constraint(model.key_set_with_only_last_ts,
                                     rule=limiting_n_out)
    model.limiting_n_in_wait_charge = Constraint(model.key_set_with_only_last_five_ts_CS,
                                     rule=limiting_n_in_wait_charge)
    model.limiting_n_wait = Constraint(model.key_set_with_only_last_five_ts_CS,
                                     rule=limiting_n_wait)
    model.limiting_n_wait_charge_next = Constraint(model.key_set_with_only_last_five_ts_CS,
                                     rule=limiting_n_wait_charge_next)

    model.limiting_n_in_charge = Constraint(model.key_set_with_only_last_five_ts_CS,
                                     rule=limiting_n_in_charge)

    model.limiting_n_charge1 = Constraint(model.key_set_with_only_last_five_ts_CS,
                                     rule=limiting_n_charge1)
    model.limiting_n_charge2 = Constraint(model.key_set_with_only_last_four_ts_CS,
                                     rule=limiting_n_charge2)
    model.limiting_n_charge3 = Constraint(model.key_set_with_only_last_three_ts_CS,
                                     rule=limiting_n_charge3)
    model.limiting_n_output_charged1 = Constraint(model.key_set_with_only_last_four_ts_CS,
                                     rule=limiting_n_output_charged1)
    model.limiting_n_output_charged2 = Constraint(model.key_set_with_only_last_three_ts_CS,
                                     rule=limiting_n_output_charged2)
    model.limiting_n_output_charged3 = Constraint(model.key_set_with_only_last_two_ts_CS,
                                     rule=limiting_n_output_charged3)
    model.limiting_n_finished_charge1 = Constraint(model.key_set_with_only_last_four_ts_CS,
                                     rule=limiting_n_finished_charge1)
    model.limiting_n_finished_charge2= Constraint(model.key_set_with_only_last_three_ts_CS,
                                     rule=limiting_n_finished_charge2)
    model.limiting_n_finished_charge3= Constraint(model.key_set_with_only_last_two_ts_CS,
                                     rule=limiting_n_finished_charge3)
    model.limiting_n_exit_charge= Constraint(model.key_set_with_only_last_ts_CS,
                                     rule=limiting_n_exit_charge)


    model.inserting_time_jump_n_pass_exit = Constraint(
        model.key_set_without_last_t, rule=inserting_time_jump_n_pass_exit
    )
    model.inserting_time_jump_Q_charging_1 = Constraint(
        model.key_set_without_last_t_CS, rule=inserting_time_jump_Q_charging_1
    )
    model.inserting_time_jump_Q_charging_2 = Constraint(
        model.key_set_without_last_t_CS, rule=inserting_time_jump_Q_charging_2
    )
    model.inserting_time_jump_Q_charging_3 = Constraint(
        model.key_set_without_last_t_CS, rule=inserting_time_jump_Q_charging_3
    )

    model.inserting_time_jump_n_charging_1 = Constraint(
        model.key_set_without_last_t_CS, rule=inserting_time_jump_n_charging_1
    )
    model.inserting_time_jump_n_charging_2 = Constraint(
        model.key_set_without_last_t_CS, rule=inserting_time_jump_n_charging_2
    )
    model.inserting_time_jump_n_charging_3 = Constraint(
        model.key_set_without_last_t_CS, rule=inserting_time_jump_n_charging_3
    )

    model.inserting_time_jump_n_charged_exit = Constraint(
        model.key_set_without_last_t_CS, rule=inserting_time_jump_n_charged_exit
    )
    # model.inserting_time_jump_Q_charged_exit = Constraint(
    #     model.key_set_without_last_t_CS, rule=inserting_time_jump_Q_charged_exit
    # )
    model.inserting_time_jump_Q_passed_exit = Constraint(
        model.key_set_without_last_t, rule=inserting_time_jump_Q_passed_exit
    )

    del model.key_set_without_last_t_CS
    del model.key_set_without_last_t
    del model.key_set_without_last_t_NO_CS

    print("Time jumps finished")
    # ISSUE here
    # backwards time jump (t-1)

    # TODO: this can be also more efficient: !!

    model.key_set_without_first_t_CS = [                                                                                    #ALle C's mit CS ohne Timestep 0
        el
        for el in model.charging_cells_key_set
        if el[0] > model.fleet_depart_times[el[2]]
    ]
    #print("key_set_without_first_t_CS ist:", model.key_set_without_first_t_CS)
    #print(len(model.key_set_without_first_t_CS))

    model.queue_n = Constraint(model.key_set_without_first_t_CS, rule=queue_n)
    model.queue_Q = Constraint(model.key_set_without_first_t_CS, rule=queue_Q)

    model.entering_charging_station_n = Constraint(
        model.key_set_without_first_t_CS, rule=entering_charging_station_n
    )
    model.entering_charging_station_Q = Constraint(
        model.key_set_without_first_t_CS, rule=entering_charging_station_Q
    )
    del model.key_set_without_first_t_CS

