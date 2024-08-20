import numpy as np
import os
import pandas as pd
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
"""Python Dateien einfügen"""
import Ausgabe as ta
from Funktionen import *
from FehlendeFunktionen import *
from Ausgabe import *
#from _optimization_utils import write_output_files

#from _optimization_utils import *

"""Bibliotheken einfügen"""
import pandas as pd
import numpy as np
import time
from termcolor import colored
from pyomo.util.model_size import build_model_size_report
from scipy.stats import norm
import math
import csv
import geopandas as gpd
from shapely import wkt
from shapely.geometry import MultiLineString, Point
from pyproj import Geod
import pickle
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import pyomo
from pyomo.core.util import quicksum
from pyomo.environ import *

"""Results Files"""

def write_output_file_Fleet_infos(model, time_of_optimization, filename):
    results = get_variables_from_model2(model)

    """Flotten Details"""
    fleet_specifics = pd.DataFrame()
    for f in model.nb_fleet:  # model.nb_fleet = range(0, nb_fleets)
        d = {}
        d["fleet_id"] = f
        d["charge_cap"] = model.fleet_charge_cap[f]
        d["batt_cap"] = model.fleet_batt_cap[f]
        d["d_spec"] = model.fleet_d_spec[f]
        d["incoming"] = sum(model.fleet_incoming[f].values())
        if sum(model.fleet_incoming[f].values()) - np.sum(results["n_arrived_vehicles"][:, :, f]) > 0.01:
            d["all_arrived"] = False
        else:
            d["all_arrived"] = True

        d["arrival_SOC"] = np.sum(results["Q_arrived_vehicles"][:, :, f]) / (
                    model.fleet_charge_cap[f] * np.sum(results["n_arrived_vehicles"][:, :, f]))

        fleet_specifics = fleet_specifics.append(d, ignore_index=True)

    fleet_specifics.to_csv("results/" + "Fleet_Infos" + "(" + time_of_optimization + ")" ".csv")

def write_output_file_Energy_charged(model, time_of_optimization, filename):
    results = get_variables_from_model2(model)

    """Nur charged"""
    total_charged = pd.DataFrame()
    inds_of_all_cells = [ind for ind in model.nb_cell if model.cell_charging_cap[ind] >= 0]
    for ij in range(0, len(inds_of_all_cells)):
        d = {}
        c = inds_of_all_cells[ij]
        d["Cell ID"] = c
        d["Capacity"] = model.cell_charging_cap[c]
        d["Max_Capacity"] = 0

        for t in model.nb_timestep:
            d["Energy Charged at t=" + str(t)] = np.sum(results["E_charge1"][t, c, :]) + np.sum(
                results["E_charge2"][t, c, :]) + np.sum(results["E_charge3"][t, c, :])

        total_charged = total_charged.append(d, ignore_index=True)

    for idx in range(91):
        row_list = total_charged.loc[idx, :].values.flatten().tolist()
        row_list = row_list[3:]
        total_charged.loc[idx,'Max_Capacity']=max(row_list)

    stefan=total_charged.loc[:, 'Max_Capacity']
    total_charged.to_excel("results/" + "Energy_charged" + "(" + time_of_optimization + ")" ".xlsx", index=False)
    # Erstellen des Diagramms für die Summe der "Energy Charged at t"
    timesteps = model.nb_timestep
    total_energy_charged = []

    for t in timesteps:
        total_energy_t = 0
        for c in inds_of_all_cells:
            total_energy_t += np.sum(results["E_charge1"][t, c, :]) + np.sum(results["E_charge2"][t, c, :]) + np.sum(
                results["E_charge3"][t, c, :])
        total_energy_charged.append(total_energy_t)

    # Erstellen und Speichern des Diagramms
    plt.figure(figsize=(14, 8))
    plt.plot(timesteps, total_energy_charged, label='Total Energy Charged', marker='o')
    plt.xlabel('Timestep')
    plt.ylabel('Total Energy Charged')
    plt.title('Total Energy Charged Over Time (all cells)')
    plt.legend(loc='upper right')
    plt.grid(True)

    # Verzeichnis für die Speicherung erstellen, falls es nicht existiert
    output_dir = os.path.join('results', 'Pictures')
    os.makedirs(output_dir, exist_ok=True)

    # Speichern des Diagramms
    output_path = os.path.join(output_dir, f'total_energy_charged_over_time_{time_of_optimization}.png')
    plt.savefig(output_path)

    # Anzeigen des Diagramms
    plt.show()
    # Visualisierung der geladenen Energie für ausgewählte Zellen
    selected_cells = [13,19,10,9]  # Beispielhafte Liste der ausgewählten Zellen, anpassen nach Bedarf

    plt.figure(figsize=(14, 8))
    for c in selected_cells:
        energy_charged = []
        for t in timesteps:
            if c in inds_of_all_cells:
                energy_t = np.sum(results["E_charge1"][t, c, :]) + np.sum(results["E_charge2"][t, c, :]) + np.sum(
                    results["E_charge3"][t, c, :])
                energy_charged.append(energy_t)
            else:
                energy_charged.append(0)

        plt.plot(timesteps, energy_charged, label=f'Cell {c}', marker='o')

    plt.xlabel('Timestep')
    plt.ylabel('Energy Charged')
    plt.title('Energy Charged Over Time for Selected Cells')
    plt.legend(loc='upper right')
    plt.grid(True)

    # Speichern des Diagramms für ausgewählte Zellen
    output_path_selected = os.path.join(output_dir, f'energy_charged_selected_cells_{time_of_optimization}.png')
    plt.savefig(output_path_selected)

    # Anzeigen des Diagramms für ausgewählte Zellen
    plt.show()

def write_output_file_Charging(model, time_of_optimization, filename):
    results = get_variables_from_model2(model)

    """Alle Charging Variablen"""
    charging = pd.DataFrame()
    inds_of_all_cells = [ind for ind in model.nb_cell if model.cell_charging_cap[ind] >= 0]
    for ij in range(0, len(inds_of_all_cells)):
        d = {}
        c = inds_of_all_cells[ij]
        d["Cell ID"] = c
        d["Capacity"] = model.cell_charging_cap[c]
        for t in model.nb_timestep:
            d["n_charge1 at t=" + str(t)] = np.sum(results["n_charge1"][t, c, :])
            d["n_output_charged1 t=" + str(t)] = np.sum(results["n_output_charged1"][t, c, :])
            d["n_finished_charge1 at t=" + str(t)] = np.sum(results["n_finished_charge1"][t, c, :])
            d["n_charge2 at t=" + str(t)] = np.sum(results["n_charge2"][t, c, :])
            d["n_output_charged2 t=" + str(t)] = np.sum(results["n_output_charged2"][t, c, :])
            d["n_finished_charge2 at t=" + str(t)] = np.sum(results["n_finished_charge2"][t, c, :])
            d["n_charge3 at t=" + str(t)] = np.sum(results["n_charge3"][t, c, :])
            d["n_output_charged3 t=" + str(t)] = np.sum(results["n_output_charged3"][t, c, :])
            d["n_finished_charge3 at t=" + str(t)] = np.sum(results["n_finished_charge3"][t, c, :])
            d["unused_capacity at t=" + str(t)] = np.sum(results["unused_capacity"][t, c, :])

        charging = charging.append(d, ignore_index=True)

    charging.to_excel("results/" + "Charging" + "(" + time_of_optimization + ")" ".xlsx", index=False)

def write_output_file_Bewegung(model, time_of_optimization, filename):
    results = get_variables_from_model2(model)

    """Alle Bewegungs Variablen"""
    movement = pd.DataFrame()
    inds_of_all_cells = [ind for ind in model.nb_cell if model.cell_charging_cap[ind] >= 0]
    for ij in range(0, len(inds_of_all_cells)):
        d = {}
        c = inds_of_all_cells[ij]
        d["Cell ID"] = c
        d["Capacity"] = model.cell_charging_cap[c]
        for t in model.nb_timestep:
            d["n_in at t=" + str(t)] = np.sum(results["n_in"][t, c, :])
            d["n_pass at t=" + str(t)] = np.sum(results["n_pass"][t, c, :])
            d["n_in_wait_charge at t=" + str(t)] = np.sum(results["n_in_wait_charge"][t, c, :])

        movement = movement.append(d, ignore_index=True)

    movement.to_excel("results/" + "Bewegungsdaten" + "(" + time_of_optimization + ")" ".xlsx", index=False)

def write_output_file_n_in_wait_charge(model, time_of_optimization, filename):
    results = get_variables_from_model2(model)

    """Nur n_in_wait_charge"""
    inCS = pd.DataFrame()
    inds_of_all_cells = [ind for ind in model.nb_cell if model.cell_charging_cap[ind] >= 0]
    for ij in range(0, len(inds_of_all_cells)):
        d = {}
        c = inds_of_all_cells[ij]
        d["Cell ID"] = c
        d["Capacity"] = model.cell_charging_cap[c]
        for t in model.nb_timestep:
            d["n_in_wait_charge at t=" + str(t)] = np.sum(results["n_in_wait_charge"][t, c, :])

        inCS = inCS.append(d, ignore_index=True)

    inCS.to_excel("results/" + "n_in_wait_charge" + "(" + time_of_optimization + ")" ".xlsx", index=False)

def write_output_file_n_pass(model, time_of_optimization, filename):
    results = get_variables_from_model2(model)

    """Nur n_pass"""
    passing = pd.DataFrame()
    inds_of_all_cells = [ind for ind in model.nb_cell if model.cell_charging_cap[ind] >= 0]
    for ij in range(0, len(inds_of_all_cells)):
        d = {}
        c = inds_of_all_cells[ij]
        d["Cell ID"] = c
        d["Capacity"] = model.cell_charging_cap[c]
        for t in model.nb_timestep:
            d["n_pass at t=" + str(t)] = np.sum(results["n_pass"][t, c, :])

        passing = passing.append(d, ignore_index=True)

    passing.to_excel("results/" + "n_pass" + "(" + time_of_optimization + ")" ".xlsx", index=False)

def write_output_file_n_in(model, time_of_optimization, filename):
    results = get_variables_from_model2(model)

    """Nur n_in"""
    entering = pd.DataFrame()
    inds_of_all_cells = [ind for ind in model.nb_cell if model.cell_charging_cap[ind] >= 0]
    for ij in range(0, len(inds_of_all_cells)):
        d = {}
        c = inds_of_all_cells[ij]
        d["Cell ID"] = c
        d["Capacity"] = model.cell_charging_cap[c]
        for t in model.nb_timestep:
            d["n_in at t=" + str(t)] = np.sum(results["n_in"][t, c, :])

        entering = entering.append(d, ignore_index=True)

    entering.to_excel("results/" + "n_in" + "(" + time_of_optimization + ")" ".xlsx", index=False)

def write_output_file_incoming_vehicles(model, time_of_optimization, filename):
    results = get_variables_from_model2(model)

    """Nur n_incoming_vehicles"""
    departing = pd.DataFrame()
    inds_of_all_cells = [ind for ind in model.nb_cell if model.cell_charging_cap[ind] >= 0]
    for ij in range(0, len(inds_of_all_cells)):
        d = {}
        c = inds_of_all_cells[ij]
        d["Cell ID"] = c
        d["Capacity"] = model.cell_charging_cap[c]
        for t in model.nb_timestep:
            d["n_incoming_vehicles at t=" + str(t)] = np.sum(results["n_incoming_vehicles"][t, c, :])

        departing = departing.append(d, ignore_index=True)

    departing.to_excel("results/" + "incoming_vehicles" + "(" + time_of_optimization + ")" ".xlsx", index=False)

def write_output_file_arrived_vehicles(model, time_of_optimization, filename):
    results = get_variables_from_model2(model)

    """Nur n_arriving_vehicles"""
    arriving = pd.DataFrame()
    inds_of_all_cells = [ind for ind in model.nb_cell if model.cell_charging_cap[ind] >= 0]
    for ij in range(0, len(inds_of_all_cells)):
        d = {}
        c = inds_of_all_cells[ij]
        d["Cell ID"] = c
        d["Capacity"] = model.cell_charging_cap[c]
        for t in model.nb_timestep:
            d["n_arrived_vehicles at t=" + str(t)] = np.sum(results["n_arrived_vehicles"][t, c, :])

        arriving = arriving.append(d, ignore_index=True)

    arriving.to_excel("results/" + "arrived_vehicles" + "(" + time_of_optimization + ")" ".xlsx", index=False)

def write_output_file_charging_stations(model, time_of_optimization, filename):
    results = get_variables_from_model2(model)

    """Gesamt Charging Cell Ausgabe"""  # model.nb_cell = range(0, nb_cells)
    cs_specifics = pd.DataFrame()
    inds_of_charging_cells = [ind for ind in model.nb_cell if model.cell_charging_cap[ind] > 0]
    for ij in range(0, len(inds_of_charging_cells)):
        d = {}
        c = inds_of_charging_cells[ij]
        d["Cell ID"] = c
        d["Capacity"] = model.cell_charging_cap[c]
        for t in model.nb_timestep:
            d["n_in_wait_charge at t=" + str(t)] = np.sum(results["n_in_wait_charge"][t, c, :])
            d["Waiting Vehicles at t=" + str(t)] = np.sum(results["n_wait"][t, c, :]) + np.sum(
                results["n_wait_charge_next"][t, c, :])
            d["Charging Vehicles at t=" + str(t)] = np.sum(results["n_charge1"][t, c, :]) + np.sum(
                results["n_charge2"][t, c, :]) + np.sum(results["n_charge3"][t, c, :])
            d["Energy Charged at t=" + str(t)] = np.sum(results["E_charge1"][t, c, :]) + np.sum(
                results["E_charge2"][t, c, :]) + np.sum(results["E_charge3"][t, c, :])
            # d["Passenger Vehicles at t=" + str(t)] = np.sum(results["n_pass"][t, c, :]) + np.sum(results["n_pass"][t, c, :]) + np.sum(results["n_pass"][t, c, :])
            d["n_in at t=" + str(t)] = np.sum(results["n_in"][t, c, :])  # nur bei 2
            d["n_incoming_vehicles at t=" + str(t)] = np.sum(results["n_incoming_vehicles"][t, c, :])
            d["n_exit at t=" + str(t)] = np.sum(results["n_exit"][t, c, :])  # nur bei 2
            d["n_arrived_vehicles at t=" + str(t)] = np.sum(results["n_arrived_vehicles"][t, c, :])
            d["n_pass at t=" + str(t)] = np.sum(results["n_pass"][t, c, :])

        cs_specifics = cs_specifics.append(d, ignore_index=True)

    cs_specifics.to_excel("results/" + "charging_station_infos" + "(" + time_of_optimization + ")" ".xlsx", index=False)

def get_variables_from_model2(model):                                                                                        #Mit meinen Results
    n_wait_charge_next = np.zeros(
        [
            len(model.nb_timestep),
            len(model.nb_cell),
            len(model.nb_fleet),
        ]
    )
    n_charge1 = np.zeros(
        [
            len(model.nb_timestep),
            len(model.nb_cell),
            len(model.nb_fleet),
        ]
    )

    n_charge2 = np.zeros(
        [
            len(model.nb_timestep),
            len(model.nb_cell),
            len(model.nb_fleet),
        ]
    )

    n_arrived_vehicles = np.zeros(
        [
            len(model.nb_timestep),
            len(model.nb_cell),
            len(model.nb_fleet),
        ]
    )
    E_charge1 = np.zeros(
        [
            len(model.nb_timestep),
            len(model.nb_cell),
            len(model.nb_fleet),
        ]
    )

    Q_arrived_vehicles = np.zeros(
        [
            len(model.nb_timestep),
            len(model.nb_cell),
            len(model.nb_fleet),
        ]
    )
    n_wait = np.zeros(
        [
            len(model.nb_timestep),
            len(model.nb_cell),
            len(model.nb_fleet),
        ]
    )
    E_charge2 = np.zeros(
        [
            len(model.nb_timestep),
            len(model.nb_cell),
            len(model.nb_fleet),
        ]
    )

    E_charge3 = np.zeros(
        [
            len(model.nb_timestep),
            len(model.nb_cell),
            len(model.nb_fleet),
        ]
    )
    n_charge3 = np.zeros(
        [
            len(model.nb_timestep),
            len(model.nb_cell),
            len(model.nb_fleet),
        ]
    )
    n_arrived_vehicles = np.zeros(
        [
            len(model.nb_timestep),
            len(model.nb_cell),
            len(model.nb_fleet),
        ]
    )
    n_pass = np.zeros(
        [
            len(model.nb_timestep),
            len(model.nb_cell),
            len(model.nb_fleet),
        ]
    )
    n_in = np.zeros(
        [
            len(model.nb_timestep),
            len(model.nb_cell),
            len(model.nb_fleet),
        ]
    )
    n_exit = np.zeros(
        [
            len(model.nb_timestep),
            len(model.nb_cell),
            len(model.nb_fleet),
        ]
    )
    n_incoming_vehicles = np.zeros(
        [
            len(model.nb_timestep),
            len(model.nb_cell),
            len(model.nb_fleet),
        ]
    )
    n_in_wait_charge = np.zeros(
        [
            len(model.nb_timestep),
            len(model.nb_cell),
            len(model.nb_fleet),
        ]
    )
    n_arrived_vehicles = np.zeros(
        [
            len(model.nb_timestep),
            len(model.nb_cell),
            len(model.nb_fleet),
        ]
    )
    n_charge1 = np.zeros(
        [
            len(model.nb_timestep),
            len(model.nb_cell),
            len(model.nb_fleet),
        ]
    )
    n_output_charged1 = np.zeros(
        [
            len(model.nb_timestep),
            len(model.nb_cell),
            len(model.nb_fleet),
        ]
    )
    n_finished_charge1 = np.zeros(
        [
            len(model.nb_timestep),
            len(model.nb_cell),
            len(model.nb_fleet),
        ]
    )
    n_charge2 = np.zeros(
        [
            len(model.nb_timestep),
            len(model.nb_cell),
            len(model.nb_fleet),
        ]
    )
    n_output_charged2 = np.zeros(
        [
            len(model.nb_timestep),
            len(model.nb_cell),
            len(model.nb_fleet),
        ]
    )
    n_finished_charge2 = np.zeros(
        [
            len(model.nb_timestep),
            len(model.nb_cell),
            len(model.nb_fleet),
        ]
    )
    n_charge3 = np.zeros(
        [
            len(model.nb_timestep),
            len(model.nb_cell),
            len(model.nb_fleet),
        ]
    )
    n_output_charged3 = np.zeros(
        [
            len(model.nb_timestep),
            len(model.nb_cell),
            len(model.nb_fleet),
        ]
    )
    n_finished_charge3 = np.zeros(
        [
            len(model.nb_timestep),
            len(model.nb_cell),
            len(model.nb_fleet),
        ]
    )
    unused_capacity = np.zeros(
        [
            len(model.nb_timestep),
            len(model.nb_cell),
            len(model.nb_fleet),
        ]
    )
    for t in model.nb_timestep:
        for c in model.nb_cell:
            for f in model.nb_fleet:
                if (t, c, f) in model.key_set:

                    Q_arrived_vehicles[t, c, f] = model.Q_arrived_vehicles[
                        t, c, f
                    ].value
                    n_pass[t, c, f] = model.n_pass[t, c, f].value
                    n_in[t, c, f] = model.n_in[t, c, f].value
                    n_exit[t, c, f] = model.n_exit[t, c, f].value
                    n_incoming_vehicles[t, c, f] = model.n_incoming_vehicles[t, c, f].value
                    n_arrived_vehicles[t, c, f] = model.n_arrived_vehicles[t, c, f].value

                    if (t, c, f) in model.charging_cells_key_set:
                        n_wait[t, c, f] = model.n_wait[t, c, f].value
                        unused_capacity[t, c] = model.unused_capacity[t, c].value
                        n_wait_charge_next[t, c, f] = model.n_wait_charge_next[t, c, f].value
                        n_charge1[t, c, f] = model.n_charge1[t, c, f].value
                        n_charge2[t, c, f] = model.n_charge2[t, c, f].value
                        n_charge3[t, c, f] = model.n_charge3[t, c, f].value
                        E_charge1[t, c, f] = model.E_charge1[t, c, f].value
                        E_charge2[t, c, f] = model.E_charge2[t, c, f].value
                        E_charge3[t, c, f] = model.E_charge3[t, c, f].value
                        n_in_wait_charge[t, c, f] = model.n_in_wait_charge[t, c, f].value
                        n_charge1[t, c, f] = model.n_charge1[t, c, f].value
                        n_output_charged1[t, c, f] = model.n_output_charged1[t, c, f].value
                        n_finished_charge1[t, c, f] = model.n_finished_charge1[t, c, f].value
                        n_charge2[t, c, f] = model.n_charge2[t, c, f].value
                        n_output_charged2[t, c, f] = model.n_output_charged2[t, c, f].value
                        n_finished_charge2[t, c, f] = model.n_finished_charge2[t, c, f].value
                        n_charge3[t, c, f] = model.n_charge3[t, c, f].value
                        n_output_charged3[t, c, f] = model.n_output_charged3[t, c, f].value
                        n_finished_charge3[t, c, f] = model.n_finished_charge3[t, c, f].value

    return {
        "n_arrived_vehicles": n_arrived_vehicles,
        "unused_capacity": unused_capacity,
        "Q_arrived_vehicles": Q_arrived_vehicles,
        "n_wait": n_wait,
        "n_wait_charge_next": n_wait_charge_next,
        "n_charge1": n_charge1,
        "n_charge2": n_charge2,
        "n_charge3": n_charge3,
        "E_charge1": E_charge1,
        "E_charge2": E_charge2,
        "E_charge3": E_charge3,
        "n_pass": n_pass,
        "n_in": n_in,
        "n_exit": n_exit,
        "n_incoming_vehicles": n_incoming_vehicles,
        "n_arrived_vehicles": n_arrived_vehicles,
        "n_pass": n_pass,
        "n_in_wait_charge": n_in_wait_charge,
        "n_charge1": n_charge1,
        "n_output_charged1": n_output_charged1,
        "n_finished_charge1": n_finished_charge1,
        "n_charge2": n_charge2,
        "n_output_charged2": n_output_charged2,
        "n_finished_charge2": n_finished_charge2,
        "n_charge3": n_charge3,
        "n_output_charged3": n_output_charged3,
        "n_finished_charge3": n_finished_charge3
    }


def export_energy_comparison_to_excel(charging_model, time_of_optimization):
    data = []

    # Überprüfung der geladenen Energie und maximal mögliche Energie pro Zeitschritt und Zelle
    for t in charging_model.nb_timestep:
        for c in charging_model.nb_cell:
            if charging_model.cell_charging_cap[c] > 0:
                total_energy_charged = 0
                for f in charging_model.nb_fleet:
                    if (t, c, f) in charging_model.E_charge1:
                        total_energy_charged += (
                            charging_model.E_charge1[t, c, f].value +
                            charging_model.E_charge2[t, c, f].value +
                            charging_model.E_charge3[t, c, f].value
                        )
                max_energy_possible = charging_model.cell_charging_cap[c] * charging_model.time_resolution
                utilization_percentage = (total_energy_charged / max_energy_possible) * 100 if max_energy_possible > 0 else 0
                data.append([t, c, total_energy_charged, max_energy_possible, charging_model.cell_charging_cap[c], utilization_percentage])

    # Erstellen eines DataFrames
    df = pd.DataFrame(data, columns=["Zeitschritt", "Zelle", "Geladene Energie (kWh)", "Maximale mögliche Energie (kWh)", "Gesamtkapazität der Zelle (kW)", "Auslastung (%)"])

    # Exportieren des DataFrames in eine Excel-Datei
    excel_filename = f"results/energy_comparison_{time_of_optimization}.xlsx"
    df.to_excel(excel_filename, index=False)


def write_output_file_Energy_charged_each(model, time_of_optimization, filename):
    results = get_variables_from_model2(model)

    """Nur charged"""
    total_charged_each = pd.DataFrame()
    inds_of_all_cells = [ind for ind in model.nb_cell if model.cell_charging_cap[ind] >= 0]
    for ij in range(0, len(inds_of_all_cells)):
        d = {}
        c = inds_of_all_cells[ij]
        d["Cell ID"] = c
        d["Capacity"] = model.cell_charging_cap[c]

        for t in model.nb_timestep:
            d["Energy Charged1 at t=" + str(t)] = np.sum(results["E_charge1"][t, c, :])
            d["Energy Charged2 at t=" + str(t)] = np.sum(results["E_charge2"][t, c, :])
            d["Energy Charged3 at t=" + str(t)] = np.sum(results["E_charge3"][t, c, :])

        total_charged_each = total_charged_each.append(d, ignore_index=True)

    total_charged_each.to_excel("results/" + "Energy_charged_each" + "(" + time_of_optimization + ")" ".xlsx", index=False)