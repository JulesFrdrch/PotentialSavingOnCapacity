"""Python Dateien einfügen"""
from Funktionen import *
from FehlendeFunktionen import *
from Ausgabe import *
from Zielfunktion import *
from plots import *
from input_capacity_reduction import *
from uncontrolled_cars import *
#from _optimization_utils import write_output_files
#from _optimization_utils import *

"""Bibliotheken einfügen"""
import pandas as pd
import numpy as np
import time
import os
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

"""Grunddaten"""
tkomplett = time.time()
SOC_min = 0.1
SOC_max = 1
t_min = 0.2
time_resolution = 0.25  #Zeitauflösung von 15 Minuten
nb_time_steps = 120     #120 Time Steps = 30 Stunden, 96 für 24 Stunden



"""Zelleninformationen einlesen"""
#cells = pd.read_csv("data/Zellen_input_reduzierung_test.csv")
cells = pd.read_csv("data/20220722-232828_cells_input.csv")
#cells = pd.read_csv("data/___cells_only3flotten.csv")


#cells = pd.read_csv("data/___cells_random_test2_allCS.csv")
nb_cells = len(cells) #Anzahl der Zellen

"""Definition Zeithorizont"""
time_frame = range(0, nb_time_steps + 1) #Zeithorizont Len=121

"""Flotteninformationen einlesen"""
#for fleet_filename in ["summer_workdayfleet_input_20220719_compressed_probe"]:
for fleet_filename in ["___flotten_random_test2"]:
#for fleet_filename in ["___flotten_random_test3"]:                                                     #200.000er Test auf Flotte 1
#for fleet_filename in ["___flotten_wenige_Fahrzeuge"]:

    fleet_df = read_fleets(pd.read_csv("data/" + fleet_filename + ".csv", delimiter=";"))
    print('Die Flotten CSV Datei konnte erfolgreich eingelesen werden')
    fleet_df["start_timestep"] = [int(el) for el in fleet_df.start_timestep]
    fleet_df["fleet_id"] = range(0, len(fleet_df))
    fleet_df["fleet_id"] = range(0, len(fleet_df))
    fleet_df = fleet_df.set_index("fleet_id")
    print(fleet_df)
    fleet_df = fleet_df[fleet_df.index.isin(range(0, 50))]                                                                   #0, 3 = 3 Einträge (0,1,2)
    nb_fleets = len(fleet_df)
    print("Anzahl der Time Steps:\t",nb_time_steps,"Dies entspricht:",nb_time_steps*0.25,"Stunde" ,"\nAnzahl der Zellen:\t\t", nb_cells,"\nAnzahl der Flotten:\t\t" ,nb_fleets)
    #print(fleet_df.loc[2]) #Eine Zeile auslesen

    """Aufteilen der Flotte in 2 Flotten"""                                                                                 #Glaube gar nicht Relevant
    random_fleet_df=fleet_df.loc[lambda fleet_df:fleet_df['random_fleet'] == 1]  #Teilt fleet_df auf und speichter nur die random Flotten auf dem neuen df
    print('\nDaraus folgt ein DataFrame mit den random User welche wie folgt aussieht:')
    print(random_fleet_df)
    controlled_fleet_df=fleet_df.loc[lambda fleet_df:fleet_df['random_fleet'] == 0]
    print('\nDaraus folgt ein DataFrame mit den controlled User welche wie folgt aussieht:')
    print(controlled_fleet_df)

    """Ausgabe der DataFrame Dimensionen"""
    print("")
    print('Die Dimension des Gesamten   DataFrames ist: '+ str(fleet_df.shape))
    print('Die Dimension des Random     DataFrames ist: '+ str(random_fleet_df.shape))
    print('Die Dimension des Controlled DataFrames ist: '+ str(controlled_fleet_df.shape))


    """OPTIMIZATION"""
    charging_model = ConcreteModel()

    """Zeitmessung"""
    start = time.time() #Zeit in Sekunden die seit dem Epoch vergangen sind

    """Entscheidungsvariablen erstellen"""
    print("\nDefining decision variables ...")
    t0 = time.time()
    add_decision_variables_and_create_key_sets(
        charging_model,
        time_resolution,
        nb_fleets,
        nb_cells,
        nb_time_steps,
        SOC_min,
        SOC_max,
        fleet_df,   #Dieses Dataframe ist relevant für die Funktion und alle aufgerufenen Unterfunktionen
        cells,      #Muss aber auch bei initialize_fleets geändert werden!
        t_min,
    )
    print("... took ", str(time.time() - t0), " sec")

    """Change of the input capacities"""
    # Reduktion aller Zellen-Kapazitäten
    reduction_factor = 0.5  # Reduzieren der Kapazitäten um 50%, 0.8 wäre 20%
    reduce_cell_capacities(charging_model, reduction_factor)

    # Reduktion ausgewählter Zellen-Kapazitäten
    #selected_cells = [2, 3, 4, 5]  # IDs der Zellen, deren Kapazitäten reduziert werden sollen
    #reduction_factor = 0.8  # Reduzieren der Kapazitäten um 20%
    #reduce_selected_cell_capacities(charging_model, selected_cells, reduction_factor)

    # Löschen der Kapazität ausgewählter Zellen
    #selected_cells_to_zero = [13, 19, 10, 9]  # IDs der Zellen, deren Kapazität auf 0 gesetzt werden soll
    #set_selected_cells_to_zero_capacity(charging_model, selected_cells_to_zero)




    """Flotten initialisieren"""
    print("\nInitializing fleets ...")
    t2 = time.time()  # Flotten und ihre Eigenschaften festlegen
    initialize_fleets(charging_model, fleet_df)  # Abfahrt der Autos
    print("... took ", str(time.time() - t2), " sec")

    """Zellen initialisieren"""
    print("\nInitializing cell geometry ...")
    t3 = time.time()  # Warum passiert hier nix?
    # initialize_cells(charging_model, cells)
    print("... took ", str(time.time() - t3), " sec")

    """Zustände definieren"""
    print("\nConstraining vehicles activities and states ...")
    t4 = time.time()
    constr_vehicle_states(charging_model)                                                       #Zustände definiert wie auf Folie
    print("... took ", str(time.time() - t4), " sec")

    """Fehlende Funktionen in Antonias Programm"""
    print("\nFehlende Funktionen ...")
    t5 = time.time()
    FehlendeFunktionen(charging_model)                                                          #Fehlende Funktionen
    print("... took ", str(time.time() - t5), " sec")

    print("\nConstraining charging activity at cells ...")
    t6 = time.time()
    restraint_charging_capacity(charging_model)                                                 #Ladeinfrastruktur
    print("... took ", str(time.time() - t6), " sec")


    print("\nConstraining random fleet cars ...")
    t7 = time.time()
    #define_fleet_37_routing_constraints(charging_model)                                            #uncontrolled_cars
    print("... took ", str(time.time() - t7), " sec")



    print("\nAdding objective function ...")
    t8 = time.time()
    minimize_waiting_and_charging(charging_model)                                             #Zielfunktion

    cost_per_kw = 0.1  # Beispielwert für die Kosten pro kW, anpassbar
    #minimize_waiting_and_maximize_savings(charging_model, cost_per_kw)

    print("... took ", str(time.time() - t8), " sec")
    # _file = open("Math-Equations.txt", "w", encoding="utf-8")
    # charging_model.pprint(ostream=_file, verbose=False, prefix="")
    # _file.close()
    print(build_model_size_report(charging_model))



    """Gurobi-Solver"""
    opt = SolverFactory("gurobi")
    # opt.options["TimeLimit"] = 14400                                                                                      #Zeitlimit in sec
    # opt.options["OptimalityTol"] = 1e-2                                                                                   #Toleranz für Optimallösung
    # opt.options["BarConvTol"] = 1e-11                                                                                     #Konvergenztoleranz für die Barrieremethode
    # opt.options["Cutoff"] = 1e-3                                                                                          #Wert ab dem der Solver die Suche nach einer Lösung abbricht
    # opt.options["CrossoverBasis"] = 0                                                                                     #Deaktiviert die Verwendung von Crossover-Basislösungen
    opt.options["Crossover"] = 0                                                                                            #Deaktiviert das Crossover-Verfahren
    opt.options["Method"] = 2                                                                                               #Legt die Methode fest: Siehe Dokument!
    opt_success = opt.solve(
        charging_model, report_timing=True, tee=True                                                                        #Lösen des Optimierungsmodells
    )                                                                                                                       #report_time=True & tee=True aktiieren das Berichtswesen

    print(
        colored(
            "\nTotal time of model initialization and solution: "
            + str(time.time() - start)
            + "\n",
            "green",
        )
    )

    time_of_optimization = time.strftime("%Y%m%d-%H%M%S")

    """Ausgabe Datein erstellen"""
    print("\nAusgabe Datein erzeugen ...")
    t9 = time.time()

    """Info über alle Zellen"""
    print_cell_info(cells)

    """Ausgabe der unused_capacity und Erzeugung eines Plots"""
    print("Values of unused_capacity, cell_charging_cap, and diff:")
    for t in charging_model.nb_timestep:
        for c in charging_model.nb_cell:
            # Prüfen, ob der Index (t, c) in model.key_set vorhanden ist
            if (t, c) in charging_model.t_cs:
                unused_capacity_value = charging_model.unused_capacity[t, c].value
                cell_capacity = charging_model.cell_charging_cap[c]
                diff = unused_capacity_value - cell_capacity
                print(f"timestep: {t}, cell: {c}, unused_capacity: {unused_capacity_value}, cell_capacity: {cell_capacity}, diff: {diff}")

    #print_fleet_39_activity(charging_model)

    """Aufruf der Plot-Funktion für die Darstellung der unused_capacity"""
    unused_capacity_35000_to_48000(charging_model, time_of_optimization)
    unused_capacity_20000_to_35000(charging_model, time_of_optimization)
    unused_capacity_15000_to_20000(charging_model, time_of_optimization)
    unused_capacity_10000_to_15000(charging_model, time_of_optimization)
    unused_capacity_7000_to_10000(charging_model, time_of_optimization)
    unused_capacity_4000_to_7000(charging_model, time_of_optimization)
    unused_capacity_2000_to_4000(charging_model, time_of_optimization)
    unused_capacity_1000_to_2000(charging_model, time_of_optimization)
    unused_capacity_up_to_1000(charging_model, time_of_optimization)

    """Detaillierte Ausgabe für einzelne Zellen"""
    print("Debug: Unused capacity values for cell 13:")
    for t in charging_model.nb_timestep:
        if (t, 13) in charging_model.t_cs:
            unused_capacity_value = charging_model.unused_capacity[t, 13].value
            print(f"Timestep: {t}, Unused capacity: {unused_capacity_value}")




    """AusgabeDateien"""
    write_output_file_charging_stations(charging_model, time_of_optimization, fleet_filename)               #n_in_wait_charge, Queue=wait+wait_charge_next, Summe(n_charge)
                                                                                                            #Summe(E_charge), n_in, n_incoming_vehciles, n_exit, n_arrived, n_pass
    write_output_file_arrived_vehicles(charging_model, time_of_optimization, fleet_filename)                #n_arrived_vehicles
    write_output_file_incoming_vehicles(charging_model, time_of_optimization, fleet_filename)               #n_incoming_vehicles
    write_output_file_n_in(charging_model, time_of_optimization, fleet_filename)                            #n_in
    write_output_file_n_pass(charging_model, time_of_optimization, fleet_filename)                          #n_in
    write_output_file_Bewegung(charging_model, time_of_optimization, fleet_filename)                        #n_in, n_in_wait_charge, n_pass
    write_output_file_Charging(charging_model, time_of_optimization, fleet_filename)                        #Für 1,2,3 (n_charge, n_output_charged, n_finished)
    write_output_file_Energy_charged(charging_model, time_of_optimization, fleet_filename)                   #Summe(E_charged1 + E_charged2 + E_charged3)
    write_output_file_Energy_charged_each(charging_model, time_of_optimization, fleet_filename)              #E_charged1, E_charged2, E_charged3
    write_output_file_Fleet_infos(charging_model, time_of_optimization, fleet_filename)                     #Flotten Infos
    export_energy_comparison_to_excel(charging_model, time_of_optimization)


    # Ergebnisse aus dem Modell holen
    results = get_variables_from_model2(charging_model)

    # Diagramme für Fahrzeug_Anzahl_Variablen
    plot_variable_over_time(charging_model, 'n_pass', results, time_of_optimization, 'n_pass')
    plot_variable_over_time(charging_model, 'n_in', results, time_of_optimization, 'n_in')
    plot_variable_over_time(charging_model, 'n_incoming_vehicles', results, time_of_optimization, 'n_incoming_vehicles')
    plot_variable_over_time(charging_model, 'n_arrived_vehicles', results, time_of_optimization, 'n_arrived_vehicles')
    plot_variable_over_time(charging_model, 'n_in_wait_charge', results, time_of_optimization, 'n_in_wait_charge')
    plot_combined_variable_over_time(charging_model, results, time_of_optimization)

    print("... took ", str(time.time() - t9), " sec")

    print("")


    n_exit = np.zeros(
        [
            len(charging_model.nb_timestep),
            len(charging_model.nb_cell),
            len(charging_model.nb_fleet),
        ]
    )
    n_to_charge = np.zeros(
        [
            len(charging_model.nb_timestep),
            len(charging_model.nb_cell),
            len(charging_model.nb_fleet),
        ]
    )
    n_incoming_vehicles = np.zeros(
        [
            len(charging_model.nb_timestep),
            len(charging_model.nb_cell),
            len(charging_model.nb_fleet),
        ]
    )
    Q_in_pass = np.zeros(
        [
            len(charging_model.nb_timestep),
            len(charging_model.nb_cell),
            len(charging_model.nb_fleet),
        ]
    )
    E_consumed_pass = np.zeros(
        [
            len(charging_model.nb_timestep),
            len(charging_model.nb_cell),
            len(charging_model.nb_fleet),
        ]
    )
    n_in_wait_charge = np.zeros(
        [
            len(charging_model.nb_timestep),
            len(charging_model.nb_cell),
            len(charging_model.nb_fleet),
        ]
    )
    n_in = np.zeros(
        [
            len(charging_model.nb_timestep),
            len(charging_model.nb_cell),
            len(charging_model.nb_fleet),
        ]
    )

    n_in_charge = np.zeros(
        [
            len(charging_model.nb_timestep),
            len(charging_model.nb_cell),
            len(charging_model.nb_fleet),
        ]
    )

    n_to_charge = np.zeros(
        [
            len(charging_model.nb_timestep),
            len(charging_model.nb_cell),
            len(charging_model.nb_fleet),
        ]
    )

    Q_exit_pass = np.zeros(
        [
            len(charging_model.nb_timestep),
            len(charging_model.nb_cell),
            len(charging_model.nb_fleet),
        ]
    )

    n_charge1 = np.zeros(
        [
            len(charging_model.nb_timestep),
            len(charging_model.nb_cell),
            len(charging_model.nb_fleet),
        ]
    )

    n_charge2 = np.zeros(
        [
            len(charging_model.nb_timestep),
            len(charging_model.nb_cell),
            len(charging_model.nb_fleet),
        ]
    )

    n_arrived_vehicles = np.zeros(
        [
            len(charging_model.nb_timestep),
            len(charging_model.nb_cell),
            len(charging_model.nb_fleet),
        ]
    )

    n_finished_charging = np.zeros(
        [
            len(charging_model.nb_timestep),
            len(charging_model.nb_cell),
            len(charging_model.nb_fleet),
        ]
    )

    Q_input_charge1 = np.zeros(
        [
            len(charging_model.nb_timestep),
            len(charging_model.nb_cell),
            len(charging_model.nb_fleet),
        ]
    )

    Q_in_charge = np.zeros(
        [
            len(charging_model.nb_timestep),
            len(charging_model.nb_cell),
            len(charging_model.nb_fleet),
        ]
    )

    Q_in = np.zeros(
        [
            len(charging_model.nb_timestep),
            len(charging_model.nb_cell),
            len(charging_model.nb_fleet),
        ]
    )

    Q_incoming_vehicles = np.zeros(
        [
            len(charging_model.nb_timestep),
            len(charging_model.nb_cell),
            len(charging_model.nb_fleet),
        ]
    )
    Q_pass = np.zeros(
        [
            len(charging_model.nb_timestep),
            len(charging_model.nb_cell),
            len(charging_model.nb_fleet),
        ]
    )

    Q_exit = np.zeros(
        [
            len(charging_model.nb_timestep),
            len(charging_model.nb_cell),
            len(charging_model.nb_fleet),
        ]
    )
    Q_exit_charged = np.zeros(
        [
            len(charging_model.nb_timestep),
            len(charging_model.nb_cell),
            len(charging_model.nb_fleet),
        ]
    )

    Q_output_charge1 = np.zeros(
        [
            len(charging_model.nb_timestep),
            len(charging_model.nb_cell),
            len(charging_model.nb_fleet),
        ]
    )
    Q_output_charge1 = np.zeros(
        [
            len(charging_model.nb_timestep),
            len(charging_model.nb_cell),
            len(charging_model.nb_fleet),
        ]
    )
    Q_output_charge2 = np.zeros(
        [
            len(charging_model.nb_timestep),
            len(charging_model.nb_cell),
            len(charging_model.nb_fleet),
        ]
    )
    E_charge1 = np.zeros(
        [
            len(charging_model.nb_timestep),
            len(charging_model.nb_cell),
            len(charging_model.nb_fleet),
        ]
    )

    Q_arrived_vehicles = np.zeros(
        [
            len(charging_model.nb_timestep),
            len(charging_model.nb_cell),
            len(charging_model.nb_fleet),
        ]
    )
    n_pass = np.zeros(
        [
            len(charging_model.nb_timestep),
            len(charging_model.nb_cell),
            len(charging_model.nb_fleet),
        ]
    )
    n_exit_charge = np.zeros(
        [
            len(charging_model.nb_timestep),
            len(charging_model.nb_cell),
            len(charging_model.nb_fleet),
        ]
    )
    n_wait_charge_next = np.zeros(
        [
            len(charging_model.nb_timestep),
            len(charging_model.nb_cell),
            len(charging_model.nb_fleet),
        ]
    )
    n_in_wait = np.zeros(
        [
            len(charging_model.nb_timestep),
            len(charging_model.nb_cell),
            len(charging_model.nb_fleet),
        ]
    )

    E_consumed_charge_wait = np.zeros(
        [
            len(charging_model.nb_timestep),
            len(charging_model.nb_cell),
            len(charging_model.nb_fleet),
        ]
    )

    E_consumed_exit_charge = np.zeros(
        [
            len(charging_model.nb_timestep),
            len(charging_model.nb_cell),
            len(charging_model.nb_fleet),
        ]
    )

    Q_wait_charge_next = np.zeros(
        [
            len(charging_model.nb_timestep),
            len(charging_model.nb_cell),
            len(charging_model.nb_fleet),
        ]
    )

    n_wait_charge_next = np.zeros(
        [
            len(charging_model.nb_timestep),
            len(charging_model.nb_cell),
            len(charging_model.nb_fleet),
        ]
    )

    n_wait = np.zeros(
        [
            len(charging_model.nb_timestep),
            len(charging_model.nb_cell),
            len(charging_model.nb_fleet),
        ]
    )

    n_charge2 = np.zeros(
        [
            len(charging_model.nb_timestep),
            len(charging_model.nb_cell),
            len(charging_model.nb_fleet),
        ]
    )
    E_charge2 = np.zeros(
        [
            len(charging_model.nb_timestep),
            len(charging_model.nb_cell),
            len(charging_model.nb_fleet),
        ]
    )

    Q_out = np.zeros(
        [
            len(charging_model.nb_timestep),
            len(charging_model.nb_cell),
            len(charging_model.nb_fleet),
        ]
    )
    E_charge3 = np.zeros(
        [
            len(charging_model.nb_timestep),
            len(charging_model.nb_cell),
            len(charging_model.nb_fleet),
        ]
    )
    Q_wait = np.zeros(
        [
            len(charging_model.nb_timestep),
            len(charging_model.nb_cell),
            len(charging_model.nb_fleet),
        ]
    )
    Q_in_wait = np.zeros(
        [
            len(charging_model.nb_timestep),
            len(charging_model.nb_cell),
            len(charging_model.nb_fleet),
        ]
    )

    Q_in_charge_wait = np.zeros(
        [
            len(charging_model.nb_timestep),
            len(charging_model.nb_cell),
            len(charging_model.nb_fleet),
        ]
    )

    n_out = np.zeros(
        [
            len(charging_model.nb_timestep),
            len(charging_model.nb_cell),
            len(charging_model.nb_fleet),
        ]
    )

    n_charge3 = np.zeros(
        [
            len(charging_model.nb_timestep),
            len(charging_model.nb_cell),
            len(charging_model.nb_fleet),
        ]
    )

    Q_input_charge2 = np.zeros(
        [
            len(charging_model.nb_timestep),
            len(charging_model.nb_cell),
            len(charging_model.nb_fleet),
        ]
    )

    n_output_charged1 = np.zeros(
        [
            len(charging_model.nb_timestep),
            len(charging_model.nb_cell),
            len(charging_model.nb_fleet),
        ]
    )
    Q_finished_charging = np.zeros(
        [
            len(charging_model.nb_timestep),
            len(charging_model.nb_cell),
            len(charging_model.nb_fleet),
        ]
    )
    n_incoming_vehicles = np.zeros(
        [
            len(charging_model.nb_timestep),
            len(charging_model.nb_cell),
            len(charging_model.nb_fleet),
        ]
    )
    unused_capacity = np.zeros(
        [
            len(charging_model.nb_timestep),
            len(charging_model.nb_cell),
            len(charging_model.nb_fleet),
        ]
    )

    for t in charging_model.nb_timestep:
        for c in charging_model.nb_cell:
            for f in charging_model.nb_fleet:
                if (t, c, f) in charging_model.key_set:

                    Q_out[t, c, f] = charging_model.Q_out[t, c, f].value

                    n_incoming_vehicles[t, c, f] = charging_model.n_incoming_vehicles[t, c, f].value

                    n_arrived_vehicles[t, c, f] = charging_model.n_arrived_vehicles[t, c, f].value

                    Q_arrived_vehicles[t, c, f] = charging_model.Q_arrived_vehicles[t, c, f].value
                    Q_incoming_vehicles[t, c, f] = charging_model.Q_incoming_vehicles[t, c, f].value
                    Q_exit[t, c, f] = charging_model.Q_exit[t, c, f].value
                    Q_pass[t, c, f] = charging_model.Q_pass[t, c, f].value
                    n_pass[t, c, f] = charging_model.n_pass[t, c, f].value

                    E_consumed_pass[t, c, f] = charging_model.E_consumed_pass[t, c, f].value

                    if (t, c, f) in charging_model.charging_cells_key_set:
                        n_wait[t, c, f] = charging_model.n_wait[t, c, f].value

                        E_charge1[t, c, f] = charging_model.E_charge1[t, c, f].value
                        n_charge1[t, c, f] = charging_model.n_charge1[t, c, f].value
                        E_charge1[t, c, f] = charging_model.E_charge1[t, c, f].value
                        E_charge2[t, c, f] = charging_model.E_charge2[t, c, f].value
                        E_charge3[t, c, f] = charging_model.E_charge3[t, c, f].value

                        E_consumed_charge_wait[t, c, f] = charging_model.E_consumed_charge_wait[t, c, f].value
                        E_consumed_exit_charge[t, c, f] = charging_model.E_consumed_exit_charge[t, c, f].value
                        unused_capacity[t, c] = charging_model.unused_capacity[t, c].value

                        Q_finished_charging[t, c, f] = charging_model.Q_finished_charging[t, c, f].value
                        n_finished_charging[t, c, f] = charging_model.n_finished_charging[t, c, f].value
                        Q_in_charge[t, c, f] = charging_model.Q_in_charge[t, c, f].value
                        Q_in_charge_wait[t, c, f] = charging_model.Q_in_charge_wait[t, c, f].value
                        Q_in_wait[t, c, f] = charging_model.Q_in_wait[t, c, f].value
                        n_in_wait_charge[t, c, f] = charging_model.n_in_wait_charge[t, c, f].value

    print("n_arrived_vehicles", np.sum(n_arrived_vehicles))
    print("n_incoming_vehicles", np.sum(n_incoming_vehicles))

    total_cons = np.sum(Q_pass) + np.sum(Q_finished_charging) - np.sum(Q_exit) + (
                np.sum(Q_in_charge_wait) - np.sum(Q_in_wait) - np.sum(Q_in_charge))
    print("Total energy consumed", total_cons)

    print(
        "Total E_charged",
        sum(sum(sum(E_charge1))) + sum(sum(sum(E_charge2))) + sum(sum(sum(E_charge3))),
    )
    print("check", np.sum(Q_arrived_vehicles) - np.sum(Q_incoming_vehicles),
          sum(sum(sum(E_charge1))) + sum(sum(sum(E_charge2))) + sum(sum(sum(E_charge3))) - total_cons)

    print("E_consumed_pass ist:", np.sum(E_consumed_pass))
    print("n_pass ist:", np.sum(n_pass))
    print("E_consumed_charge_wait ist:", np.sum(E_consumed_charge_wait))
    print("n_in_wait_charge ist:", np.sum(n_in_wait_charge))
    print("Q_in_charge_wait ist:", np.sum(Q_in_charge_wait))
    print("E_consumed_exit_charge ist:", np.sum(E_consumed_exit_charge))
    print("n_finished_charging ist:", np.sum(n_finished_charging))
    print("unused ist:", np.sum(unused_capacity))



    # Endzeitpunkt
    end_time = time.time()

    # Berechnung der Gesamtzeit
    total_time_seconds = end_time - tkomplett
    total_time_minutes = total_time_seconds / 60
    total_time_hours = total_time_minutes / 60

    # Ergebnisprüfung und Ausgabe der Berechnungszeit
    print(colored("\nTotal time of model initialization and solution:", "green"))
    print(f"{total_time_seconds:.2f} seconds")
    print(f"{total_time_minutes:.2f} minutes")
    print(f"{total_time_hours:.2f} hours")
