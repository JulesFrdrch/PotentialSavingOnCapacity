"""Python Dateien einfügen"""
from functions import *
from missing_functions import *
from Ausgabe import *
from objective_function import *
from plots import *
from input_capacity_reduction import *
from uncontrolled_cars import *
from unused_capacities import *
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


t_min_controlled = 0.1
t_min_random = 0.25
SOC_loading_controlled = 0.5
SOC_upper_threshold = 0.4
SOC_lower_threshold = 0.2
SOC_finished_charging_random = 0.8


"""Zelleninformationen einlesen"""
#cells = pd.read_csv("data/20220722-232828_cells_input.csv")        #30h
#cells = pd.read_csv("data/cell_input_sensivity_10.csv")                                                       #X5

cells = pd.read_csv("data/2024_A2_cell_input_empty_start_end.csv")                                       #A2 1 bis 14 und leere erste und letzte Zelle

print(cells)
print("ich bin eine TEST Ausgabe")

#cells = pd.read_csv("data/___cells_random_test2_allCS.csv")
nb_cells = len(cells) #Anzahl der Zellen

"""Definition Zeithorizont"""
time_frame = range(0, nb_time_steps + 1) #Zeithorizont Len=97

"""Flotteninformationen einlesen"""
#for fleet_filename in ["AT_A2_100(random 0.00, controlled 1.00)"]:                               # RND 0   CTR 100
#for fleet_filename in ["AT_A2_100(random 0.10, controlled 0.90)"]:                               # RND 10  CTR 90
#for fleet_filename in ["AT_A2_100(random 0.20, controlled 0.80)"]:                               # RND 20  CTR 80
#for fleet_filename in ["AT_A2_100(random 0.34, controlled 0.66)"]:                               # RND 34  CTR 66
#for fleet_filename in ["AT_A2_100(random 0.50, controlled 0.50)"]:                               # RND 50  CTR 50
#for fleet_filename in ["AT_A2_100(random 0.66, controlled 0.34)"]:                               # RND 66  CTR 34
#for fleet_filename in ["AT_A2_100(random 0.80, controlled 0.20)_s"]:                               # RND 80  CTR 20
#for fleet_filename in ["Sensivity_20_d"]:                               # RND 90 CTR 10
for fleet_filename in ["AT_A2_100(random 0.80, controlled 0.20)"]:                               # RND 100 CTR 0


    #fleet_df = read_fleets(pd.read_csv("data/" + fleet_filename + ".csv", delimiter=";"))
    fleet_df = read_fleets(pd.read_csv("data/data_x1/" + fleet_filename + ".csv", delimiter=";"))
    print(f"Die Flotten CSV Datei '{fleet_filename}.csv' konnte erfolgreich eingelesen werden")
    fleet_df["start_timestep"] = [int(el) for el in fleet_df.start_timestep]
    fleet_df["fleet_id"] = range(0, len(fleet_df))
    fleet_df["fleet_id"] = range(0, len(fleet_df))
    fleet_df = fleet_df.set_index("fleet_id")
    print(fleet_df)
    #fleet_df = fleet_df[fleet_df.index.isin(range(0, 50))]                                                                   #0, 3 = 3 Einträge (0,1,2)
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

    """Entscheidungsvariablen erstellen"""     #Dieses Dataframe (fleet_df) ist relevant für die Funktion und alle aufgerufenen Unterfunktionen
    print("\nDefining decision variables ...")           #Muss aber auch bei initialize_fleets geändert werden!
    t0 = time.time()
    add_decision_variables_and_create_key_sets_reals(charging_model,time_resolution,nb_fleets,nb_cells,nb_time_steps,SOC_min,SOC_max,fleet_df,cells,t_min,SOC_upper_threshold,SOC_lower_threshold,SOC_finished_charging_random,SOC_loading_controlled,t_min_random,t_min_controlled)
    #add_decision_variables_and_create_key_sets_integers(charging_model, time_resolution, nb_fleets, nb_cells,nb_time_steps, SOC_min, SOC_max, fleet_df, cells, t_min,SOC_upper_threshold, SOC_lower_threshold,SOC_finished_charging_random, SOC_loading_controlled, t_min_random,t_min_controlled)
    print("... took ", str(time.time() - t0), " sec")

    """Change of the input capacities"""
    # Reduktion aller Zellen-Kapazitäten
    #reduction_factor = 0.5  # Reduzieren der Kapazitäten um 50%, 0.8 wäre 20%
    #reduce_cell_capacities(charging_model, reduction_factor)

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
    t99 = time.time()  # Warum passiert hier nix?
    # initialize_cells(charging_model, cells)
    print("... took ", str(time.time() - t99), " sec")

    """Zustände definieren"""
    print("\nConstraining vehicles activities and states ...")
    t3 = time.time()
    """Variante Antonia"""
    #constr_vehicle_states(charging_model)                                                   # Zustände definiert wie auf Folie
    """Variante Julius"""
    constr_vehicle_states_with_uncontrolled_and_controlled_cars(charging_model)
    print("... took ", str(time.time() - t3), " sec")

    """Fehlende Funktionen in Antonias Programm"""
    print("\nFehlende Funktionen ...")
    t4 = time.time()
    #missing_functions(charging_model)                                                          #Fehlende Funktionen
    print("... took ", str(time.time() - t4), " sec")

    print("\nConstraining charging activity at cells ...")
    t5 = time.time()
    restraint_charging_capacity(charging_model)                                                 #Ladeinfrastruktur
    print("... took ", str(time.time() - t5), " sec")


    print("\nConstraining random fleet cars ...")
    t6 = time.time()
    #uncontrolled_cars_decision(charging_model)
    #control_random_fleet_vehicles_with_big_m(charging_model)
    # Füge die Steuerung der random_fleet hinzu
    #control_random_fleet_with_big_m(charging_model, M=1e6, charge_threshold=0.35)

    print("... took ", str(time.time() - t6), " sec")

    print("\nConstraining unused capacities ...")
    t7 = time.time()
    unused_capacities(charging_model)
    print("... took ", str(time.time() - t7), " sec")


    """Setting the queue to zero as a constraint"""
    set_n_wait_and_n_wait_charge_next_to_zero(charging_model)


    print("\nAdding objective function ...")
    t8 = time.time()
    #minimize_waiting_and_maximize_unused_capacities(charging_model)                                             #Zielfunktion
    maximize_unused_capacities(charging_model)
    #minimize_waiting(charging_model)
    #multi_objective_function(charging_model)


    print("... took ", str(time.time() - t8), " sec")
    # _file = open("Math-Equations.txt", "w", encoding="utf-8")
    # charging_model.pprint(ostream=_file, verbose=False, prefix="")
    # _file.close()
    print(build_model_size_report(charging_model))









    # Optimierung starten
    #opt_success = opt.solve(
    #    charging_model, report_timing=True, tee=True                                                                        #Lösen des Optimierungsmodells
    #)                                                                                                                       #report_time=True & tee=True aktivieren das Berichtswesen

    """Gurobi-Solver"""
    opt = SolverFactory("gurobi")
    # opt.options["TimeLimit"] = 14400                                                                                      #Zeitlimit in sec
    # opt.options["OptimalityTol"] = 1e-2                                                                                   #Toleranz für Optimallösung
    # opt.options["BarConvTol"] = 1e-11                                                                                     #Konvergenztoleranz für die Barrieremethode
    # opt.options["Cutoff"] = 1e-3                                                                                          #Wert ab dem der Solver die Suche nach einer Lösung abbricht
    # opt.options["CrossoverBasis"] = 0                                                                                     #Deaktiviert die Verwendung von Crossover-Basislösungen
    opt.options["Crossover"] = 0                                                                                            #Deaktiviert das Crossover-Verfahren
    opt.options["Method"] = 2                                                                                               #Legt die Methode fest: Siehe Dokument!
    opt_success = opt.solve(charging_model, report_timing=True, tee=True)


    #opt = SolverFactory("gurobi", solver_io="python")
    # opt.options["TimeLimit"] = 14400
    # opt.options["OptimalityTol"] = 1e-2
    #opt.options["BarConvTol"] = 1e-11
    # opt.options["Cutoff"] = 1e-3
    # opt.options["CrossoverBasis"] = 0
    #opt.options["Crossover"] = 0
    #opt.options["Method"] = 3
    #opt_success = opt.solve(
        #charging_model, report_timing=True, tee=True
    #)

    #opt = SolverFactory("gurobi")
    #opt.options["Method"] = 2  # Dual Simplex-Methode
    #opt.options["MIPGap"] = 0.05  # 5% Toleranz
    # opt.options["TimeLimit"] = 600  # Maximal 10 Minuten Laufzeit
    #opt.options["Heuristics"] = 0.1  # 10% der Zeit für Heuristiken
    #opt.options["Cuts"] = 1  # Moderate Anzahl von Cuts
    #opt.options["Threads"] = 8  # Nutzt 8 Threads
    #opt.options["Presolve"] = 2  # Aggressives Presolve
    #opt_success = opt.solve(
        #charging_model, report_timing=True, tee=True
    #)

    print(
        colored(
            "\nTotal time of model initialization and solution: "
            + str(time.time() - start)
            + "\n",
            "green",
        )
    )

    time_of_optimization = time.strftime("%Y%m%d-%H%M%S")


    """Ausgabe für unused_capacity_new (nur c)"""
    # Correctly iterating over the indices
    for c in charging_model.nb_cell:
        # Safely accessing the value if it's defined
        if c in charging_model.Unused_capacity_new:
            unused_capacity_value = charging_model.Unused_capacity_new[c].value
            if unused_capacity_value is not None:
                print(f"Cell {c}: Unused Capacity = {unused_capacity_value}")
            else:
                print(f"Cell {c}: Unused Capacity is not set.")
        else:
            print(f"Cell {c}: No such index in Unused_capacity_new")

    plot_unused_capacity_new(charging_model, time_of_optimization)

    """Ausgabe Datein erstellen"""
    print("\nAusgabe Datein erzeugen ...")
    t9 = time.time()

    """Info über alle Zellen"""
    print_cell_info(cells)

    """Ausgabe der unused_capacity (c, t) und Erzeugung eines Plots"""
    #print("Values of unused_capacity, cell_charging_cap, and diff:")
    for t in charging_model.nb_timestep:
        for c in charging_model.nb_cell:
            # Prüfen, ob der Index (t, c) in model.key_set vorhanden ist
            if (t, c) in charging_model.t_cs:
                unused_capacity_value = charging_model.unused_capacity[t, c].value
                cell_capacity = charging_model.cell_charging_cap[c]
                diff = unused_capacity_value - cell_capacity
                #print(f"timestep: {t}, cell: {c}, unused_capacity: {unused_capacity_value}, cell_capacity: {cell_capacity}, diff: {diff}")


    """Plots nur für gesamte Input Daten"""
    """Aufruf der Plot-Funktion für die Darstellung der unused_capacity"""
    # TODO: Funktion drauf machen
    #unused_capacity_35000_to_48000(charging_model, time_of_optimization)
    #unused_capacity_20000_to_35000(charging_model, time_of_optimization)
    #unused_capacity_15000_to_20000(charging_model, time_of_optimization)
    #unused_capacity_10000_to_15000(charging_model, time_of_optimization)
    #unused_capacity_7000_to_10000(charging_model, time_of_optimization)
    #unused_capacity_4000_to_7000(charging_model, time_of_optimization)
    #unused_capacity_2000_to_4000(charging_model, time_of_optimization)
    #unused_capacity_1000_to_2000(charging_model, time_of_optimization)
    #unused_capacity_up_to_1000(charging_model, time_of_optimization)

    """Plots nur für A2"""
    """Aufruf der Plot-Funktion für die Darstellung der unused_capacity"""
    #unused_capacity_A2(charging_model, time_of_optimization)

    """Detaillierte Ausgabe für einzelne Zellen"""
    #print("Debug: Unused capacity values for cell 13:")
    for t in charging_model.nb_timestep:
        if (t, 13) in charging_model.t_cs:
            unused_capacity_value = charging_model.unused_capacity[t, 13].value
            #print(f"Timestep: {t}, Unused capacity: {unused_capacity_value}")




    """AusgabeDateien"""

    """Controlled fleets 0-9"""
    write_fleet_energy_and_vehicle_details_to_xlsx_with_soc(charging_model, fleet_id=0,
                                                            filename="fleet_0_energy_details_with_soc.xlsx")
    write_fleet_energy_and_vehicle_details_to_xlsx_with_soc(charging_model, fleet_id=1,
                                                            filename="fleet_1_energy_details_with_soc.xlsx")
    write_fleet_energy_and_vehicle_details_to_xlsx_with_soc(charging_model, fleet_id=2,
                                                            filename="fleet_2_energy_details_with_soc.xlsx")
    write_fleet_energy_and_vehicle_details_to_xlsx_with_soc(charging_model, fleet_id=3,
                                                            filename="fleet_3_energy_details_with_soc.xlsx")
    write_fleet_energy_and_vehicle_details_to_xlsx_with_soc(charging_model, fleet_id=4,
                                                            filename="fleet_4_energy_details_with_soc.xlsx")
    write_fleet_energy_and_vehicle_details_to_xlsx_with_soc(charging_model, fleet_id=5,
                                                            filename="fleet_5_energy_details_with_soc.xlsx")
    write_fleet_energy_and_vehicle_details_to_xlsx_with_soc(charging_model, fleet_id=6,
                                                            filename="fleet_6_energy_details_with_soc.xlsx")
    write_fleet_energy_and_vehicle_details_to_xlsx_with_soc(charging_model, fleet_id=7,
                                                            filename="fleet_7_energy_details_with_soc.xlsx")
    write_fleet_energy_and_vehicle_details_to_xlsx_with_soc(charging_model, fleet_id=8,
                                                            filename="fleet_8_energy_details_with_soc.xlsx")
    write_fleet_energy_and_vehicle_details_to_xlsx_with_soc(charging_model, fleet_id=9,
                                                            filename="fleet_9_energy_details_with_soc.xlsx")

    write_fleet_energy_and_vehicle_charging_details_to_xlsx(charging_model, fleet_id=0,
                                                            filename="fleet_0_charging_details.xlsx")
    write_fleet_energy_and_vehicle_charging_details_to_xlsx(charging_model, fleet_id=1,
                                                            filename="fleet_1_charging_details.xlsx")
    write_fleet_energy_and_vehicle_charging_details_to_xlsx(charging_model, fleet_id=2,
                                                            filename="fleet_2_charging_details.xlsx")
    write_fleet_energy_and_vehicle_charging_details_to_xlsx(charging_model, fleet_id=3,
                                                            filename="fleet_3_charging_details.xlsx")
    write_fleet_energy_and_vehicle_charging_details_to_xlsx(charging_model, fleet_id=4,
                                                            filename="fleet_4_charging_details.xlsx")
    write_fleet_energy_and_vehicle_charging_details_to_xlsx(charging_model, fleet_id=5,
                                                            filename="fleet_5_charging_details.xlsx")
    write_fleet_energy_and_vehicle_charging_details_to_xlsx(charging_model, fleet_id=6,
                                                            filename="fleet_6_charging_details.xlsx")
    write_fleet_energy_and_vehicle_charging_details_to_xlsx(charging_model, fleet_id=7,
                                                            filename="fleet_7_charging_details.xlsx")
    write_fleet_energy_and_vehicle_charging_details_to_xlsx(charging_model, fleet_id=8,
                                                            filename="fleet_8_charging_details.xlsx")
    write_fleet_energy_and_vehicle_charging_details_to_xlsx(charging_model, fleet_id=9,
                                                            filename="fleet_9_charging_details.xlsx")


    write_combined_fleet_0_to_9_energy_and_vehicle_details_to_xlsx_with_soc(model=charging_model,filename="combined_fleets_0_to_9_movement_details.xlsx")
    write_combined_fleet_0_to_9_energy_and_vehicle_charging_details_to_xlsx(charging_model, filename="combined_fleets_0_to_9_charging_details.xlsx")


    """random fleets 10-19"""
    write_fleet_energy_and_vehicle_details_to_xlsx_with_soc(charging_model, fleet_id=10,
                                                            filename="fleet_10_energy_details_with_soc.xlsx")
    write_fleet_energy_and_vehicle_details_to_xlsx_with_soc(charging_model, fleet_id=11,
                                                            filename="fleet_11_energy_details_with_soc.xlsx")
    write_fleet_energy_and_vehicle_details_to_xlsx_with_soc(charging_model, fleet_id=12,
                                                            filename="fleet_12_energy_details_with_soc.xlsx")
    write_fleet_energy_and_vehicle_details_to_xlsx_with_soc(charging_model, fleet_id=13,
                                                            filename="fleet_13_energy_details_with_soc.xlsx")
    write_fleet_energy_and_vehicle_details_to_xlsx_with_soc(charging_model, fleet_id=14,
                                                            filename="fleet_14_energy_details_with_soc.xlsx")
    write_fleet_energy_and_vehicle_details_to_xlsx_with_soc(charging_model, fleet_id=15,
                                                            filename="fleet_15_energy_details_with_soc.xlsx")
    write_fleet_energy_and_vehicle_details_to_xlsx_with_soc(charging_model, fleet_id=16,
                                                            filename="fleet_16_energy_details_with_soc.xlsx")
    write_fleet_energy_and_vehicle_details_to_xlsx_with_soc(charging_model, fleet_id=17,
                                                            filename="fleet_17_energy_details_with_soc.xlsx")
    write_fleet_energy_and_vehicle_details_to_xlsx_with_soc(charging_model, fleet_id=18,
                                                            filename="fleet_18_energy_details_with_soc.xlsx")
    write_fleet_energy_and_vehicle_details_to_xlsx_with_soc(charging_model, fleet_id=19,
                                                            filename="fleet_19_energy_details_with_soc.xlsx")

    write_fleet_energy_and_vehicle_charging_details_to_xlsx(charging_model, fleet_id=10,
                                                            filename="fleet_10_charging_details.xlsx")
    write_fleet_energy_and_vehicle_charging_details_to_xlsx(charging_model, fleet_id=11,
                                                            filename="fleet_11_charging_details.xlsx")
    write_fleet_energy_and_vehicle_charging_details_to_xlsx(charging_model, fleet_id=12,
                                                            filename="fleet_12_charging_details.xlsx")
    write_fleet_energy_and_vehicle_charging_details_to_xlsx(charging_model, fleet_id=13,
                                                            filename="fleet_13_charging_details.xlsx")
    write_fleet_energy_and_vehicle_charging_details_to_xlsx(charging_model, fleet_id=14,
                                                            filename="fleet_14_charging_details.xlsx")
    write_fleet_energy_and_vehicle_charging_details_to_xlsx(charging_model, fleet_id=15,
                                                            filename="fleet_15_charging_details.xlsx")
    write_fleet_energy_and_vehicle_charging_details_to_xlsx(charging_model, fleet_id=16,
                                                            filename="fleet_16_charging_details.xlsx")
    write_fleet_energy_and_vehicle_charging_details_to_xlsx(charging_model, fleet_id=17,
                                                            filename="fleet_17_charging_details.xlsx")
    write_fleet_energy_and_vehicle_charging_details_to_xlsx(charging_model, fleet_id=18,
                                                            filename="fleet_18_charging_details.xlsx")
    write_fleet_energy_and_vehicle_charging_details_to_xlsx(charging_model, fleet_id=19,
                                                            filename="fleet_19_charging_details.xlsx")

    write_combined_fleet_10_to_19_energy_and_vehicle_charging_details_to_xlsx(charging_model, filename="combined_fleets_10_to_19_charging_details.xlsx")
    write_combined_fleet_10_to_19_energy_and_vehicle_details_to_xlsx_with_soc(model=charging_model,filename="combined_fleets_10_to_19_movement_details.xlsx")



    write_output_file_charging_stations(charging_model, time_of_optimization, fleet_filename)               #n_in_wait_charge, Queue=wait+wait_charge_next, Summe(n_charge)
                                                                                                            #Summe(E_charge), n_in, n_incoming_vehciles, n_exit, n_arrived, n_pass
    write_output_file_arrived_vehicles(charging_model, time_of_optimization, fleet_filename)                #n_arrived_vehicles
    write_output_file_incoming_vehicles(charging_model, time_of_optimization, fleet_filename)               #n_incoming_vehicles
    write_output_file_n_in(charging_model, time_of_optimization, fleet_filename)                            #n_in
    write_output_file_n_pass(charging_model, time_of_optimization, fleet_filename)                          #n_in
    write_output_file_Bewegung(charging_model, time_of_optimization, fleet_filename)                        #n_in, n_in_wait_charge, n_pass
    write_output_file_Charging(charging_model, time_of_optimization, fleet_filename)                        #Für 1,2,3 (n_charge, n_output_charged, n_finished)
    #write_output_file_Energy_charged(charging_model, time_of_optimization, fleet_filename)                   #Summe(E_charged1 + E_charged2 + E_charged3)
    write_output_file_Energy_charged_each(charging_model, time_of_optimization, fleet_filename)              #E_charged1, E_charged2, E_charged3
    write_output_file_Fleet_infos(charging_model, time_of_optimization, fleet_filename)                     #Flotten Infos
    export_energy_comparison_to_excel(charging_model, time_of_optimization)


    # Ergebnisse aus dem Modell holen
    results = get_variables_from_model2(charging_model)

    # Diagramme für Fahrzeug_Anzahl_Variablen
    plot_variable_over_time(charging_model, 'n_pass', results, time_of_optimization, 'n_pass')
    plot_variable_over_time(charging_model, 'n_in', results, time_of_optimization, 'n_in')
    plot_variable_over_time(charging_model, 'n_wait', results, time_of_optimization, 'n_wait')
    plot_variable_over_time(charging_model, 'n_wait_charge_next', results, time_of_optimization, 'n_wait_charge_next')
    plot_variable_over_time(charging_model, 'n_incoming_vehicles', results, time_of_optimization, 'n_incoming_vehicles')
    plot_variable_over_time(charging_model, 'n_arrived_vehicles', results, time_of_optimization, 'n_arrived_vehicles')
    plot_variable_over_time(charging_model, 'n_in_wait_charge', results, time_of_optimization, 'n_in_wait_charge')
    plot_combined_variable_over_time(charging_model, results, time_of_optimization)

    # Diagramme für die wartenden Fahrzeuge in den Zellen mit CS der A2
    plot_waiting_vehicles_cell1(charging_model, time_of_optimization)
    plot_waiting_vehicles_cell2(charging_model, time_of_optimization)
    plot_waiting_vehicles_cell3(charging_model, time_of_optimization)
    plot_waiting_vehicles_cell4(charging_model, time_of_optimization)
    plot_waiting_vehicles_cell7(charging_model, time_of_optimization)
    plot_waiting_vehicles_cell11(charging_model, time_of_optimization)
    plot_waiting_vehicles_cell13(charging_model, time_of_optimization)
    plot_waiting_vehicles_cell14(charging_model, time_of_optimization)


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
                        n_wait_charge_next[t, c, f] = charging_model.n_wait_charge_next[t, c, f].value

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

    print("Departing and Arriving Vehicles:")
    print("Amount of incoming vehicles:", np.sum(n_incoming_vehicles))
    print("Amount of arriving vehicles:", np.sum(n_arrived_vehicles))
    print("")
    print("Overview of the Energy Consumption and Charged")
    total_cons = np.sum(Q_pass) + np.sum(Q_finished_charging) - np.sum(Q_exit) + (
                np.sum(Q_in_charge_wait) - np.sum(Q_in_wait) - np.sum(Q_in_charge))
    print("Total energy consumed:", total_cons)

    print(
        "Total energy charged:",
        sum(sum(sum(E_charge1))) + sum(sum(sum(E_charge2))) + sum(sum(sum(E_charge3))),)

    calculate_and_print_energy_split_by_fleet(charging_model)

    print("check", np.sum(Q_arrived_vehicles) - np.sum(Q_incoming_vehicles),
          sum(sum(sum(E_charge1))) + sum(sum(sum(E_charge2))) + sum(sum(sum(E_charge3))) - total_cons)
    print("the check considers Q_arrived_vehicles - incoming_vehicles and the total charged energy - total energy consumed")

    print("")
    print("Total energy consumped by passing the CS, E_consumed_pass:", np.sum(E_consumed_pass))
    print("Total amount of vehicles passing the CS, n_pass:", np.sum(n_pass))
    print("")
    print("Total energy consumed by entering the CS, E_consumed_charge_wait:", np.sum(E_consumed_charge_wait))
    print("Total energy consumed by leaving the CS, E_consumed_exit_charge:", np.sum(E_consumed_exit_charge))
    print("Total amount of vehicle into the CS, n_in_wait_charge:", np.sum(n_in_wait_charge))
    #calculate_and_print_vehicles_waiting_to_charge_by_fleet(charging_model)                                                kann raus
    calculate_and_print_vehicles_waiting_to_charge_by_fleet_new(charging_model)


    print("")
    print("Overview of the waiting vehicles")
    print("Total amount of waiting vehicles, n_wait:", np.sum(n_wait))
    print("Total amount of vehicles charging next, n_wait_charge_next:", np.sum(n_wait_charge_next))
    print("")
    print("Total amount ot vehicles finishing charging, n_finished_charging:", np.sum(n_finished_charging))
    print("")

    print("Q_in_charge_wait ist:", np.sum(Q_in_charge_wait))



    #print("unused ist:", np.sum(unused_capacity))

    # Initialisiere die Summe
    total_unused_capacity_new = 0
    # Schleife durch alle Zellen, um die ungenutzten Kapazitäten zu summieren
    for c in charging_model.nb_cell:
        if c in charging_model.Unused_capacity_new:
            unused_capacity_value = charging_model.Unused_capacity_new[c].value
            if unused_capacity_value is not None:
                total_unused_capacity_new += unused_capacity_value

    # Ausgabe der Summe in der Konsole
    print(f"Total: {total_unused_capacity_new}")

    print("Amount of incoming vehicles:", np.sum(n_incoming_vehicles))

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
