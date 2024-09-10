import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from Ausgabe import *



def unused_capacity_35000_to_48000(charging_model, time_of_optimization):
    # Stellen sicher, dass nb_timestep und nb_cell als Listen oder Bereiche vorliegen
    timesteps = list(charging_model.nb_timestep)
    cells = list(charging_model.nb_cell)

    # Daten für die Visualisierung berechnen
    nb_timestep = len(timesteps)
    nb_cell = len(cells)
    unused_capacity_values = np.zeros((nb_timestep, nb_cell))

    for t in timesteps:
        for c in cells:
            if (t, c) in charging_model.t_cs:
                unused_capacity_values[t, c] = charging_model.unused_capacity[t, c].value

    plt.figure(figsize=(14, 8))

    for c in [9, 29, 34]:  #Cells 9, 29 and 34
        plt.plot(timesteps, unused_capacity_values[:, c], label=f'Cell {c}')

    plt.xlabel('Timestep')
    plt.ylabel('Unused Capacity')
    plt.title('Unused Capacity Over Time for Cells with at least 35.000 Watt')
    plt.legend(loc='upper right')
    plt.grid(True)

    # Verzeichnis für die Speicherung erstellen, falls es nicht existiert
    output_dir = os.path.join('results', 'pictures')
    os.makedirs(output_dir, exist_ok=True)

    # Speichern des Diagramms
    output_path = os.path.join(output_dir, f'unused_capacity_over_35000W_time{time_of_optimization}.png')
    plt.savefig(output_path)
    #plt.show()

def unused_capacity_20000_to_35000(charging_model, time_of_optimization):
    # Stellen sicher, dass nb_timestep und nb_cell als Listen oder Bereiche vorliegen
    timesteps = list(charging_model.nb_timestep)
    cells = list(charging_model.nb_cell)

    # Daten für die Visualisierung berechnen
    nb_timestep = len(timesteps)
    nb_cell = len(cells)
    unused_capacity_values = np.zeros((nb_timestep, nb_cell))

    for t in timesteps:
        for c in cells:
            if (t, c) in charging_model.t_cs:
                unused_capacity_values[t, c] = charging_model.unused_capacity[t, c].value

    plt.figure(figsize=(14, 8))

    for c in [30, 24]:  #Cells 30 and 24
        plt.plot(timesteps, unused_capacity_values[:, c], label=f'Cell {c}')

    plt.xlabel('Timestep')
    plt.ylabel('Unused Capacity')
    plt.title('Unused Capacity Over Time for Cells between 20.000 and 35.000 Watt')
    plt.legend(loc='upper right')
    plt.grid(True)

    # Verzeichnis für die Speicherung erstellen, falls es nicht existiert
    output_dir = os.path.join('results', 'pictures')
    os.makedirs(output_dir, exist_ok=True)

    # Speichern des Diagramms
    output_path = os.path.join(output_dir, f'unused_capacity_between_20000_and_35000W_time{time_of_optimization}.png')
    plt.savefig(output_path)
    #plt.show()

def unused_capacity_15000_to_20000(charging_model, time_of_optimization):
    # Stellen sicher, dass nb_timestep und nb_cell als Listen oder Bereiche vorliegen
    timesteps = list(charging_model.nb_timestep)
    cells = list(charging_model.nb_cell)

    # Daten für die Visualisierung berechnen
    nb_timestep = len(timesteps)
    nb_cell = len(cells)
    unused_capacity_values = np.zeros((nb_timestep, nb_cell))

    for t in timesteps:
        for c in cells:
            if (t, c) in charging_model.t_cs:
                unused_capacity_values[t, c] = charging_model.unused_capacity[t, c].value

    plt.figure(figsize=(14, 8))

    for c in [27, 13, 61]:  #Cells 27, 13 and 61
        plt.plot(timesteps, unused_capacity_values[:, c], label=f'Cell {c}')

    plt.xlabel('Timestep')
    plt.ylabel('Unused Capacity')
    plt.title('Unused Capacity Over Time for Cells between 15.000 and 20.000 Watt')
    plt.legend(loc='upper right')
    plt.grid(True)

    # Verzeichnis für die Speicherung erstellen, falls es nicht existiert
    output_dir = os.path.join('results', 'pictures')
    os.makedirs(output_dir, exist_ok=True)

    # Speichern des Diagramms
    output_path = os.path.join(output_dir, f'unused_capacity_between_15000_and_20000W_time{time_of_optimization}.png')
    plt.savefig(output_path)
    #plt.show()

def unused_capacity_10000_to_15000(charging_model, time_of_optimization):
    # Stellen sicher, dass nb_timestep und nb_cell als Listen oder Bereiche vorliegen
    timesteps = list(charging_model.nb_timestep)
    cells = list(charging_model.nb_cell)

    # Daten für die Visualisierung berechnen
    nb_timestep = len(timesteps)
    nb_cell = len(cells)
    unused_capacity_values = np.zeros((nb_timestep, nb_cell))

    for t in timesteps:
        for c in cells:
            if (t, c) in charging_model.t_cs:
                unused_capacity_values[t, c] = charging_model.unused_capacity[t, c].value

    plt.figure(figsize=(14, 8))

    for c in [85, 74, 47, 42, 23]:  #Cells 85, 74, 47, 42 and 23
        plt.plot(timesteps, unused_capacity_values[:, c], label=f'Cell {c}')

    plt.xlabel('Timestep')
    plt.ylabel('Unused Capacity')
    plt.title('Unused Capacity Over Time for Cells between 10.000 and 15.000 Watt')
    plt.legend(loc='upper right')
    plt.grid(True)

    # Verzeichnis für die Speicherung erstellen, falls es nicht existiert
    output_dir = os.path.join('results', 'pictures')
    os.makedirs(output_dir, exist_ok=True)

    # Speichern des Diagramms
    output_path = os.path.join(output_dir, f'unused_capacity_between_10000_and_15000W_time{time_of_optimization}.png')
    plt.savefig(output_path)
    #plt.show()

def unused_capacity_7000_to_10000(charging_model, time_of_optimization):
    # Stellen sicher, dass nb_timestep und nb_cell als Listen oder Bereiche vorliegen
    timesteps = list(charging_model.nb_timestep)
    cells = list(charging_model.nb_cell)

    # Daten für die Visualisierung berechnen
    nb_timestep = len(timesteps)
    nb_cell = len(cells)
    unused_capacity_values = np.zeros((nb_timestep, nb_cell))

    for t in timesteps:
        for c in cells:
            if (t, c) in charging_model.t_cs:
                unused_capacity_values[t, c] = charging_model.unused_capacity[t, c].value

    plt.figure(figsize=(14, 8))

    for c in [54, 19, 18, 10, 83]:  #Cells 54, 19, 18, 10, 83
        plt.plot(timesteps, unused_capacity_values[:, c], label=f'Cell {c}')

    plt.xlabel('Timestep')
    plt.ylabel('Unused Capacity')
    plt.title('Unused Capacity Over Time for Cells between 7.000 and 10.000 Watt')
    plt.legend(loc='upper right')
    plt.grid(True)

    # Verzeichnis für die Speicherung erstellen, falls es nicht existiert
    output_dir = os.path.join('results', 'pictures')
    os.makedirs(output_dir, exist_ok=True)

    # Speichern des Diagramms
    output_path = os.path.join(output_dir, f'unused_capacity_between_7000_and_10000W_time{time_of_optimization}.png')
    plt.savefig(output_path)
    #plt.show()

def unused_capacity_4000_to_7000(charging_model, time_of_optimization):
    # Stellen sicher, dass nb_timestep und nb_cell als Listen oder Bereiche vorliegen
    timesteps = list(charging_model.nb_timestep)
    cells = list(charging_model.nb_cell)

    # Daten für die Visualisierung berechnen
    nb_timestep = len(timesteps)
    nb_cell = len(cells)
    unused_capacity_values = np.zeros((nb_timestep, nb_cell))

    for t in timesteps:
        for c in cells:
            if (t, c) in charging_model.t_cs:
                unused_capacity_values[t, c] = charging_model.unused_capacity[t, c].value

    plt.figure(figsize=(14, 8))

    for c in [3, 55, 16, 48]:  #Cells 3, 55, 16, 48
        plt.plot(timesteps, unused_capacity_values[:, c], label=f'Cell {c}')

    plt.xlabel('Timestep')
    plt.ylabel('Unused Capacity')
    plt.title('Unused Capacity Over Time for Cells between 4.000 and 7.000 Watt')
    plt.legend(loc='upper right')
    plt.grid(True)

    # Verzeichnis für die Speicherung erstellen, falls es nicht existiert
    output_dir = os.path.join('results', 'pictures')
    os.makedirs(output_dir, exist_ok=True)

    # Speichern des Diagramms
    output_path = os.path.join(output_dir, f'unused_capacity_between_4000_and_7000W_time{time_of_optimization}.png')
    plt.savefig(output_path)
    #plt.show()

def unused_capacity_2000_to_4000(charging_model, time_of_optimization):
    # Stellen sicher, dass nb_timestep und nb_cell als Listen oder Bereiche vorliegen
    timesteps = list(charging_model.nb_timestep)
    cells = list(charging_model.nb_cell)

    # Daten für die Visualisierung berechnen
    nb_timestep = len(timesteps)
    nb_cell = len(cells)
    unused_capacity_values = np.zeros((nb_timestep, nb_cell))

    for t in timesteps:
        for c in cells:
            if (t, c) in charging_model.t_cs:
                unused_capacity_values[t, c] = charging_model.unused_capacity[t, c].value

    plt.figure(figsize=(14, 8))

    for c in [89, 49, 63, 15, 38, 58, 2, 5, 70, 91]:  #Cells 89, 49, 63, 15, 38, 58 2, 5, 70, 91
        plt.plot(timesteps, unused_capacity_values[:, c], label=f'Cell {c}')

    plt.xlabel('Timestep')
    plt.ylabel('Unused Capacity')
    plt.title('Unused Capacity Over Time for Cells between 2.000 and 4.000 Watt')
    plt.legend(loc='upper right')
    plt.grid(True)

    # Verzeichnis für die Speicherung erstellen, falls es nicht existiert
    output_dir = os.path.join('results', 'pictures')
    os.makedirs(output_dir, exist_ok=True)

    # Speichern des Diagramms
    output_path = os.path.join(output_dir, f'unused_capacity_between_2000_and_4000W_time{time_of_optimization}.png')
    plt.savefig(output_path)
    #plt.show()

def unused_capacity_1000_to_2000(charging_model, time_of_optimization):
    # Stellen sicher, dass nb_timestep und nb_cell als Listen oder Bereiche vorliegen
    timesteps = list(charging_model.nb_timestep)
    cells = list(charging_model.nb_cell)

    # Daten für die Visualisierung berechnen
    nb_timestep = len(timesteps)
    nb_cell = len(cells)
    unused_capacity_values = np.zeros((nb_timestep, nb_cell))

    for t in timesteps:
        for c in cells:
            if (t, c) in charging_model.t_cs:
                unused_capacity_values[t, c] = charging_model.unused_capacity[t, c].value

    plt.figure(figsize=(14, 8))

    for c in [7, 68, 72, 4, 21, 79, 53]:  #Cells 7, 68, 72, 4, 21, 79, 53
        plt.plot(timesteps, unused_capacity_values[:, c], label=f'Cell {c}')

    plt.xlabel('Timestep')
    plt.ylabel('Unused Capacity')
    plt.title('Unused Capacity Over Time for Cells between 1.000 and 2.000 Watt')
    plt.legend(loc='upper right')
    plt.grid(True)

    # Verzeichnis für die Speicherung erstellen, falls es nicht existiert
    output_dir = os.path.join('results', 'pictures')
    os.makedirs(output_dir, exist_ok=True)

    # Speichern des Diagramms
    output_path = os.path.join(output_dir, f'unused_capacity_between_1000_and_2000W_time{time_of_optimization}.png')
    plt.savefig(output_path)
    #plt.show()

def unused_capacity_up_to_1000(charging_model, time_of_optimization):
    # Stellen sicher, dass nb_timestep und nb_cell als Listen oder Bereiche vorliegen
    timesteps = list(charging_model.nb_timestep)
    cells = list(charging_model.nb_cell)

    # Daten für die Visualisierung berechnen
    nb_timestep = len(timesteps)
    nb_cell = len(cells)
    unused_capacity_values = np.zeros((nb_timestep, nb_cell))

    for t in timesteps:
        for c in cells:
            if (t, c) in charging_model.t_cs:
                unused_capacity_values[t, c] = charging_model.unused_capacity[t, c].value

    plt.figure(figsize=(14, 8))

    for c in [46, 25, 12, 37]:  #Cells 46, 25, 12, 37
        plt.plot(timesteps, unused_capacity_values[:, c], label=f'Cell {c}')

    plt.xlabel('Timestep')
    plt.ylabel('Unused Capacity')
    plt.title('Unused Capacity Over Time for Cells up to 1.000 Watt')
    plt.legend(loc='upper right')
    plt.grid(True)

    # Verzeichnis für die Speicherung erstellen, falls es nicht existiert
    output_dir = os.path.join('results', 'pictures')
    os.makedirs(output_dir, exist_ok=True)

    # Speichern des Diagramms
    output_path = os.path.join(output_dir, f'unused_capacity_up_to_1000W_time{time_of_optimization}.png')
    plt.savefig(output_path)
    #plt.show()

def unused_capacity_A2(charging_model, time_of_optimization):
    # Stellen sicher, dass nb_timestep und nb_cell als Listen oder Bereiche vorliegen
    timesteps = list(charging_model.nb_timestep)
    cells = list(charging_model.nb_cell)

    # Daten für die Visualisierung berechnen
    nb_timestep = len(timesteps)
    nb_cell = len(cells)
    unused_capacity_values = np.zeros((nb_timestep, nb_cell))

    for t in timesteps:
        for c in cells:
            if (t, c) in charging_model.t_cs:
                unused_capacity_values[t, c] = charging_model.unused_capacity[t, c].value

    plt.figure(figsize=(14, 8))

    for c in [1, 2, 3, 6, 10, 12, 13]:  #Cells for A2
        plt.plot(timesteps, unused_capacity_values[:, c], label=f'Cell {c}')

    plt.xlabel('Timestep')
    plt.ylabel('Unused Capacity')
    plt.title('Unused Capacity Over Time for Cells of the A2')
    plt.legend(loc='upper right')
    plt.grid(True)

    # Verzeichnis für die Speicherung erstellen, falls es nicht existiert
    output_dir = os.path.join('results', 'pictures')
    os.makedirs(output_dir, exist_ok=True)

    # Speichern des Diagramms
    output_path = os.path.join(output_dir, f'unused_capacity_A2_{time_of_optimization}.png')
    plt.savefig(output_path)
    #plt.show()

def plot_total_energy_charged(timesteps, total_energy_charged, time_of_optimization):
    """
    Erstellt ein Diagramm für die gesamte geladene Energie über die Zeit.

    :param timesteps: Liste der Zeitschritte
    :param total_energy_charged: Liste der gesamten geladenen Energie für jeden Zeitschritt
    :param time_of_optimization: Zeitstempel der Optimierung, um das Diagramm eindeutig zu benennen
    """
    plt.figure(figsize=(14, 8))
    plt.plot(timesteps, total_energy_charged, label='Total Energy Charged', marker='o')
    plt.xlabel('Timestep')
    plt.ylabel('Total Energy Charged')
    plt.title('Total Energy Charged Over Time (all cells)')
    plt.legend(loc='upper right')
    plt.grid(True)

    # Verzeichnis für die Speicherung erstellen, falls es nicht existiert
    output_dir = os.path.join('results', 'pictures')
    os.makedirs(output_dir, exist_ok=True)

    # Speichern des Diagramms
    output_path = os.path.join(output_dir, f'total_energy_charged_over_time_{time_of_optimization}.png')
    plt.savefig(output_path)

    # Anzeigen des Diagramms
    #plt.show()

def plot_energy_charged_for_selected_cells(timesteps, results, inds_of_all_cells, selected_cells, time_of_optimization):
    """
    Erstellt ein Diagramm für die geladene Energie über die Zeit für ausgewählte Zellen.

    :param timesteps: Liste der Zeitschritte
    :param results: Dictionary mit geladenen Energiewerten
    :param inds_of_all_cells: Liste der IDs aller Zellen
    :param selected_cells: Liste der ausgewählten Zellen
    :param time_of_optimization: Zeitstempel der Optimierung, um das Diagramm eindeutig zu benennen
    """
    plt.figure(figsize=(14, 8))
    for c in selected_cells:
        energy_charged = []
        for t in timesteps:
            if c in inds_of_all_cells:
                energy_t = (
                    np.sum(results["E_charge1"][t, c, :])
                    + np.sum(results["E_charge2"][t, c, :])
                    + np.sum(results["E_charge3"][t, c, :])
                )
                energy_charged.append(energy_t)
            else:
                energy_charged.append(0)

        plt.plot(timesteps, energy_charged, label=f'Cell {c}', marker='o')

    plt.xlabel('Timestep')
    plt.ylabel('Energy Charged')
    plt.title('Energy Charged Over Time for Selected Cells')
    plt.legend(loc='upper right')
    plt.grid(True)

    # Verzeichnis für die Speicherung erstellen, falls es nicht existiert
    output_dir = os.path.join('results', 'pictures')
    os.makedirs(output_dir, exist_ok=True)

    # Speichern des Diagramms für ausgewählte Zellen
    output_path_selected = os.path.join(output_dir, f'energy_charged_selected_cells_{time_of_optimization}.png')
    plt.savefig(output_path_selected)

    # Anzeigen des Diagramms für ausgewählte Zellen
    #plt.show()

def plot_variable_over_time(model, variable_name, results, time_of_optimization, variable_label):
    """
    Erstellt ein Diagramm für eine gegebene Variable über die Zeit.

    :param model: Das Pyomo-Modell, das die Zeitstufen enthält.
    :param variable_name: Der Name der zu plottenden Variablen.
    :param results: Dictionary mit den Ergebnissen der Modellvariablen.
    :param time_of_optimization: Zeitstempel der Optimierung, um das Diagramm eindeutig zu benennen.
    :param variable_label: Beschriftung für die y-Achse des Diagramms.
    """
    timesteps = model.nb_timestep
    total_variable_values = []

    for t in timesteps:
        total_value_t = 0
        for c in model.nb_cell:
            total_value_t += np.sum(results[variable_name][t, c, :])
        total_variable_values.append(total_value_t)

    # Erstellen und Speichern des Diagramms
    plt.figure(figsize=(14, 8))
    plt.plot(timesteps, total_variable_values, label=f'Total {variable_label}', marker='o')
    plt.xlabel('Timestep')
    plt.ylabel(variable_label)
    plt.title(f'Total {variable_label} Over Time (all cells)')
    plt.legend(loc='upper right')
    plt.grid(True)

    # Verzeichnis für die Speicherung erstellen, falls es nicht existiert
    output_dir = os.path.join('results', 'Pictures')
    os.makedirs(output_dir, exist_ok=True)

    # Speichern des Diagramms
    output_path = os.path.join(output_dir, f'{variable_name}_over_time_{time_of_optimization}.png')
    plt.savefig(output_path)

    # Anzeigen des Diagramms
    #plt.show()

def plot_combined_variable_over_time(model, results, time_of_optimization):
    """
    Erstellt ein Diagramm für die kombinierte Variable (n_wait_charge_next + n_wait) über die Zeit.

    :param model: Das Pyomo-Modell, das die Zeitstufen enthält.
    :param results: Dictionary mit den Ergebnissen der Modellvariablen.
    :param time_of_optimization: Zeitstempel der Optimierung, um das Diagramm eindeutig zu benennen.
    """
    timesteps = model.nb_timestep
    total_combined_values = []

    for t in timesteps:
        total_value_t = 0
        for c in model.nb_cell:
            total_value_t += np.sum(results["n_wait_charge_next"][t, c, :]) + np.sum(results["n_wait"][t, c, :])
        total_combined_values.append(total_value_t)

    # Erstellen und Speichern des Diagramms
    plt.figure(figsize=(14, 8))
    plt.plot(timesteps, total_combined_values, label='Total (n_wait_charge_next + n_wait)', marker='o')
    plt.xlabel('Timestep')
    plt.ylabel('Combined Wait and Charge')
    plt.title('Total (n_wait_charge_next + n_wait) Over Time (all cells)')
    plt.legend(loc='upper right')
    plt.grid(True)

    # Verzeichnis für die Speicherung erstellen, falls es nicht existiert
    output_dir = os.path.join('results', 'Pictures')
    os.makedirs(output_dir, exist_ok=True)

    # Speichern des Diagramms
    output_path = os.path.join(output_dir, f'combined_wait_charge_over_time_{time_of_optimization}.png')
    plt.savefig(output_path)

    # Anzeigen des Diagramms
    #plt.show()


def plot_unused_capacity_new(model, time_of_optimization):
    """
    Plots the Unused_capacity_new variable for all cells with charging stations,
    marks the cell capacity as horizontal red lines, and fills the space between
    unused capacity and cell capacity with green.

    Parameters:
    - model: The Pyomo model containing the optimization results.
    - time_of_optimization: Timestamp or identifier to include in the file name.
    """
    # Create a list to store the cell indices, corresponding unused capacities, and cell capacities
    cell_indices = []
    unused_capacities = []
    cell_capacities = []

    # Iterate through cells with charging stations
    for c in model.cs_cells:
        if c in model.Unused_capacity_new:
            unused_capacity_value = model.Unused_capacity_new[c].value
            cell_capacity = model.cell_charging_cap[c]  # Retrieve the capacity for each cell

            if unused_capacity_value is not None:
                cell_indices.append(c)
                unused_capacities.append(unused_capacity_value)
                cell_capacities.append(cell_capacity)
            else:
                print(f"Warning: Unused capacity for cell {c} is not set.")
        else:
            print(f"Warning: No Unused_capacity_new value found for cell {c}")

    # Plotting the unused capacities as bars
    plt.figure(figsize=(10, 6))
    plt.bar(cell_indices, unused_capacities, color='blue', alpha=0.7, label='Unused Capacity')

    # Filling the space between unused capacity and cell capacity
    for i, c in enumerate(cell_indices):
        plt.fill_between([c - 0.4, c + 0.4], unused_capacities[i], cell_capacities[i], color='green', alpha=0.7,
                         label='Used Capacity' if i == 0 else "")

    # Adding horizontal lines for cell capacities
    plt.hlines(cell_capacities, [i - 0.4 for i in cell_indices], [i + 0.4 for i in cell_indices], colors='red',
               label='Cell Capacity', linewidth=2)

    plt.xlabel('Cell Index')
    plt.ylabel('Capacity (kW)')
    plt.title('Unused Capacity, Used Capacity und Cell Capacity for Cells with Charging Station')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()

    # Verzeichnis für die Speicherung erstellen, falls es nicht existiert
    output_dir = os.path.join('results', 'Pictures')
    os.makedirs(output_dir, exist_ok=True)

    # Speichern des Diagramms
    output_path = os.path.join(output_dir, f'unused_capacity_new{time_of_optimization}.png')
    plt.savefig(output_path)

    # Optional: Anzeigen des Diagramms
    # plt.show()
