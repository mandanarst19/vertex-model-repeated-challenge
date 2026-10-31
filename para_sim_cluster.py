import numpy as np
import os
import time 
import sys
from multiprocessing import Pool
import multiprocessing as mp

start_time = time.time()
script_dir = os.path.dirname(os.path.abspath(__file__))
from vertex_lite.cells import Cells
from vertex_lite.mesh import Mesh
output_dir = os.path.join(script_dir, "results/resultados_cluster")
os.makedirs(output_dir, exist_ok=True)
print("Guardando archivo en:", output_dir)
# input("Presiona Enter para continuar...")
start_time = time.time()
import vertex_lite as model
from vertex_lite.custom import shape_index
from vertex_lite.forces import TargetArea, Tension, Perimeter, Pressure
import vertex_lite.initialisation as init
from vertex_lite.run_select import run, basic_simulation
from vertex_lite import custom , coloring
import math
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import gc
import dill
dt= 0.001
viscosity = 0.02
P=0.0
K=1.0 #area elasticity
G= 0.04 #contractility of the cell
L=0.075 #line tensions
Lambda_0 = 0.68 #lambda for the mutants
t_end = 15
rand=np.random.RandomState(123) #random seed for reproducibility
N_cell_across= 20
N_cell_up= 20
N_total= N_cell_across * N_cell_up

params=[K,G,L]
N_Step = int(t_end / dt)
skip = 20

mutant_percentages = [0.025, 0.075,0.125, 0.175, 0.225, 0.275, 0.3, 0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
# mutant_percentages = [0.25]

noise_percentages = [0.0,0.05, 0.1,0.15, 0.2, 0.25,0.3, 0.35, 0.4, 0.45, 0.5,0.55, 0.6, 0.65, 0.7]
# noise_percentages =[0.25]
N_simulations = 10
results = {}


def main_sim(args,save_dill=False,compress=True):
    (mesh_type, mutant_percentage, sim_id,N_cell_across, N_cell_up, dt, K, G, L, Lambda_0,
     P, t_end, N_Step, skip) = args
    
    rand = np.random.RandomState(sim_id)
    mesh = init.toroidal_hex_mesh(N_cell_up, N_cell_across, noise=mesh_type, rand=rand)
    
    N_mutants = int(mutant_percentage * N_total)
    cells = model.Cells(mesh, properties={
        'K': K,
        'Gamma': G,
        'P': 0.0,
        'boundary_P': P,
        'Lambda': L,
        'Lambda_boundary': 0.5,
        'A0': 1.0
    })

    # Fase de relajación inicial
    step_init = int(50 / dt)
    history_init = run(basic_simulation(cells, TargetArea() + Tension() + Perimeter() + Pressure()), step_init, int(1 / dt))
    cells = history_init[-1].copy()
    del history_init
    
    # Mutación
    ids_mutant = custom.cluster_mutants(cells, mutant_percentage, rand, seed=64)
    cells.properties['parent_group'] = np.zeros(len(cells), dtype=int)
    cells.properties['parent_group'][ids_mutant] = 1
    cells.properties['Gamma'] = np.array([G, G])[cells.properties['parent_group']]
    cells.properties['Lambda'] = np.array([L, Lambda_0])[cells.properties['parent_group']]

    # Simulación principal

    history = run(basic_simulation(cells, TargetArea() + Tension() + Perimeter() + Pressure()), N_Step, skip)
    print(f"  Simulación {mesh_type}/{mutant_percentage}/{sim_id} completada.",flush=True)
    df_cells,df_global,df_mutant,df_extrusion,df_interface = analyze_history(history, mesh_type, mutant_percentage, sim_id)
    sim_dir = os.path.join(output_dir, f"noise_{mesh_type}", f"pct_{mutant_percentage}",f"sim_{sim_id}")
    os.makedirs(sim_dir, exist_ok=True)
    df_cells.to_parquet(os.path.join(sim_dir, "cells.parquet"))
    df_global.to_parquet(os.path.join(sim_dir, "global.parquet"))
    df_mutant.to_parquet(os.path.join(sim_dir, "mutant.parquet"))
    df_extrusion.to_parquet(os.path.join(sim_dir, "extrusion.parquet"))
    df_interface.to_parquet(os.path.join(sim_dir, "interface.parquet"))

    history_path = os.path.join(sim_dir, "history.dill")
    
    if save_dill:
        if compress:
            import gzip
            with gzip.open(history_path + ".gz", "wb") as f:
                dill.dump(history, f)
        else:
            with open(history_path, "wb") as f:
                dill.dump(history, f)
    del history
    gc.collect()
    print(f"  Simulación {mesh_type}/{mutant_percentage}/{sim_id} saved.",flush=True)
    return (mesh_type, mutant_percentage, sim_id)

def analyze_history(history: list[Cells], mesh_type, mutant_percentage, sim_id):
    data_global = []
    data_mutant = []
    data_extrusion = []
    data_cells = []
    data_interface = []
    energy_terms = TargetArea() + Tension() + Perimeter() + Pressure()

    # --- Identify mutants ---
    cells0 = history[0]
    mutant_cells = np.where(cells0.properties["parent_group"] == 1)[0]
    alive_mutants = set(mutant_cells)
    
    extrusion_dict = {}

    mutant_neigh = {m: None for m in mutant_cells}

    cluster_neigh = None
    for t_idx, cells in enumerate(history):
# --- actualizar vivos ---
        for m in list(alive_mutants):
            if cells.mesh.area[m] <= 0:
                if m not in extrusion_dict:
                    extrusion_dict[m] = t_idx
                alive_mutants.remove(m)

        # --- recalcular cluster si sigue vivo ---
        if alive_mutants:
            cluster_neigh = custom.selected_neighbours(
                cells, list(alive_mutants)
            )


        # --- construir neigh_set usando último valor válido ---
        clust = cluster_neigh if cluster_neigh is not None else []
        if clust:
            for m in alive_mutants:
                mutant_neigh[m] = clust
        n_lados_all = np.array([len(cells.mesh.boundary(i)) for i in range(len(cells))])
        unique, counts = np.unique(n_lados_all, return_counts=True)
        shape_index_all=np.full(len(cells), np.nan)
        for i in range(len(cells)):
            s=shape_index(cells, face_id=i, normalize=False)
            if s is not None and s>0:
                shape_index_all[i]=s

            t_eff = custom.calculate_t_eff(cells, cell_id=i)
            # bound = cells.mesh.boundary(i)
            # center = cells.mesh.vertices[:, bound].mean(axis=1) if cells.mesh.area[i] > 0 and not np.array_equal(bound, [-1]) and len(bound) >=3 else (np.nan, np.nan)
            center =  custom.calculate_centroid(cells, cell_id=i)
            # center_mean =  custom.calculate_center_mean(cells, cell_id=i)
            data_cells.append({
                
                'time': t_idx*skip,
                'cell_id': i,
                'mutant': int(i in mutant_cells),
                'neighbor_of_mutant': int(i in clust),
                'n_lados': n_lados_all[i],
                'shape_index': shape_index_all[i],
                'alive': int(cells.mesh.area[i]>0),
                'area': cells.mesh.area[i],
                'perimeter': cells.mesh.perimeter[i],
                'centroid_x': center[0],
                'centroid_y': center[1],
                # 'center_mean_x': center_mean[0],
                # 'center_mean_y': center_mean[1],
                't_eff': t_eff
                
                
            })
        shape_global = np.nanmean(shape_index_all)

        interface_perimeter = custom.calculate_interface_perimeter(cells)
        interface_by_mutant = custom.calculate_interface_perimeter_by_mutant(cells)
        energy_target_area = TargetArea().energy(cells)
        energy_tension = Tension().energy(cells)
        energy_perimeter = Perimeter().energy(cells)
        energy_pressure = Pressure().energy(cells)
        energy_total = energy_terms.energy(cells)

        for n_lados, count in zip(unique, counts):
            data_global.append({
                
                
                'time': t_idx*skip,
                'shape_global': shape_global,
                'n_lados': n_lados,
                'n_cells': count,
                'total_cells': len(cells),
                'frac_cells': count / len(cells)
                
            })
        for m in mutant_cells:
            
            if mutant_neigh[m] is None:
                continue

            neigh_m = mutant_neigh[m]
            shape_index_neigh = np.nanmean(shape_index_all[neigh_m])
            n_lados_neigh = np.mean(n_lados_all[neigh_m])

            # Añadir entrada a la lista
            data_mutant.append({
                
                
                'time': t_idx*skip,
                'mutant_id': m,
                'shape_global': shape_global,
                'shape_neigh': shape_index_neigh,
                'n_lados_neigh': n_lados_neigh,
                'interface_perimeter': interface_perimeter,
                'T_eff_global': np.nanmean([custom.calculate_t_eff(cells, cell_id=i) for i in range(len(cells))]),
                'T_eff_neigh': np.nanmean([custom.calculate_t_eff(cells, cell_id=i) for i in neigh_m]),
                'interface_by_mutant': interface_by_mutant.get(m, 0.0)
            })
        # Do not append extrusion rows every time-step (would duplicate entries).
        # We'll build the final list after scanning the whole history.
        data_interface.append({
            'time': t_idx*skip, 
            'interface_perimeter': interface_perimeter,
            'energy': energy_total,
            'energy_target_area': energy_target_area,
            'energy_tension': energy_tension,
            'energy_perimeter': energy_perimeter,
            'energy_pressure': energy_pressure,
            'n_alive_mutants': len(alive_mutants),
            'n_mutant_neighbors': len(clust),
            'n_interface_edges': len(custom.get_interface_edges(cells)[0]),
        })
        # (no per-step appends here) -> build final extrusion list after scanning history

    # Ensure one row per initial mutant: record first extrusion time or NaN if never extruded
    for m in mutant_cells:
        t_ext = extrusion_dict.get(m, np.nan)
        data_extrusion.append({
            'mutant_id': m,
            'extrusion_time': (t_ext * skip) if not np.isnan(t_ext) else np.nan
        })

    
        

        

    return(
        pd.DataFrame(data_cells),
        pd.DataFrame(data_global),
        pd.DataFrame(data_mutant),
        pd.DataFrame(data_extrusion),
        pd.DataFrame(data_interface),
    )


if __name__ == "__main__":
    start_time = time.time()

    tasks = [
        (
            mesh_type, pct, sim_id,
            N_cell_across, N_cell_up, dt,
            K, G, L, Lambda_0, P, t_end,
            N_Step, skip # save_dill=False, compress=True
        )
        for mesh_type in noise_percentages
        for pct in mutant_percentages
        for sim_id in range(N_simulations)
    ]

    cpu_count = mp.cpu_count()
    chunksize = max(1, len(tasks) // (cpu_count * 4))

    print(f"Starting {len(tasks)} tasks across {cpu_count} CPUs...")
    with Pool(processes=cpu_count) as pool:
        for _ in pool.imap_unordered(main_sim, tasks, chunksize=chunksize):
            pass

    elapsed_time = time.time() - start_time
    mins, secs = divmod(elapsed_time, 60)
    print(f"Tiempo total de computación: {int(mins)} min {secs:.1f} sec")
    print("All done!")