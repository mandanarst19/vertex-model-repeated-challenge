import itertools
import numpy as np
import vertex_lite as model
import vertex_lite.initialisation as init
from vertex_lite.forces import TargetArea, Tension, Perimeter, Pressure
# run simulation
def run(simulation,N_step,skip) -> list[model.Cells]:
    return [cells.copy() for cells in itertools.islice(simulation,0,int(N_step),int(skip))]
#simulation without division
def basic_simulation(cells: model.Cells, force, dt=0.001, T1_eps=0.04, viscosity=0.02):
    """
    Run a basic simulation without cell division.

    Parameters:
    cells (Cells): The cells to simulate.
    force (function): The force function to apply to the cells.
    dt (float): The time step for the simulation.
    T1_eps (float): The threshold for T1 transitions.

    Yields:
    Cells: The updated cells after each time step.
    """
    while True:
        cells.mesh , number_T1,edg_rem = cells.mesh.transition(T1_eps)
        F = force(cells)/viscosity
        expansion = 0.05*np.average(F*cells.mesh.vertices,1)*dt
        dv = dt*model.sum_vertices(cells.mesh.edges,F) 
        cells.mesh = cells.mesh.moved(dv).scaled(1.0+ expansion)
        yield cells