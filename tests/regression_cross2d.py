import sys, numpy as np
repo, out = sys.argv[1], sys.argv[2]
sys.path.insert(0, repo)
import vertex_lite as model
import vertex_lite.initialisation as init
from vertex_lite import custom
from vertex_lite.forces import TargetArea, Tension, Perimeter, Pressure
from vertex_lite.run_select import run, basic_simulation

D, pct, sim_id = 0.25, 0.25, 0
rand = np.random.RandomState(sim_id)
mesh = init.toroidal_hex_mesh(20, 20, noise=D, rand=rand)
print("has_boundary:", mesh.has_boundary())
cells = model.Cells(mesh, properties={'K':1.0,'Gamma':0.04,'P':0.0,'boundary_P':0.0,
                                      'Lambda':0.075,'Lambda_boundary':0.5,'A0':1.0})
F = TargetArea()+Tension()+Perimeter()+Pressure()
hist0 = run(basic_simulation(cells, F), 2000, 1000)
cells = hist0[-1].copy()
ids = custom.cluster_mutants(cells, pct, rand, seed=64)
grp = np.zeros(len(cells), dtype=int); grp[ids] = 1
cells.properties['parent_group'] = grp
cells.properties['Gamma'] = np.array([0.04,0.04])[grp]
cells.properties['Lambda'] = np.array([0.075,0.68])[grp]
hist = run(basic_simulation(cells, F), 3000, 20)
last = hist[-1]
np.savez(out, vertices=last.mesh.vertices, area=last.mesh.area,
         perim=last.mesh.perimeter, mutants=np.sort(ids),
         n_alive=np.array([(h.mesh.area>0).sum() for h in hist]))
print("n_face:", len(last), " alive at end:", (last.mesh.area>0).sum(), " mutants:", len(ids))
