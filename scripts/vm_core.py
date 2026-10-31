import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import vertex_lite as model
import vertex_lite.initialisation as init
from vertex_lite import custom
from vertex_lite.forces import TargetArea, Tension, Perimeter, Pressure
from vertex_lite.run_select import run, basic_simulation

DT, K, GAMMA, L_WT, L_MUT, A0 = 0.001, 1.0, 0.04, 0.075, 0.68, 1.0
N_ACROSS = N_UP = 20
SKIP = 20
FORCE = TargetArea() + Tension() + Perimeter() + Pressure()


def relax(D, sim_id, T1_eps):
    rand = np.random.RandomState(sim_id)
    mesh = init.toroidal_hex_mesh(N_UP, N_ACROSS, noise=D, rand=rand)
    cells = model.Cells(mesh, properties={'K': K, 'Gamma': GAMMA, 'P': 0.0, 'boundary_P': 0.0,
                                          'Lambda': L_WT, 'Lambda_boundary': 0.5, 'A0': A0})
    hist = run(basic_simulation(cells, FORCE, T1_eps=T1_eps), int(50 / DT), int(1 / DT))
    return hist[-1].copy(), rand.get_state()


def choose_mutants(cells, rand, mode, f, seed_cell=64):
    n = int(f * len(cells))
    if mode == 'random':
        return rand.choice(len(cells), size=n, replace=False)
    return custom.cluster_mutants(cells, f, rand, seed=seed_cell)


def challenge(relaxed, rand_state, mode, f, T1_eps, seed_cell=64, n_steps=15000, ids=None):
    rand = np.random.RandomState(); rand.set_state(rand_state)
    cells = relaxed.copy()
    if ids is None:
        ids = choose_mutants(cells, rand, mode, f, seed_cell)
    grp = np.zeros(len(cells), dtype=int); grp[ids] = 1
    cells.properties['parent_group'] = grp
    cells.properties['Gamma'] = np.array([GAMMA, GAMMA])[grp]
    cells.properties['Lambda'] = np.array([L_WT, L_MUT])[grp]
    hist = run(basic_simulation(cells, FORCE, T1_eps=T1_eps), n_steps, SKIP)
    return hist, np.asarray(ids)


def neighbour_mask(mesh, source_mask):
    fid, rev = mesh.face_id_by_edge, mesh.edges.reverse
    nb = np.zeros(mesh.n_face, bool)
    nb[fid[rev[source_mask[fid]]]] = True
    return nb & ~source_mask


def measure(hist, ids):
    N = len(hist[0]); is_mut = np.zeros(N, bool); is_mut[ids] = True
    nb0 = neighbour_mask(hist[0].mesh, is_mut)
    nb_dyn = nb0.copy()
    keys = ['phi', 'p_glob', 'p_loc', 'dhex_glob', 'dhex_loc', 'n_wt_alive',
            'frac5_loc', 'frac6_loc', 'frac7_loc', 'frac5_glob', 'frac6_glob', 'frac7_glob']
    out = {k: [] for k in keys}
    for c in hist:
        m = c.mesh
        area, per = m.area, m.perimeter
        alive = area > 0
        mut_alive = is_mut & alive
        if mut_alive.any():
            nb_dyn = neighbour_mask(m, mut_alive)
        p = np.full(N, np.nan); ok = alive & (per > 0)
        p[ok] = per[ok] / np.sqrt(area[ok])
        n = np.bincount(m.face_id_by_edge, minlength=N)
        d = np.abs(n - 6).astype(float)
        loc_p = nb_dyn & alive; loc_t = nb0 & alive
        out['phi'].append(mut_alive.sum() / len(ids))
        out['p_glob'].append(np.nanmean(p[alive]))
        out['p_loc'].append(np.nanmean(p[loc_p]) if loc_p.any() else np.nan)
        out['dhex_glob'].append(d[alive].mean())
        out['dhex_loc'].append(d[loc_t].mean() if loc_t.any() else np.nan)
        out['n_wt_alive'].append((alive & ~is_mut).sum())
        for k in (5, 6, 7):
            out[f'frac{k}_loc'].append((n[loc_t] == k).mean() if loc_t.any() else np.nan)
            out[f'frac{k}_glob'].append((n[alive] == k).mean())
    res = {k: np.array(v) for k, v in out.items()}
    res['t_steps'] = np.arange(len(hist)) * SKIP
    res['mutant_ids'] = np.asarray(ids)
    return res


def t_x(phi, x, t):
    idx = np.where(phi < x)[0]
    return t[idx[0]] if len(idx) else np.nan
