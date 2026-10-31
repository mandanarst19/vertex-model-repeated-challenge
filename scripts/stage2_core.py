import os, json, argparse
from collections import defaultdict
import numpy as np
from multiprocessing import Pool
from vm_core import *

EPS, D0, F1, N2, T_PRIME, T2 = 0.04, 0.25, 0.25, 20, 15000, 6000
R_CLONE_DISK = np.sqrt(N2 * A0 / np.pi) + np.sqrt(4 * A0 / np.pi)
R_CLONE1_DISK = np.sqrt(F1 * 400 * A0 / np.pi) + np.sqrt(4 * A0 / np.pi)


def mi(d, L):  return d - L * np.round(d / L)

def centroids(mesh):
    fid, N = mesh.face_id_by_edge, mesh.n_face
    cnt = np.bincount(fid, minlength=N).astype(float); cnt[cnt == 0] = np.nan
    c = np.stack([np.bincount(fid, weights=mesh.vertices[k], minlength=N) / cnt for k in (0, 1)], 1)
    return c, np.array([mesh.geometry.width, mesh.geometry.height])

def site_of(mesh, cell_ids):
    c, L = centroids(mesh); ref = c[cell_ids[0]]
    return ref + mi(c[cell_ids] - ref, L).mean(0)

def dist_to(mesh, site):
    c, L = centroids(mesh); return np.linalg.norm(mi(c - site, L), axis=1)


def measure2(hist, ids, disk_mask):
    N = len(hist[0]); is_mut = np.zeros(N, bool); is_mut[ids] = True
    nb0 = neighbour_mask(hist[0].mesh, is_mut); nb = nb0.copy(); out = defaultdict(list)
    for c in hist:
        m = c.mesh; area, per = m.area, m.perimeter; alive = area > 0; ma = is_mut & alive
        if ma.any(): nb = neighbour_mask(m, ma)
        p = np.full(N, np.nan); ok = alive & (per > 0); p[ok] = per[ok] / np.sqrt(area[ok])
        d = np.abs(np.bincount(m.face_id_by_edge, minlength=N) - 6).astype(float)
        mean = lambda a, mask: a[mask & alive].mean() if (mask & alive).any() else np.nan
        out['phi'].append(ma.sum() / len(ids)); out['p_glob'].append(np.nanmean(p[alive]))
        out['p_disk'].append(np.nanmean(p[disk_mask & alive]) if (disk_mask & alive).any() else np.nan)
        out['dhex_glob'].append(d[alive].mean()); out['dhex_disk'].append(mean(d, disk_mask))
        out['dhex_ring'].append(mean(d, nb0)); out['dhex_frozen'].append(mean(d, nb)); out['n_alive'].append(alive.sum())
    r = {k: np.array(v) for k, v in out.items()}; r['t_steps'] = np.arange(len(hist)) * SKIP; r['mutant_ids'] = np.asarray(ids)
    return r


def second_clone(base, seed_cell, sim_id, T=T2):
    cells = base.copy(); assert cells.mesh.area[seed_cell] > 0
    ids = custom.cluster_mutants(cells, N2 / len(cells), np.random.RandomState(10000 + sim_id), seed=int(seed_cell))
    assert len(ids) == N2, len(ids)
    grp = np.zeros(len(cells), int); grp[ids] = 1
    cells.properties.update(parent_group=grp, Gamma=np.array([GAMMA, GAMMA])[grp], Lambda=np.array([L_WT, L_MUT])[grp])
    hist = run(basic_simulation(cells, FORCE, T1_eps=EPS), T, SKIP)
    m0 = hist[0].mesh; site = site_of(m0, ids)
    is_mut = np.zeros(len(cells), bool); is_mut[ids] = True
    disk = (dist_to(m0, site) <= R_CLONE_DISK) & (m0.area > 0) & ~is_mut
    r = measure2(hist, ids, disk); r['disk_size'] = disk.sum(); r['seed_cell'] = seed_cell
    return r


def one_sim(task):
    sim_id, Dstar, conds, first_seed, out = task
    os.makedirs(out, exist_ok=True)
    rel, rs = relax(D0, sim_id, EPS)
    info = dict(sim_id=sim_id, first_seed=int(first_seed))
    if 'a' in conds or 'b' in conds:
        hist1, ids1 = challenge(rel, rs, 'cluster', F1, EPS, seed_cell=first_seed, n_steps=T_PRIME)
        primed = hist1[-1]; m = primed.mesh; N = len(primed)
        is1 = np.zeros(N, bool); is1[ids1] = True
        ring = np.where(neighbour_mask(hist1[0].mesh, is1) & (m.area > 0))[0]
        site1 = site_of(m, ring); dd = dist_to(m, site1); alive = m.area > 0
        n_mut_left = int((m.area[ids1] > 0).sum())
        al = np.where(alive)[0]
        p = m.perimeter[alive] / np.sqrt(m.area[alive]); dg = np.abs(np.bincount(m.face_id_by_edge, minlength=N) - 6)
        reg = alive & (dd <= R_CLONE1_DISK)
        info.update(n_mut_left=n_mut_left, n_alive=int(alive.sum()), p_glob_primed=float(p.mean()), dhex_glob_primed=float(dg[alive].mean()),
                    p_region1=float((m.perimeter[reg] / np.sqrt(m.area[reg])).mean()), dhex_region1=float(dg[reg].mean()),
                    box=[float(m.geometry.width), float(m.geometry.height)])
        if n_mut_left == 0:
            seeds = {'a': al[np.argmin(dd[al])], 'b': al[np.argmax(dd[al])]}
            for k in ('a', 'b'):
                if k in conds:
                    r = second_clone(primed, seeds[k], sim_id); np.savez_compressed(f"{out}/{k}_sim{sim_id}.npz", **r)
    if 'c' in conds:
        np.savez_compressed(f"{out}/c_sim{sim_id}.npz", **second_clone(rel, 64, sim_id))
    if 'd' in conds:
        relD, _ = relax(Dstar, sim_id, EPS)
        np.savez_compressed(f"{out}/d_sim{sim_id}.npz", **second_clone(relD, 64, sim_id))
    json.dump(info, open(f"{out}/info_sim{sim_id}.json", 'w'))
    return sim_id


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--sims', type=int, nargs=2, default=[0, 20]); ap.add_argument('--conds', default='abcd')
    ap.add_argument('--first-seed', default='64'); ap.add_argument('--out', default='../results/stage2')
    ap.add_argument('--dstar', type=float, default=None)
    a = ap.parse_args()
    Dstar = a.dstar if a.dstar is not None else json.load(open('../results/stage2_calibration.json'))['D_star']
    fs = lambda s: 64 if a.first_seed == '64' else int(np.random.RandomState(777 + s).randint(400))
    tasks = [(s, Dstar, a.conds, fs(s), a.out) for s in range(*a.sims)
             if not os.path.exists(f"{a.out}/info_sim{s}.json")]
    print('D* =', Dstar, ' tasks:', len(tasks), flush=True)
    with Pool(min(os.cpu_count(), len(tasks))) as pool:
        for s in pool.imap_unordered(one_sim, tasks): print('done sim', s, flush=True)
