import glob, json, os, sys
import numpy as np
from scipy.stats import mannwhitneyu
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from vm_core import t_x

root = sys.argv[1] if len(sys.argv) > 1 else '../results/stage2'
figdir = '../figures'; os.makedirs(figdir, exist_ok=True); rng = np.random.RandomState(1)
NAMES = {'a': 'PRIMED-SAME', 'b': 'PRIMED-FAR', 'c': 'NAIVE', 'd': 'MATCHED'}
COL = {'a': '#a3166e', 'b': '#d98bbb', 'c': '#3b7ea1', 'd': '#e08a1e'}
PSTAR = 3.81

def load(k): return {int(f.split('sim')[-1][:-4]): dict(np.load(f)) for f in sorted(glob.glob(f"{root}/{k}_sim*.npz"))}

def metrics(r):
    t, ph = r['t_steps'], r['phi']; f = lambda k: r[k][-1]
    return dict(t50=t_x(ph, .5, t), t5=t_x(ph, .05, t),
                R_glob=float(np.nansum(r['p_glob'] - r['p_glob'][0]) * 20), R_disk=float(np.nansum(r['p_disk'] - r['p_disk'][0]) * 20),
                T_above=float((r['p_glob'] > PSTAR).sum() * 20), peak_glob=float(np.nanmax(r['p_glob'])), peak_disk=float(np.nanmax(r['p_disk'])),
                p0_glob=float(r['p_glob'][0]), dhex_glob_end=float(f('dhex_glob')), dhex_disk_end=float(f('dhex_disk')),
                dhex_ring_end=float(f('dhex_ring')), dhex_frozen_end=float(f('dhex_frozen')))

def cliff(a, b): return float(np.mean([int(x > y) - int(x < y) for x in a for y in b]))
def boot(a, b, n=10000):
    d = [np.mean(rng.choice(a, len(a))) - np.mean(rng.choice(b, len(b))) for _ in range(n)]
    return np.percentile(d, [2.5, 97.5])

data = {k: load(k) for k in 'abcd'}; M = {k: {s: metrics(r) for s, r in v.items()} for k, v in data.items()}
infos = [json.load(open(f)) for f in sorted(glob.glob(f"{root}/info_sim*.json"))]
print('realisations with info:', len(infos), '| excluded (mutant left in phase 1):', [i['sim_id'] for i in infos if i.get('n_mut_left', 0) > 0])
if infos and 'p_glob_primed' in infos[0]:
    print('primed residual: <p>_glob %.3f, <Dhex>_glob %.3f, N alive %.0f, region1 <p> %.3f <Dhex> %.3f' % tuple(
        np.mean([i[k] for i in infos if 'p_glob_primed' in i]) for k in ('p_glob_primed', 'dhex_glob_primed', 'n_alive', 'p_region1', 'dhex_region1')))

keys = ['t50', 't5', 'R_glob', 'R_disk', 'T_above', 'peak_glob', 'peak_disk', 'p0_glob', 'dhex_glob_end', 'dhex_disk_end', 'dhex_ring_end', 'dhex_frozen_end']
print('\n%-16s' % 'metric' + ''.join('%-24s' % f'{k} {NAMES[k]}' for k in 'abcd'))
summary = {}
for key in keys:
    row = {}
    for k in 'abcd':
        v = np.array([m[key] for m in M[k].values()], float); row[k] = (float(np.nanmean(v)), float(np.nanstd(v, ddof=1) / np.sqrt(np.isfinite(v).sum())) if np.isfinite(v).sum() > 1 else np.nan, int(np.isfinite(v).sum()))
    summary[key] = row
    print('%-16s' % key + ''.join('%-24s' % ('%.3g ± %.2g (n=%d)' % row[k]) for k in 'abcd'))

tests = []
for (x, y) in (('a', 'c'), ('a', 'd'), ('a', 'b'), ('d', 'c')):
    for key in keys:
        A = np.array([m[key] for m in M[x].values()], float); B = np.array([m[key] for m in M[y].values()], float)
        A, B = A[np.isfinite(A)], B[np.isfinite(B)]
        if len(A) < 3 or len(B) < 3: continue
        u = mannwhitneyu(A, B, alternative='two-sided'); lo, hi = boot(A, B)
        tests.append(dict(contrast=f'{x}-{y}', metric=key, diff=float(A.mean() - B.mean()), ci=[float(lo), float(hi)], cliff=cliff(A, B), p=float(u.pvalue), nA=len(A), nB=len(B)))
prim = [t for t in tests if t['contrast'] in ('a-c', 'a-d') and t['metric'] in ('t50', 'R_glob')]
order = np.argsort([t['p'] for t in prim]); m = len(prim)
run = 0
for rank, i in enumerate(order):
    run = max(run, min(1, (m - rank) * prim[i]['p'])); prim[i]['p_holm'] = run
print('\nPRIMARY (pre-registered), Holm over 4 tests:')
for t in prim: print('  %s %-7s diff=%.4g CI95=[%.3g, %.3g] Cliff=%.2f p=%.3g p_Holm=%.3g (nA=%d nB=%d)' % (t['contrast'], t['metric'], t['diff'], *t['ci'], t['cliff'], t['p'], t['p_holm'], t['nA'], t['nB']))
print('\nSECONDARY (unadjusted p):')
for t in tests:
    if t not in prim: print('  %s %-14s diff=%.4g CI95=[%.3g, %.3g] Cliff=%.2f p=%.3g' % (t['contrast'], t['metric'], t['diff'], *t['ci'], t['cliff'], t['p']))
json.dump(dict(summary=summary, tests=tests), open(f'{figdir}/stage2_summary.json', 'w'), indent=1)

fig, axs = plt.subplots(1, 3, figsize=(14, 3.8))
for k in 'abcd':
    if not data[k]: continue
    t = next(iter(data[k].values()))['t_steps']
    for ax, key in zip(axs, ('phi', 'p_glob', 'p_disk')):
        A = np.array([r[key] for r in data[k].values()]); mu = np.nanmean(A, 0); se = np.nanstd(A, 0, ddof=1) / np.sqrt(len(A))
        ax.plot(t, mu, color=COL[k], label=f'{k}) {NAMES[k]} (n={len(A)})'); ax.fill_between(t, mu - se, mu + se, color=COL[k], alpha=.2)
for ax, ttl in zip(axs, ('phi_mut (second clone)', '<p> global', '<p> disk (r=3.65)')): ax.set_title(ttl, fontsize=10); ax.set_xlabel('steps after second clone')
axs[1].axhline(PSTAR, c='k', ls=':', lw=.7); axs[2].axhline(PSTAR, c='k', ls=':', lw=.7); axs[0].set_xlim(0, 3000); axs[0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(f'{figdir}/stage2_timeseries.png', dpi=150); plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
for ax, key in zip(axs, ('t50', 'R_glob')):
    for i, k in enumerate('abcd'):
        v = np.array([m[key] for m in M[k].values()], float); ax.scatter(np.full(len(v), i) + rng.uniform(-.12, .12, len(v)), v, s=18, color=COL[k], alpha=.8)
        ax.hlines(np.nanmean(v), i - .25, i + .25, color='k')
    ax.set_xticks(range(4)); ax.set_xticklabels([NAMES[k] for k in 'abcd'], rotation=20, fontsize=8); ax.set_title(key + ' (primary)', fontsize=10)
fig.tight_layout(); fig.savefig(f'{figdir}/stage2_primary.png', dpi=150)
