import glob, os, sys, json
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from vm_core import t_x

root = sys.argv[1] if len(sys.argv) > 1 else '../results/stage1'
figdir = os.path.join(os.path.dirname(root), '..', 'figures'); os.makedirs(figdir, exist_ok=True)
rng = np.random.RandomState(0)
COL = {'random': '#6a9f1c', 'cluster': '#a3166e'}


def load(eps, mode, f):
    fs = sorted(glob.glob(os.path.join(root, f"eps{eps}", f"{mode}_f{f}_sim*.npz")))
    return [dict(np.load(x)) for x in fs]


def sem(a, axis=0):
    return np.nanstd(a, axis=axis, ddof=1) / np.sqrt(a.shape[axis])


def boot_ratio(a, b, n=5000):
    r = [np.nanmean(rng.choice(a, len(a))) / np.nanmean(rng.choice(b, len(b))) for _ in range(n)]
    return np.percentile(r, [2.5, 97.5])


summary = {}
for eps in sorted({os.path.basename(p)[3:] for p in glob.glob(os.path.join(root, 'eps*'))}):
    summary[eps] = {}
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, f in zip(axs, (0.05, 0.25)):
        ts = {}
        for mode in ('random', 'cluster'):
            runs = load(eps, mode, f)
            if not runs: continue
            t = runs[0]['t_steps']; phi = np.array([r['phi'] for r in runs])
            m, s = phi.mean(0), sem(phi)
            ax.plot(t, m, color=COL[mode], label=mode); ax.fill_between(t, m - s, m + s, color=COL[mode], alpha=.25)
            t50 = np.array([t_x(r['phi'], .5, t) for r in runs]); t5 = np.array([t_x(r['phi'], .05, t) for r in runs])
            ts[mode] = (t50, t5)
            summary[eps][f"{mode}_f{f}"] = dict(n=len(runs), t50_mean=float(np.nanmean(t50)), t50_sem=float(sem(t50[:, None], 0)[0]),
                                                  t5_mean=float(np.nanmean(t5)), t5_sem=float(sem(t5[:, None], 0)[0]), t5_n_reached=int((~np.isnan(t5)).sum()))
        if len(ts) == 2:
            ci = boot_ratio(ts['cluster'][0], ts['random'][0])
            summary[eps][f"ratio_t50_f{f}"] = dict(ratio=float(np.nanmean(ts['cluster'][0]) / np.nanmean(ts['random'][0])), ci95=[float(ci[0]), float(ci[1])])
        for y in (.5, .05): ax.axhline(y, ls='--', c='k', lw=.6)
        ax.set_xlim(0, 5000); ax.set_title(f"f_mut = {f}  (D=0.25, T1_eps={eps})", fontsize=9); ax.set_xlabel('time (steps)')
    axs[0].set_ylabel('phi_mut'); axs[0].legend(); fig.tight_layout(); fig.savefig(os.path.join(figdir, f'stage1_phi_eps{eps}.png'), dpi=150); plt.close(fig)

    fig, axs = plt.subplots(2, 2, figsize=(10, 6), sharex='col')
    for j, mode in enumerate(('random', 'cluster')):
        runs = load(eps, mode, 0.25)
        if not runs: continue
        t = runs[0]['t_steps']
        for i, (kl, kg, lab) in enumerate((('p_loc', 'p_glob', '<p>'), ('dhex_loc', 'dhex_glob', '<Dhex>'))):
            for k, ls, nm in ((kl, '-', 'local'), (kg, '--', 'global')):
                a = np.array([r[k] for r in runs]); m, s = np.nanmean(a, 0), sem(a)
                axs[i, j].plot(t, m, ls=ls, color=COL[mode], label=nm); axs[i, j].fill_between(t, m - s, m + s, color=COL[mode], alpha=.2)
            if i == 0: axs[i, j].axhline(3.81, c='k', lw=.6, ls=':')
            axs[i, j].set_ylabel(lab); axs[i, j].legend(fontsize=8)
        axs[0, j].set_title(f"{mode}, f_mut=0.25, T1_eps={eps}", fontsize=9)
        axs[1, j].set_xlabel('time (steps)')
        fin = {k: (float(np.nanmean([r[k][-1] for r in runs])), float(sem(np.array([[r[k][-1]] for r in runs]), 0)[0])) for k in
               ('p_loc', 'p_glob', 'dhex_loc', 'dhex_glob', 'n_wt_alive', 'phi')}
        summary[eps][f"final_{mode}_f0.25"] = fin
        at5 = {}
        for tt in (2000, 5000):
            ix = int(np.where(t == tt)[0][0]) if tt in t else None
            if ix is not None: at5[tt] = {k: float(np.nanmean([r[k][ix] for r in runs])) for k in ('p_loc', 'p_glob', 'dhex_loc', 'dhex_glob')}
        summary[eps][f"at_times_{mode}_f0.25"] = at5
    fig.tight_layout(); fig.savefig(os.path.join(figdir, f'stage1_p_dhex_eps{eps}.png'), dpi=150); plt.close(fig)

json.dump(summary, open(os.path.join(figdir, 'stage1_summary.json'), 'w'), indent=1)
print(json.dumps(summary, indent=1))
