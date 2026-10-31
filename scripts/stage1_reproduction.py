import argparse, os, time
import numpy as np
from multiprocessing import Pool
from vm_core import relax, challenge, measure

D = 0.25
FS = (0.05, 0.25)
MODES = ('random', 'cluster')


def work(task):
    eps, sim_id, out = task
    t0 = time.time()
    relaxed, rstate = relax(D, sim_id, eps)
    for mode in MODES:
        for f in FS:
            fn = os.path.join(out, f"eps{eps}", f"{mode}_f{f}_sim{sim_id}.npz")
            if os.path.exists(fn):
                continue
            hist, ids = challenge(relaxed, rstate, mode, f, eps)
            res = measure(hist, ids)
            os.makedirs(os.path.dirname(fn), exist_ok=True)
            np.savez_compressed(fn, **res)
    return eps, sim_id, time.time() - t0


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--eps', type=float, nargs='+', default=[0.04, 0.01])
    ap.add_argument('--nsim', type=int, default=10)
    ap.add_argument('--out', default='results/stage1')
    a = ap.parse_args()
    tasks = [(e, s, a.out) for e in a.eps for s in range(a.nsim)]
    with Pool(min(os.cpu_count(), len(tasks))) as pool:
        for e, s, dt in pool.imap_unordered(work, tasks):
            print(f"done eps={e} sim={s}  {dt:.1f}s", flush=True)
